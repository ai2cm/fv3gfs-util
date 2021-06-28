from dataclasses import dataclass
from .buffer import Buffer
from .quantity import Quantity
from .types import NumpyModule
from typing import List, Tuple, Optional, Dict
from enum import Enum
from .rotate import rotate_scalar_data, rotate_vector_data
from .utils import device_synchronize, flatten
import numpy as np
from uuid import UUID, uuid1
from .cuda_kernels import (
    pack_scalar_f64_kernel,
    pack_vector_f64_kernel,
    unpack_scalar_f64_kernel,
    unpack_vector_f64_kernel,
)

try:
    import cupy as cp
except ImportError:
    cp = None

# Temporary "safe" code path
#  _CODE_PATH_DEVICE_WIDE_SYNC: turns off streaming and issue a single
#   device wide synchronization call instead
_CODE_PATH_DEVICE_WIDE_SYNC = False

# Keyed cached - key is a str at the moment to go around the fact that
# a slice is not hashable
# getting a string from Tuple(slices, rotation, shape, strides, itemsize)
# IndicesKey = str(Tuple[Any, int, Tuple[int], Tuple[int], int]) #noqa
INDICES_GPU_CACHE: Dict[str, "cp.ndarray"] = {}

# Simple pool of streams
STREAM_POOL: List["cp.cuda.Stream"] = []


def _pop_stream() -> "cp.cuda.Stream":
    if len(STREAM_POOL) == 0:
        return cp.cuda.Stream(non_blocking=True)
    return STREAM_POOL.pop()


def _push_stream(stream: "cp.cuda.Stream"):
    STREAM_POOL.append(stream)


def _slices_length(slices: Tuple[slice]) -> int:
    """Compute linear size from slices"""
    length = 1
    for s in slices:
        assert s.step is None
        length *= abs(s.start - s.stop)
    return length


@dataclass
class MessageMetadata:
    """Description of a single message."""

    _id: UUID
    quantity: Quantity
    send_slices: Tuple[slice]
    send_clockwise_rotation: int
    recv_slices: Tuple[slice]


class MessageBundleType(Enum):
    """Dimensionality of the message bundle."""

    UNKNOWN = 0
    SCALAR = 1
    VECTOR = 2


class MessageBundle:
    """Pack/unpack multiple nD array into/from a single buffer.

    The class is responsible for packing & unpacking, not communication.
    Order of operations:
    - get MessageBundle via get_from_quantity_module
    (user should re-use the same if it goes to the same destination)
    - queue N messages via queue_scalar & queue_vector
    - allocate (WARNING: do not queue after allocation!)
    [From here get_recv_buffer() is valid]
    - async_pack
    - synchronize
    [From here get_send_buffer() is valid]
    ... user should communicate the buffers...
    - async_unpack
    - synchronize
    [All buffers return to their buffer cache, therefore invalid]
    """

    _send_buffer: Optional[Buffer] = None
    _recv_buffer: Optional[Buffer] = None

    _x_infos: List[MessageMetadata]
    _y_infos: List[MessageMetadata]

    _type: MessageBundleType = MessageBundleType.UNKNOWN

    def __init__(self, to_rank: int, np_module: NumpyModule) -> None:
        self.to_rank = to_rank
        self._np_module = np_module
        self._x_infos = []
        self._y_infos = []
        assert self._send_buffer is None
        assert self._recv_buffer is None

    def __del__(self):
        assert self._send_buffer is None
        assert self._recv_buffer is None

    @staticmethod
    def get_from_quantity_module(
        to_rank: int, qty_module: NumpyModule
    ) -> "MessageBundle":
        if qty_module is np:
            return MessageBundleCPU(to_rank, np)
        elif qty_module is cp:
            return MessageBundleGPU(to_rank, cp)

        raise NotImplementedError(
            f"Quantity module {qty_module} has no MessageBundle implemented"
        )

    def queue_scalar_message(
        self,
        quantity: Quantity,
        send_slice: Tuple[slice],
        n_clockwise_rotation: int,
        recv_slice: Tuple[slice],
    ) -> None:
        """Add message information to the bundle.

        Send array & rotation parameter for later packing.
        Recv array for later unpacking. Pop a recv Buffer for transfer.
        Recv buffer is held by the class.
        The function will also compute & store the slicing of the arrays.
        """
        assert (
            self._type == MessageBundleType.SCALAR
            or self._type == MessageBundleType.UNKNOWN
        )
        if self._recv_buffer is not None and self._send_buffer is not None:
            raise RuntimeError(
                "Buffer previously allocated are not longer correct."
                "Make sure to queue all message before calling allocate."
            )

        self._type = MessageBundleType.SCALAR
        self._x_infos.append(
            MessageMetadata(
                _id=uuid1(),
                quantity=quantity,
                send_slices=send_slice,
                send_clockwise_rotation=n_clockwise_rotation,
                recv_slices=recv_slice,
            )
        )

    def queue_vector_message(
        self,
        x_quantity: Quantity,
        x_send_slice: Tuple[slice],
        y_quantity: Quantity,
        y_send_slice: Tuple[slice],
        n_clockwise_rotation: int,
        x_recv_slice: Tuple[slice],
        y_recv_slice: Tuple[slice],
    ) -> None:
        assert (
            self._type == MessageBundleType.VECTOR
            or self._type == MessageBundleType.UNKNOWN
        )
        if self._recv_buffer is not None and self._send_buffer is not None:
            raise RuntimeError(
                "Buffer previously allocated are not longer correct."
                "Make sure to queue all message before calling allocate."
            )
        self._type = MessageBundleType.VECTOR
        self._x_infos.append(
            MessageMetadata(
                _id=uuid1(),
                quantity=x_quantity,
                send_slices=x_send_slice,
                send_clockwise_rotation=n_clockwise_rotation,
                recv_slices=x_recv_slice,
            )
        )
        self._y_infos.append(
            MessageMetadata(
                _id=uuid1(),
                quantity=y_quantity,
                send_slices=y_send_slice,
                send_clockwise_rotation=n_clockwise_rotation,
                recv_slices=y_recv_slice,
            )
        )

    def get_recv_buffer(self):
        if self._recv_buffer is None:
            raise RuntimeError("Recv buffer can't be retrieved before allocate()")
        return self._recv_buffer

    def get_send_buffer(self):
        if self._send_buffer is None:
            raise RuntimeError("Send buffer can't be retrieved before allocate()")
        return self._send_buffer

    def allocate(self):
        # Compute required size
        buffer_size = 0
        dtype = None
        for x_info in self._x_infos:
            buffer_size += _slices_length(x_info.send_slices)
            dtype = x_info.quantity.metadata.dtype
        if self._type is MessageBundleType.VECTOR:
            for y_info in self._y_infos:
                buffer_size += _slices_length(y_info.send_slices)

        # Demand a properly size buffers
        self._send_buffer = Buffer.pop_from_cache(
            self._np_module.empty, (buffer_size), dtype
        )
        self._recv_buffer = Buffer.pop_from_cache(
            self._np_module.empty, (buffer_size), dtype
        )

    def ready(self):
        return self._send_buffer is not None and self._recv_buffer is not None

    def async_pack(self):
        """Pack all queued messages into a single send Buffer.

        Implementation should guarantee the send Buffer is ready for MPI communication.
        The send Buffer is held by this class."""
        raise NotImplementedError()

    def async_unpack(self):
        """Unpack the buffer into destination arrays.

        Implementation DOESN'T HAVE TO guarantee transfer is done."""
        raise NotImplementedError()

    def finalize(self):
        """Finalize operations after unpack.

        Implementation should guarantee all transfers are done.
        Implementation should return recv & send Buffers to the cache for reuse."""
        raise NotImplementedError()

    def synchronize(self):
        raise NotImplementedError()


class MessageBundleCPU(MessageBundle):
    def synchronize(self):
        # CPU doesn't do true async (for now)
        pass

    def async_pack(self):
        # Unpack per type
        if self._type == MessageBundleType.SCALAR:
            self._pack_scalar()
        elif self._type == MessageBundleType.VECTOR:
            self._pack_vector()
        else:
            raise RuntimeError(f"Unimplemented {self._type} message pack")

        self._send_buffer.finalize_memory_transfer()

    def _pack_scalar(self):
        offset = 0
        for x_info in self._x_infos:
            message_size = _slices_length(x_info.send_slices)
            # sending data across the boundary will rotate the data
            # n_clockwise_rotations times, due to the difference in axis orientation.\
            # Thus we rotate that number of times counterclockwise before sending,
            # to get the right final orientation
            source_view = rotate_scalar_data(
                x_info.quantity.data[x_info.send_slices],
                x_info.quantity.dims,
                x_info.quantity.np,
                -x_info.send_clockwise_rotation,
            )
            self._send_buffer.assign_from(
                flatten(source_view),
                buffer_slice=np.index_exp[offset : offset + message_size],
            )
            offset += message_size

    def _pack_vector(self):
        assert len(self._x_infos) == len(self._y_infos)
        offset = 0
        for x_info, y_info in zip(self._x_infos, self._y_infos):
            # sending data across the boundary will rotate the data
            # n_clockwise_rotations times, due to the difference in axis orientation
            # Thus we rotate that number of times counterclockwise before sending,
            # to get the right final orientation
            x_view, y_view = rotate_vector_data(
                x_info.quantity.data[x_info.send_slices],
                y_info.quantity.data[y_info.send_slices],
                -x_info.send_clockwise_rotation,
                x_info.quantity.dims,
                x_info.quantity.np,
            )

            # Pack X/Y messages in the buffer
            self._send_buffer.assign_from(
                flatten(x_view),
                buffer_slice=np.index_exp[offset : offset + x_view.size],
            )
            offset += x_view.size
            self._send_buffer.assign_from(
                flatten(y_view),
                buffer_slice=np.index_exp[offset : offset + y_view.size],
            )
            offset += y_view.size

    def async_unpack(self):
        # Unpack per type
        if self._type == MessageBundleType.SCALAR:
            self._unpack_scalar()
        elif self._type == MessageBundleType.VECTOR:
            self._unpack_vector()
        else:
            raise RuntimeError(f"Unimplemented {self._type} message unpack")

        self._recv_buffer.finalize_memory_transfer()

        # Pop the buffers back in the cache
        Buffer.push_to_cache(self._send_buffer)
        self._send_buffer = None
        Buffer.push_to_cache(self._recv_buffer)
        self._recv_buffer = None

    def _unpack_scalar(self):
        offset = 0
        for x_info in self._x_infos:
            quantity_view = x_info.quantity.data[x_info.recv_slices]
            message_size = _slices_length(x_info.recv_slices)
            self._recv_buffer.assign_to(
                quantity_view,
                buffer_slice=np.index_exp[offset : offset + message_size],
                buffer_reshape=quantity_view.shape,
            )
            offset += message_size

    def _unpack_vector(self):
        offset = 0
        for x_info, y_info in zip(self._x_infos, self._y_infos):
            quantity_view = x_info.quantity.data[x_info.recv_slices]
            message_size = _slices_length(x_info.recv_slices)
            self._recv_buffer.assign_to(
                quantity_view,
                buffer_slice=np.index_exp[offset : offset + message_size],
                buffer_reshape=quantity_view.shape,
            )
            offset += message_size
            quantity_view = y_info.quantity.data[y_info.recv_slices]
            message_size = _slices_length(y_info.recv_slices)
            self._recv_buffer.assign_to(
                quantity_view,
                buffer_slice=np.index_exp[offset : offset + message_size],
                buffer_reshape=quantity_view.shape,
            )
            offset += message_size

    def finalize(self):
        pass


class MessageBundleGPU(MessageBundle):
    @dataclass
    class CuKernelArgs:
        """All arguments requireds for the CUDA kernels."""

        stream: "cp.cuda.Stream"
        x_send_indices: "cp.ndarray"
        x_recv_indices: "cp.ndarray"
        y_send_indices: Optional["cp.ndarray"]
        y_recv_indices: Optional["cp.ndarray"]

    def __init__(self, to_rank: int, np_module: NumpyModule) -> None:
        super().__init__(to_rank, np_module)
        self._cu_kernel_args: Dict[UUID, MessageBundleGPU.CuKernelArgs] = {}

    def _build_flatten_indices(
        self,
        key,
        shape_of_sliced_data: Tuple[int],
        dims,
        strides: Tuple[int],
        itemsize: int,
        slices: Tuple[slice],
        rotate: bool,
        rotation: int,
    ):
        # Have to go down to numpy to leverage indices calculation
        arr_indices = np.zeros(shape_of_sliced_data, dtype=np.int32, order="C")

        # Get offset from first index
        offset_dims = []
        for s in slices:
            offset_dims.append(s.start)
        offset_to_slice = sum(np.array(offset_dims) * strides) // itemsize

        # Flatten the index into an indices array
        with np.nditer(
            arr_indices, flags=["multi_index"], op_flags=["writeonly"], order="K",
        ) as it:
            for array_value in it:
                offset = sum(np.array(it.multi_index) * strides) // itemsize
                array_value[...] = offset_to_slice + offset

        if rotate:
            # sending data across the boundary will rotate the data
            # n_clockwise_rotations times, due to the difference in axis orientation.
            # Thus we rotate that number of times counterclockwise before sending,
            # to get the right final orientation. We apply those rotations to the
            # indices here to prepare for a straightforward copy in cu kernel
            arr_indices = rotate_scalar_data(arr_indices, dims, cp, -rotation)
        arr_indices_gpu = cp.asarray(arr_indices.flatten(order="C"))
        INDICES_GPU_CACHE[key] = arr_indices_gpu

    def _flatten_indices(
        self, message_info: MessageMetadata, slices: Tuple[slice], rotate: bool
    ) -> "cp.ndarray":
        """Extract a flat array of indices for the send message.

        Also take care of rotating the indices to account for axis orientation
        """
        strides = message_info.quantity.data.strides
        itemsize = message_info.quantity.data.itemsize
        shape = message_info.quantity.data.shape
        key = str(
            (slices, message_info.send_clockwise_rotation, shape, strides, itemsize)
        )

        if key not in INDICES_GPU_CACHE.keys():
            self._build_flatten_indices(
                key,
                message_info.quantity.data[slices].shape,
                message_info.quantity.dims,
                strides,
                itemsize,
                slices,
                rotate,
                message_info.send_clockwise_rotation,
            )

        # We don't return a copy since the indices are read-only in the algorithm
        return INDICES_GPU_CACHE[key]

    def allocate(self):
        # Super to get buffer allocation
        super().allocate()
        # Allocate the streams & build the indices arrays
        if len(self._y_infos) == 0:
            for x_info in self._x_infos:
                self._cu_kernel_args[x_info._id] = MessageBundleGPU.CuKernelArgs(
                    stream=_pop_stream(),
                    x_send_indices=self._flatten_indices(
                        x_info, x_info.send_slices, True
                    ),
                    x_recv_indices=self._flatten_indices(
                        x_info, x_info.recv_slices, False
                    ),
                    y_send_indices=None,
                    y_recv_indices=None,
                )
        else:
            for x_info, y_info in zip(self._x_infos, self._y_infos):
                self._cu_kernel_args[x_info._id] = MessageBundleGPU.CuKernelArgs(
                    stream=_pop_stream(),
                    x_send_indices=self._flatten_indices(
                        x_info, x_info.send_slices, True
                    ),
                    x_recv_indices=self._flatten_indices(
                        x_info, x_info.recv_slices, False
                    ),
                    y_send_indices=self._flatten_indices(
                        y_info, y_info.send_slices, True
                    ),
                    y_recv_indices=self._flatten_indices(
                        y_info, y_info.recv_slices, False
                    ),
                )

    def synchronize(self):
        if _CODE_PATH_DEVICE_WIDE_SYNC:
            self._safe_synchronize()
        else:
            self._streamed_synchronize()

    def _streamed_synchronize(self):
        for cu_kernel in self._cu_kernel_args.values():
            cu_kernel.stream.synchronize()

    def _safe_synchronize(self):
        device_synchronize()

    def _use_stream(self, stream):
        if not _CODE_PATH_DEVICE_WIDE_SYNC:
            stream.use()

    def async_pack(self):
        # Unpack per type
        if self._type == MessageBundleType.SCALAR:
            self._opt_pack_scalar()
        elif self._type == MessageBundleType.VECTOR:
            self._opt_pack_vector()
        else:
            raise RuntimeError(f"Unimplemented {self._type} message pack")

    def _opt_pack_scalar(self):
        offset = 0
        for x_info in self._x_infos:
            cu_kernel_args = self._cu_kernel_args[x_info._id]

            # Use private stream
            self._use_stream(cu_kernel_args.stream)

            # Message size
            message_size = _slices_length(x_info.send_slices)

            if x_info.quantity.metadata.dtype != np.float64:
                raise RuntimeError(f"Kernel requires f64 given {np.float64}")

            # Launch kernel
            blocks = 128
            grid_x = (message_size // blocks) + 1
            pack_scalar_f64_kernel(
                (blocks,),
                (grid_x,),
                (
                    x_info.quantity.data[:],  # source_array
                    cu_kernel_args.x_send_indices,  # indices
                    message_size,  # nIndex
                    offset,
                    self._send_buffer.array,
                ),
            )

            # Next message offset into send buffer
            offset += message_size

    def _opt_pack_vector(self):
        assert len(self._x_infos) == len(self._y_infos)
        offset = 0
        for x_info, y_info in zip(self._x_infos, self._y_infos):
            cu_kernel_args = self._cu_kernel_args[x_info._id]

            # Use private stream
            self._use_stream(cu_kernel_args.stream)

            # Message sizes
            message_size_x = _slices_length(x_info.send_slices)
            message_size_y = _slices_length(y_info.send_slices)
            message_size = message_size_x + message_size_y

            if x_info.quantity.metadata.dtype != np.float64:
                raise RuntimeError(f"Kernel requires f64 given {np.float64}")

            # Launch kernel
            blocks = 128
            grid_x = (message_size // blocks) + 1
            pack_vector_f64_kernel(
                (blocks,),
                (grid_x,),
                (
                    x_info.quantity.data[:],  # source_array_x
                    y_info.quantity.data[:],  # source_array_y
                    cu_kernel_args.x_send_indices,  # indices_x
                    cu_kernel_args.y_send_indices,  # indices_y
                    message_size_x,  # nIndex_x
                    message_size_y,  # nIndex_y
                    offset,
                    (-x_info.send_clockwise_rotation) % 4,  # rotation
                    self._send_buffer.array,
                ),
            )

            # Next message offset into send buffer
            offset += message_size

    def async_unpack(self):
        # Unpack per type
        if self._type == MessageBundleType.SCALAR:
            self._opt_unpack_scalar()
        elif self._type == MessageBundleType.VECTOR:
            self._opt_unpack_vector()
        else:
            raise RuntimeError(f"Unimplemented {self._type} message unpack")

    def finalize(self):
        # Synchronize all work
        self.synchronize()

        # Push the buffers back in the cache
        Buffer.push_to_cache(self._send_buffer)
        self._send_buffer = None
        Buffer.push_to_cache(self._recv_buffer)
        self._recv_buffer = None

        # Push the streams back in the pool
        for cu_info in self._cu_kernel_args.values():
            _push_stream(cu_info.stream)

    def _cupy_unpack_scalar(self):
        offset = 0
        for x_info in self._x_infos:
            quantity_view = x_info.quantity.data[x_info.recv_slices]
            message_size = _slices_length(x_info.recv_slices)
            self._recv_buffer.assign_to(
                quantity_view,
                buffer_slice=np.index_exp[offset : offset + message_size],
                buffer_reshape=quantity_view.shape,
            )
            offset += message_size

    def _opt_unpack_scalar(self):
        offset = 0
        for x_info in self._x_infos:
            # Use private stream
            kernel_args = self._cu_kernel_args[x_info._id]
            self._use_stream(kernel_args.stream)

            # Launch kernel
            message_size = _slices_length(x_info.recv_slices)
            blocks = 128
            grid_x = (message_size // blocks) + 1
            unpack_scalar_f64_kernel(
                (blocks,),
                (grid_x,),
                (
                    self._recv_buffer.array,  # source_buffer
                    kernel_args.x_recv_indices,  # indices
                    message_size,  # nIndex
                    offset,
                    x_info.quantity.data[:],  # destination_array
                ),
            )

            # Next message offset into recv buffer
            offset += message_size

    def _opt_unpack_vector(self):
        assert len(self._x_infos) == len(self._y_infos)
        offset = 0
        for x_info, y_info in zip(self._x_infos, self._y_infos):
            # We only have writte a f64 kernel
            if x_info.quantity.metadata.dtype != np.float64:
                raise RuntimeError(f"Kernel requires f64 given {np.float64}")

            # Use private stream
            cu_kernel_args = self._cu_kernel_args[x_info._id]
            self._use_stream(cu_kernel_args.stream)

            # Message sizes
            message_size_x = _slices_length(x_info.recv_slices)
            message_size_y = _slices_length(y_info.recv_slices)
            message_size = message_size_x + message_size_y

            # Launch kernel
            blocks = 128
            grid_x = (message_size // blocks) + 1
            unpack_vector_f64_kernel(
                (blocks,),
                (grid_x,),
                (
                    self._recv_buffer.array,
                    cu_kernel_args.x_recv_indices,  # indices_x
                    cu_kernel_args.y_recv_indices,  # indices_y
                    message_size_x,  # nIndex_x
                    message_size_y,  # nIndex_y
                    offset,
                    x_info.quantity.data[:],  # destination_array_x
                    y_info.quantity.data[:],  # destination_array_y
                ),
            )

            # Next message offset into send buffer
            offset += message_size
