from dataclasses import dataclass
from .buffer import Buffer
from .quantity import Quantity
from .types import NumpyModule
from typing import List, Tuple, Optional, Dict, Any
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


@dataclass
class HaloUpdateSpec:
    strides: Tuple[int]
    itemsize: int
    shape: Tuple[int]
    origin: Tuple[int, ...]
    extent: Tuple[int, ...]
    dims: Tuple[str, ...]
    numpy_module: NumpyModule
    dtype: Any


# ------------------------------------------------------------------------
# Temporary "safe" code path
#  _CODE_PATH_DEVICE_WIDE_SYNC: turns off streaming and issue a single
#   device wide synchronization call instead

_CODE_PATH_DEVICE_WIDE_SYNC = False

# ------------------------------------------------------------------------
# Simple pool of streams to lower the driver pressure
# Use _pop/_push_stream to manipulate the pool

STREAM_POOL: List["cp.cuda.Stream"] = []


def _pop_stream() -> "cp.cuda.Stream":
    if len(STREAM_POOL) == 0:
        return cp.cuda.Stream(non_blocking=True)
    return STREAM_POOL.pop()


def _push_stream(stream: "cp.cuda.Stream"):
    STREAM_POOL.append(stream)


# ------------------------------------------------------------------------
# Packed buffer description & helpers

# Keyed cached - key is a str at the moment to go around the fact that
# a slice is not hashable. getting a string from
# Tuple(slices, rotation, shape, strides, itemsize) e.g. # noqa
# str(Tuple[Any, int, Tuple[int], Tuple[int], int]) # noqa
INDICES_GPU_CACHE: Dict[str, "cp.ndarray"] = {}


def _slices_length(slices: Tuple[slice]) -> int:
    """Compute linear size from slices."""
    length = 1
    for s in slices:
        assert s.step is None
        length *= abs(s.start - s.stop)
    return length


@dataclass
class _ExchangeDataDescription:
    """Description of a the data exchanged."""

    _id: UUID
    specification: HaloUpdateSpec
    send_slices: Tuple[slice]
    send_clockwise_rotation: int
    recv_slices: Tuple[slice]
    send_buffer_size: int
    recv_buffer_size: int


class _HaloDataTransformerType(Enum):
    """Dimensionality of the data in the packed buffer."""

    UNKNOWN = 0
    SCALAR = 1
    VECTOR = 2


# ------------------------------------------------------------------------
# HaloDataTransformer classes


class HaloDataTransformer:
    """Transform data to exchange in a format optimized for network communication.

    Current strategy: pack/unpack multiple nD array into/from a single buffer.
    Offers a send and a recv buffer to use for exchanging data.

    The class is responsible for packing & unpacking, not communication.
    Order of operations:
    - get HaloDataTransformer via get_from_numpy_module
    (user should re-use the same if it goes to the same destination)
    - queue N packed_buffers via queue_scalar & queue_vector (can't mix and match scalar & vector)
    - allocate (WARNING: do not queue after allocation!)
    [From here get_recv_buffer() is valid]
    - async_pack
    - synchronize
    [From here get_send_buffer() is valid]
    [... user should communicate the buffers...]
    - async_unpack
    - finalize
    [All buffers are returned to their buffer cache, therefore invalid]

    If the user doesn't want the buffer return to the cache (optimal behavior if the same exchange
    is to be done multiple times), it can use `synchronize` instead of `finalize`
    """

    _send_buffer: Optional[Buffer]
    _recv_buffer: Optional[Buffer]

    _x_infos: List[_ExchangeDataDescription]
    _y_infos: List[_ExchangeDataDescription]

    def __init__(self, np_module: NumpyModule) -> None:
        """Init routine.

        Arguments:
            np_module: numpy-like module for allocation
        """
        self._type = _HaloDataTransformerType.UNKNOWN
        self._np_module = np_module
        self._x_infos = []
        self._y_infos = []
        self._send_buffer = None
        self._recv_buffer = None

    def __del__(self):
        """Del routine, making sure all buffers were inserted back into cache."""
        assert self._send_buffer is None
        assert self._recv_buffer is None

    @staticmethod
    def get_from_numpy_module(np_module: NumpyModule) -> "HaloDataTransformer":
        """Construct a module from a numpy-like module.

        Arguments:
            np_module: numpy-like module to determin child packed_buffer type.

        Return: an initialized packed buffer.
        """
        if np_module is np:
            return HaloDataTransformerCPU(np)
        elif np_module is cp:
            return HaloDataTransformerGPU(cp)

        raise NotImplementedError(
            f"Quantity module {np_module} has no HaloDataTransformer implemented"
        )

    def queue_scalar(
        self,
        specification: HaloUpdateSpec,
        send_slice: Tuple[slice],
        n_clockwise_rotation: int,
        recv_slice: Tuple[slice],
    ):
        """Add packed_buffer scalar packing information to the bundle."""
        assert (
            self._type == _HaloDataTransformerType.SCALAR
            or self._type == _HaloDataTransformerType.UNKNOWN
        )
        if self._recv_buffer is not None and self._send_buffer is not None:
            raise RuntimeError(
                "Buffer previously allocated are not longer correct."
                "Make sure to queue all packed_buffer before calling allocate."
            )

        self._type = _HaloDataTransformerType.SCALAR
        self._x_infos.append(
            _ExchangeDataDescription(
                _id=uuid1(),
                send_slices=send_slice,
                send_clockwise_rotation=n_clockwise_rotation,
                recv_slices=recv_slice,
                send_buffer_size=_slices_length(send_slice),
                recv_buffer_size=_slices_length(recv_slice),
                specification=specification,
            )
        )

    def queue_vector(
        self,
        specification_x: HaloUpdateSpec,
        specification_y: HaloUpdateSpec,
        x_send_slice: Tuple[slice],
        y_send_slice: Tuple[slice],
        n_clockwise_rotation: int,
        x_recv_slice: Tuple[slice],
        y_recv_slice: Tuple[slice],
    ):
        """Add packed_buffer vector packing information to the bundle."""
        assert (
            self._type == _HaloDataTransformerType.VECTOR
            or self._type == _HaloDataTransformerType.UNKNOWN
        )
        if self._recv_buffer is not None and self._send_buffer is not None:
            raise RuntimeError(
                "Buffer previously allocated are not longer correct."
                "Make sure to queue all packed_buffer before calling allocate."
            )
        self._type = _HaloDataTransformerType.VECTOR
        self._x_infos.append(
            _ExchangeDataDescription(
                _id=uuid1(),
                send_slices=x_send_slice,
                send_clockwise_rotation=n_clockwise_rotation,
                recv_slices=x_recv_slice,
                send_buffer_size=_slices_length(x_send_slice),
                recv_buffer_size=_slices_length(x_recv_slice),
                specification=specification_x,
            )
        )
        self._y_infos.append(
            _ExchangeDataDescription(
                _id=uuid1(),
                send_slices=y_send_slice,
                send_clockwise_rotation=n_clockwise_rotation,
                recv_slices=y_recv_slice,
                send_buffer_size=_slices_length(y_send_slice),
                recv_buffer_size=_slices_length(y_recv_slice),
                specification=specification_y,
            )
        )

    def get_recv_buffer(self) -> Buffer:
        """Retrieve receive buffer.

        WARNING: Only available _after_ `allocate` as been called.
        """
        if self._recv_buffer is None:
            raise RuntimeError("Recv buffer can't be retrieved before allocate()")
        return self._recv_buffer

    def get_send_buffer(self) -> Buffer:
        """Retrieve send buffer.

        WARNING: Only available _after_ `allocate` as been called.
        """
        if self._send_buffer is None:
            raise RuntimeError("Send buffer can't be retrieved before allocate()")
        return self._send_buffer

    def allocate(self):
        """Allocate contiguous memory buffers from description queued."""

        # Compute required size
        buffer_size = 0
        dtype = None
        for x_info in self._x_infos:
            buffer_size += _slices_length(x_info.send_slices)
            dtype = x_info.specification.dtype
        if self._type is _HaloDataTransformerType.VECTOR:
            for y_info in self._y_infos:
                buffer_size += _slices_length(y_info.send_slices)

        # Retrieve two properly sized buffers
        self._send_buffer = Buffer.pop_from_cache(
            self._np_module.empty, (buffer_size), dtype
        )
        self._recv_buffer = Buffer.pop_from_cache(
            self._np_module.empty, (buffer_size), dtype
        )

    def ready(self) -> bool:
        """Check if the buffers are ready for communication."""
        return self._send_buffer is not None and self._recv_buffer is not None

    def async_pack(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        """Pack all queued packed_buffers into a single send Buffer.

        Implementation should guarantee the send Buffer is ready for MPI communication.
        The send Buffer is held by this class."""
        raise NotImplementedError()

    def async_unpack(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        """Unpack the buffer into destination arrays.

        Implementation DOESN'T HAVE TO guarantee transfer is done."""
        raise NotImplementedError()

    def finalize(self):
        """Finalize operations after unpack.

        Implementation should guarantee all transfers are done.
        Implementation should return recv & send Buffers to the cache for reuse."""
        raise NotImplementedError()

    def synchronize(self):
        """Synchronize all operations.

        Implementation guarantees all memory is now safe to access.
        """
        raise NotImplementedError()


class HaloDataTransformerCPU(HaloDataTransformer):
    def synchronize(self):
        # CPU doesn't do true async
        pass

    def async_pack(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        # Unpack per type
        if self._type == _HaloDataTransformerType.SCALAR:
            self._pack_scalar(quantities_x)
        elif self._type == _HaloDataTransformerType.VECTOR:
            assert quantities_y is not None
            self._pack_vector(quantities_x, quantities_y)
        else:
            raise RuntimeError(f"Unimplemented {self._type} packed_buffer pack")

        assert isinstance(self._send_buffer, Buffer)  # e.g. allocate happened
        self._send_buffer.finalize_memory_transfer()

    def _pack_scalar(self, quantities: List[Quantity]):
        assert isinstance(self._send_buffer, Buffer)  # e.g. allocate happened
        offset = 0
        # TODO: check here
        for quantity in quantities:
            for x_info in self._x_infos:
                packed_buffer_size = _slices_length(x_info.send_slices)
                # sending data across the boundary will rotate the data
                # n_clockwise_rotations times, due to the difference in axis orientation.\
                # Thus we rotate that number of times counterclockwise before sending,
                # to get the right final orientation
                source_view = rotate_scalar_data(
                    quantity.data[x_info.send_slices],
                    quantity.dims,
                    quantity.np,
                    -x_info.send_clockwise_rotation,
                )
                self._send_buffer.assign_from(
                    flatten(source_view),
                    buffer_slice=np.index_exp[offset : offset + packed_buffer_size],
                )
                offset += packed_buffer_size

    def _pack_vector(self, quantities_x: List[Quantity], quantities_y: List[Quantity]):
        # TODO: check here
        assert isinstance(self._send_buffer, Buffer)  # e.g. allocate happened
        assert len(quantities_y) == len(quantities_x)
        assert len(self._x_infos) == len(self._y_infos)
        offset = 0
        for quantity_x, quantity_y in zip(quantities_x, quantities_y):
            for x_info, y_info, in zip(self._x_infos, self._y_infos):
                # sending data across the boundary will rotate the data
                # n_clockwise_rotations times, due to the difference in axis orientation
                # Thus we rotate that number of times counterclockwise before sending,
                # to get the right final orientation
                x_view, y_view = rotate_vector_data(
                    quantity_x.data[x_info.send_slices],
                    quantity_y.data[y_info.send_slices],
                    -x_info.send_clockwise_rotation,
                    quantity_x.dims,
                    quantity_x.np,
                )

                # Pack X/Y packed_buffers in the buffer
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

    def async_unpack(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        # Unpack per type
        if self._type == _HaloDataTransformerType.SCALAR:
            self._unpack_scalar(quantities_x)
        elif self._type == _HaloDataTransformerType.VECTOR:
            assert quantities_y is not None
            self._unpack_vector(quantities_x, quantities_y)
        else:
            raise RuntimeError(f"Unimplemented {self._type} packed_buffer unpack")

        assert isinstance(self._recv_buffer, Buffer)  # e.g. allocate happened
        self._recv_buffer.finalize_memory_transfer()

    def _unpack_scalar(self, quantities: List[Quantity]):
        assert isinstance(self._recv_buffer, Buffer)  # e.g. allocate happened
        # TODO: check here
        offset = 0
        for quantity in quantities:
            for x_info in self._x_infos:
                quantity_view = quantity.data[x_info.recv_slices]
                packed_buffer_size = _slices_length(x_info.recv_slices)
                self._recv_buffer.assign_to(
                    quantity_view,
                    buffer_slice=np.index_exp[offset : offset + packed_buffer_size],
                    buffer_reshape=quantity_view.shape,
                )
                offset += packed_buffer_size

    def _unpack_vector(
        self, quantities_x: List[Quantity], quantities_y: List[Quantity]
    ):
        assert isinstance(self._recv_buffer, Buffer)  # e.g. allocate happened
        # TODO: check here
        offset = 0
        for quantity_x, quantity_y in zip(quantities_x, quantities_y):
            for x_info, y_info in zip(self._x_infos, self._y_infos):
                quantity_view = quantity_x.data[x_info.recv_slices]
                packed_buffer_size = _slices_length(x_info.recv_slices)
                self._recv_buffer.assign_to(
                    quantity_view,
                    buffer_slice=np.index_exp[offset : offset + packed_buffer_size],
                    buffer_reshape=quantity_view.shape,
                )
                offset += packed_buffer_size
                quantity_view = quantity_y.data[y_info.recv_slices]
                packed_buffer_size = _slices_length(y_info.recv_slices)
                self._recv_buffer.assign_to(
                    quantity_view,
                    buffer_slice=np.index_exp[offset : offset + packed_buffer_size],
                    buffer_reshape=quantity_view.shape,
                )
                offset += packed_buffer_size

    def finalize(self):
        # Pop the buffers back in the cache
        Buffer.push_to_cache(self._send_buffer)
        self._send_buffer = None
        Buffer.push_to_cache(self._recv_buffer)
        self._recv_buffer = None


class HaloDataTransformerGPU(HaloDataTransformer):
    @dataclass
    class CuKernelArgs:
        """All arguments requireds for the CUDA kernels."""

        stream: "cp.cuda.Stream"
        x_send_indices: "cp.ndarray"
        x_recv_indices: "cp.ndarray"
        y_send_indices: Optional["cp.ndarray"]
        y_recv_indices: Optional["cp.ndarray"]

    def __init__(self, np_module: NumpyModule) -> None:
        super().__init__(np_module)
        self._cu_kernel_args: Dict[UUID, HaloDataTransformerGPU.CuKernelArgs] = {}

    def _build_flatten_indices(
        self,
        key,
        shape,
        slices: Tuple[slice],
        dims,
        strides,
        itemsize: int,
        rotate: bool,
        rotation: int,
    ):
        # Have to go down to numpy to leverage indices calculation
        arr_indices = np.zeros(shape, dtype=np.int32, order="C")[slices]

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
        self, info: _ExchangeDataDescription, slices: Tuple[slice], rotate: bool,
    ) -> "cp.ndarray":
        """Extract a flat array of indices for the send packed_buffer.

        Also take care of rotating the indices to account for axis orientation
        """
        key = str(
            (
                slices,
                info.send_clockwise_rotation,
                info.specification.shape,
                info.specification.strides,
                info.specification.itemsize,
            )
        )

        if key not in INDICES_GPU_CACHE.keys():
            self._build_flatten_indices(
                key,
                info.specification.shape,
                slices,
                info.specification.dims,
                info.specification.strides,
                info.specification.itemsize,
                rotate,
                info.send_clockwise_rotation,
            )

        # We don't return a copy since the indices are read-only in the algorithm
        return INDICES_GPU_CACHE[key]

    def allocate(self):
        # Super to get buffer allocation
        super().allocate()
        # Allocate the streams & build the indices arrays
        if len(self._y_infos) == 0:
            for x_info in self._x_infos:
                self._cu_kernel_args[x_info._id] = HaloDataTransformerGPU.CuKernelArgs(
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
                self._cu_kernel_args[x_info._id] = HaloDataTransformerGPU.CuKernelArgs(
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

    def async_pack(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        # Unpack per type
        if self._type == _HaloDataTransformerType.SCALAR:
            self._opt_pack_scalar(quantities_x)
        elif self._type == _HaloDataTransformerType.VECTOR:
            assert quantities_y is not None
            self._opt_pack_vector(quantities_x, quantities_y)
        else:
            raise RuntimeError(f"Unimplemented {self._type} packed_buffer pack")

    def _opt_pack_scalar(
        self, quantities: List[Quantity],
    ):
        assert isinstance(self._send_buffer, Buffer)  # e.g. allocate happened
        offset = 0
        for x_info, quantity in zip(self._x_infos, quantities):
            cu_kernel_args = self._cu_kernel_args[x_info._id]

            # Use private stream
            self._use_stream(cu_kernel_args.stream)

            if quantity.metadata.dtype != np.float64:
                raise RuntimeError(f"Kernel requires f64 given {np.float64}")

            # Launch kernel
            blocks = 128
            grid_x = (x_info.send_buffer_size // blocks) + 1
            pack_scalar_f64_kernel(
                (blocks,),
                (grid_x,),
                (
                    quantity.data[:],  # source_array
                    cu_kernel_args.x_send_indices,  # indices
                    x_info.send_buffer_size,  # nIndex
                    offset,
                    self._send_buffer.array,
                ),
            )

            # Next packed_buffer offset into send buffer
            offset += x_info.send_buffer_size

    def _opt_pack_vector(
        self, quantities_x: List[Quantity], quantities_y: List[Quantity]
    ):
        assert isinstance(self._send_buffer, Buffer)  # e.g. allocate happened
        assert len(self._x_infos) == len(self._y_infos)
        offset = 0
        for x_info, y_info, quantity_x, quantity_y in zip(
            self._x_infos, self._y_infos, quantities_x, quantities_y
        ):
            cu_kernel_args = self._cu_kernel_args[x_info._id]

            # Use private stream
            self._use_stream(cu_kernel_args.stream)

            # Buffer sizes
            packed_buffer_size = x_info.send_buffer_size + y_info.send_buffer_size

            if quantity_x.metadata.dtype != np.float64:
                raise RuntimeError(f"Kernel requires f64 given {np.float64}")

            # Launch kernel
            blocks = 128
            grid_x = (packed_buffer_size // blocks) + 1
            pack_vector_f64_kernel(
                (blocks,),
                (grid_x,),
                (
                    quantity_x.data[:],  # source_array_x
                    quantity_y.data[:],  # source_array_y
                    cu_kernel_args.x_send_indices,  # indices_x
                    cu_kernel_args.y_send_indices,  # indices_y
                    x_info.send_buffer_size,  # nIndex_x
                    y_info.send_buffer_size,  # nIndex_y
                    offset,
                    (-x_info.send_clockwise_rotation) % 4,  # rotation
                    self._send_buffer.array,
                ),
            )

            # Next packed_buffer offset into send buffer
            offset += packed_buffer_size

    def async_unpack(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        # Unpack per type
        if self._type == _HaloDataTransformerType.SCALAR:
            self._opt_unpack_scalar(quantities_x)
        elif self._type == _HaloDataTransformerType.VECTOR:
            assert quantities_y is not None
            self._opt_unpack_vector(quantities_x, quantities_y)
        else:
            raise RuntimeError(f"Unimplemented {self._type} packed_buffer unpack")

    def _opt_unpack_scalar(self, quantities: List[Quantity]):
        assert isinstance(self._recv_buffer, Buffer)  # e.g. allocate happened
        offset = 0
        for x_info, quantity in zip(self._x_infos, quantities):
            # Use private stream
            kernel_args = self._cu_kernel_args[x_info._id]
            self._use_stream(kernel_args.stream)

            # Launch kernel
            blocks = 128
            grid_x = (x_info.recv_buffer_size // blocks) + 1
            unpack_scalar_f64_kernel(
                (blocks,),
                (grid_x,),
                (
                    self._recv_buffer.array,  # source_buffer
                    kernel_args.x_recv_indices,  # indices
                    x_info.recv_buffer_size,  # nIndex
                    offset,
                    quantity.data[:],  # destination_array
                ),
            )

            # Next packed_buffer offset into recv buffer
            offset += x_info.recv_buffer_size

    def _opt_unpack_vector(
        self, quantities_x: List[Quantity], quantities_y: List[Quantity]
    ):
        assert isinstance(self._recv_buffer, Buffer)  # e.g. allocate happened
        assert len(self._x_infos) == len(self._y_infos)
        offset = 0
        for x_info, y_info, quantity_x, quantity_y in zip(
            self._x_infos, self._y_infos, quantities_x, quantities_y
        ):
            # We only have writte a f64 kernel
            if quantity_x.metadata.dtype != np.float64:
                raise RuntimeError(f"Kernel requires f64 given {np.float64}")

            # Use private stream
            cu_kernel_args = self._cu_kernel_args[x_info._id]
            self._use_stream(cu_kernel_args.stream)

            # Buffer sizes
            packed_buffer_size = x_info.recv_buffer_size + y_info.recv_buffer_size

            # Launch kernel
            blocks = 128
            grid_x = (packed_buffer_size // blocks) + 1
            unpack_vector_f64_kernel(
                (blocks,),
                (grid_x,),
                (
                    self._recv_buffer.array,
                    cu_kernel_args.x_recv_indices,  # indices_x
                    cu_kernel_args.y_recv_indices,  # indices_y
                    x_info.recv_buffer_size,  # nIndex_x
                    y_info.recv_buffer_size,  # nIndex_y
                    offset,
                    quantity_x.data[:],  # destination_array_x
                    quantity_y.data[:],  # destination_array_y
                ),
            )

            # Next packed_buffer offset into send buffer
            offset += packed_buffer_size

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
