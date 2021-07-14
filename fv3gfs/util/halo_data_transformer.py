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
    """Describe the memory to be exchanged, including size of the halo."""

    n_halo_points: int
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
# Indices array

# Keyed cached - key is a str at the moment to go around the fact that
# a slice is not hashable. getting a string from
# Tuple(slices, rotation, shape, strides, itemsize) e.g. # noqa
# str(Tuple[Any, int, Tuple[int], Tuple[int], int]) # noqa
INDICES_GPU_CACHE: Dict[str, "cp.ndarray"] = {}


def _build_flatten_indices(
    key,
    shape,
    slices: Tuple[slice],
    dims,
    strides,
    itemsize: int,
    rotate: bool,
    rotation: int,
):
    """Build an array of indexing from a slice description.
    
    Go from a memory layout (strides, itemsize, shape) and slices into it to a
    single array of indices. We leverage numpy iterator and calculate from the multi_index
    using memory layout the index into the original memory buffer.
    """

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


# ------------------------------------------------------------------------
# HaloDataTransformer helpers


def _slices_length(slices: Tuple[slice]) -> int:
    """Compute linear size from slices."""
    length = 1
    for s in slices:
        assert s.step is None
        length *= abs(s.start - s.stop)
    return length


@dataclass
class _HaloExchangeData:
    """Memory description of the data exchanged.
    
    Args:
        specification: memory layout of the data
        pack_slices: indexing to pack
        pack_clockwise_rotation: required data rotation when packing
        unpack_slices: indexing to unpack
    """

    specification: HaloUpdateSpec
    pack_slices: Tuple[slice]
    pack_clockwise_rotation: int
    unpack_slices: Tuple[slice]

    def __post_init__(self):
        self._id = uuid1()
        self._pack_buffer_size = _slices_length(self.pack_slices)
        self._unpack_buffer_size = _slices_length(self.unpack_slices)


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
    - add N transformation with the proper halo specification via add_scalar_specification or
     add_vector_specification (can't mix and match scalar & vector)
    - compile the transformer via compile()
    [From here get_unpack_buffer() is valid]
    - async_pack
    - synchronize
    [From here get_pack_buffer() is valid]
    [... user should communicate the buffers...]
    - async_unpack
    - finalize
    [All buffers are returned to their buffer cache, therefore invalid]

    If the user doesn't want the buffer return to the cache (optimal behavior if the same exchange
    is to be done multiple times), it can use `synchronize` instead of `finalize`
    """

    _pack_buffer: Optional[Buffer]
    _unpack_buffer: Optional[Buffer]

    _infos_x: List[_HaloExchangeData]
    _infos_y: List[_HaloExchangeData]

    def __init__(self, np_module: NumpyModule) -> None:
        """Init routine.

        Arguments:
            np_module: numpy-like module for allocation
        """
        self._type = _HaloDataTransformerType.UNKNOWN
        self._np_module = np_module
        self._infos_x = []
        self._infos_y = []
        self._pack_buffer = None
        self._unpack_buffer = None

    def __del__(self):
        """Del routine, making sure all buffers were inserted back into cache."""
        assert self._pack_buffer is None
        assert self._unpack_buffer is None

    @staticmethod
    def get_from_numpy_module(np_module: NumpyModule) -> "HaloDataTransformer":
        """Construct a module from a numpy-like module.

        Arguments:
            np_module: numpy-like module to determin child transformer type.

        Return: an initialized packed buffer.
        """
        if np_module is np:
            return HaloDataTransformerCPU(np)
        elif np_module is cp:
            return HaloDataTransformerGPU(cp)

        raise NotImplementedError(
            f"Quantity module {np_module} has no HaloDataTransformer implemented"
        )

    def add_scalar_specification(
        self,
        specification: HaloUpdateSpec,
        send_slice: Tuple[slice],
        n_clockwise_rotation: int,
        recv_slice: Tuple[slice],
    ):
        """Add memory specification (scalar)."""
        assert (
            self._type == _HaloDataTransformerType.SCALAR
            or self._type == _HaloDataTransformerType.UNKNOWN
        )
        if self._unpack_buffer is not None and self._pack_buffer is not None:
            raise RuntimeError(
                "This transformer has already been compiled"
                "You can only compile one specification per Transformer."
            )

        self._type = _HaloDataTransformerType.SCALAR
        self._infos_x.append(
            _HaloExchangeData(
                specification=specification,
                pack_slices=send_slice,
                pack_clockwise_rotation=n_clockwise_rotation,
                unpack_slices=recv_slice,
            )
        )

    def add_vector_specification(
        self,
        specification_x: HaloUpdateSpec,
        specification_y: HaloUpdateSpec,
        x_pack_slice: Tuple[slice],
        y_pack_slice: Tuple[slice],
        n_clockwise_rotation: int,
        x_recv_slice: Tuple[slice],
        y_recv_slice: Tuple[slice],
    ):
        """Add memory specification (vector)."""
        assert (
            self._type == _HaloDataTransformerType.VECTOR
            or self._type == _HaloDataTransformerType.UNKNOWN
        )
        if self._unpack_buffer is not None and self._pack_buffer is not None:
            raise RuntimeError(
                "This transformer has already been compiled"
                "You can only compile one specification per Transformer."
            )
        self._type = _HaloDataTransformerType.VECTOR
        self._infos_x.append(
            _HaloExchangeData(
                specification=specification_x,
                pack_slices=x_pack_slice,
                pack_clockwise_rotation=n_clockwise_rotation,
                unpack_slices=x_recv_slice,
            )
        )
        self._infos_y.append(
            _HaloExchangeData(
                specification=specification_y,
                pack_slices=y_pack_slice,
                pack_clockwise_rotation=n_clockwise_rotation,
                unpack_slices=y_recv_slice,
            )
        )

    def get_unpack_buffer(self) -> Buffer:
        """Retrieve unpack buffer.

        WARNING: Only available _after_ `compile` as been called.
        """
        if self._unpack_buffer is None:
            raise RuntimeError("Recv buffer can't be retrieved before allocate()")
        return self._unpack_buffer

    def get_pack_buffer(self) -> Buffer:
        """Retrieve pack buffer.

        WARNING: Only available _after_ `compile` as been called.
        """
        if self._pack_buffer is None:
            raise RuntimeError("Send buffer can't be retrieved before allocate()")
        return self._pack_buffer

    def compile(self):
        """Allocate contiguous memory buffers from description queued."""

        # Compute required size
        buffer_size = 0
        dtype = None
        for edge_x in self._infos_x:
            buffer_size += edge_x._pack_buffer_size
            dtype = edge_x.specification.dtype
        if self._type is _HaloDataTransformerType.VECTOR:
            for edge_y in self._infos_y:
                buffer_size += edge_y._pack_buffer_size

        # Retrieve two properly sized buffers
        self._pack_buffer = Buffer.pop_from_cache(
            self._np_module.empty, (buffer_size), dtype
        )
        self._unpack_buffer = Buffer.pop_from_cache(
            self._np_module.empty, (buffer_size), dtype
        )

    def ready(self) -> bool:
        """Check if the buffers are ready for communication."""
        return self._pack_buffer is not None and self._unpack_buffer is not None

    def async_pack(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        """Pack all given quantities into a single send Buffer.

        Implementation should guarantee the send Buffer is ready for MPI communication.
        The send Buffer is held by this class."""
        raise NotImplementedError()

    def async_unpack(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        """Unpack the buffer into destination quantities.

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
            raise RuntimeError(f"Unimplemented {self._type} pack")

        assert isinstance(self._pack_buffer, Buffer)  # e.g. allocate happened
        self._pack_buffer.finalize_memory_transfer()

    def _pack_scalar(self, quantities: List[Quantity]):
        if __debug__:
            if len(quantities) != len(self._infos_x):
                raise RuntimeError(
                    f"Quantities count ({len(quantities)}"
                    f" is different that edges count {len(self._infos_x)}"
                )
            # TODO Per quantity check

        assert isinstance(self._pack_buffer, Buffer)  # e.g. allocate happened
        offset = 0
        for quantity, info_x in zip(quantities, self._infos_x):
            data_size = _slices_length(info_x.pack_slices)
            # sending data across the boundary will rotate the data
            # n_clockwise_rotations times, due to the difference in axis orientation.\
            # Thus we rotate that number of times counterclockwise before sending,
            # to get the right final orientation
            source_view = rotate_scalar_data(
                quantity.data[info_x.pack_slices],
                quantity.dims,
                quantity.np,
                -info_x.pack_clockwise_rotation,
            )
            self._pack_buffer.assign_from(
                flatten(source_view),
                buffer_slice=np.index_exp[offset : offset + data_size],
            )
            offset += data_size

    def _pack_vector(self, quantities_x: List[Quantity], quantities_y: List[Quantity]):
        if __debug__:
            if len(quantities_x) != len(self._infos_x) and len(quantities_y) != len(
                self._infos_y
            ):
                raise RuntimeError(
                    f"Quantities count (x: {len(quantities_x)}, y: {len(quantities_y)})"
                    f" is different that specifications count (x: {len(self._infos_x)}, y: {len(self._infos_y)}"
                )
            # TODO Per quantity check

        assert isinstance(self._pack_buffer, Buffer)  # e.g. allocate happened
        assert len(quantities_y) == len(quantities_x)
        assert len(self._infos_x) == len(self._infos_y)
        offset = 0
        for quantity_x, quantity_y, info_x, info_y, in zip(
            quantities_x, quantities_y, self._infos_x, self._infos_y
        ):
            # sending data across the boundary will rotate the data
            # n_clockwise_rotations times, due to the difference in axis orientation
            # Thus we rotate that number of times counterclockwise before sending,
            # to get the right final orientation
            x_view, y_view = rotate_vector_data(
                quantity_x.data[info_x.pack_slices],
                quantity_y.data[info_y.pack_slices],
                -info_x.pack_clockwise_rotation,
                quantity_x.dims,
                quantity_x.np,
            )

            # Pack X/Y data slices in the buffer
            self._pack_buffer.assign_from(
                flatten(x_view),
                buffer_slice=np.index_exp[offset : offset + x_view.size],
            )
            offset += x_view.size
            self._pack_buffer.assign_from(
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
            raise RuntimeError(f"Unimplemented {self._type} unpack")

        assert isinstance(self._unpack_buffer, Buffer)  # e.g. allocate happened
        self._unpack_buffer.finalize_memory_transfer()

    def _unpack_scalar(self, quantities: List[Quantity]):
        if __debug__:
            if len(quantities) != len(self._infos_x):
                raise RuntimeError(
                    f"Quantities count ({len(quantities)}"
                    f" is different that specifications count {len(self._infos_x)}"
                )
            # TODO Per quantity check

        assert isinstance(self._unpack_buffer, Buffer)  # e.g. allocate happened
        offset = 0
        for quantity, info_x in zip(quantities, self._infos_x):
            quantity_view = quantity.data[info_x.unpack_slices]
            data_size = _slices_length(info_x.unpack_slices)
            self._unpack_buffer.assign_to(
                quantity_view,
                buffer_slice=np.index_exp[offset : offset + data_size],
                buffer_reshape=quantity_view.shape,
            )
            offset += data_size

    def _unpack_vector(
        self, quantities_x: List[Quantity], quantities_y: List[Quantity]
    ):
        if __debug__:
            if len(quantities_x) != len(self._infos_x) and len(quantities_y) != len(
                self._infos_y
            ):
                raise RuntimeError(
                    f"Quantities count (x: {len(quantities_x)}, y: {len(quantities_y)})"
                    f" is different that specifications count (x: {len(self._infos_x)}, y: {len(self._infos_y)})"
                )
            # TODO Per quantity check

        assert isinstance(self._unpack_buffer, Buffer)  # e.g. allocate happened
        offset = 0
        for quantity_x, quantity_y, info_x, info_y in zip(
            quantities_x, quantities_y, self._infos_x, self._infos_y
        ):
            quantity_view = quantity_x.data[info_x.unpack_slices]
            data_size = _slices_length(info_x.unpack_slices)
            self._unpack_buffer.assign_to(
                quantity_view,
                buffer_slice=np.index_exp[offset : offset + data_size],
                buffer_reshape=quantity_view.shape,
            )
            offset += data_size
            quantity_view = quantity_y.data[info_y.unpack_slices]
            data_size = _slices_length(info_y.unpack_slices)
            self._unpack_buffer.assign_to(
                quantity_view,
                buffer_slice=np.index_exp[offset : offset + data_size],
                buffer_reshape=quantity_view.shape,
            )
            offset += data_size

    def finalize(self):
        # Pop the buffers back in the cache
        Buffer.push_to_cache(self._pack_buffer)
        self._pack_buffer = None
        Buffer.push_to_cache(self._unpack_buffer)
        self._unpack_buffer = None


class HaloDataTransformerGPU(HaloDataTransformer):
    @dataclass
    class _CuKernelArgs:
        """All arguments required for the CUDA kernels."""

        stream: "cp.cuda.Stream"
        x_send_indices: "cp.ndarray"
        x_recv_indices: "cp.ndarray"
        y_send_indices: Optional["cp.ndarray"]
        y_recv_indices: Optional["cp.ndarray"]

    def __init__(self, np_module: NumpyModule) -> None:
        super().__init__(np_module)
        self._cu_kernel_args: Dict[UUID, HaloDataTransformerGPU._CuKernelArgs] = {}

    def _flatten_indices(
        self, exchange_data: _HaloExchangeData, slices: Tuple[slice], rotate: bool,
    ) -> "cp.ndarray":
        """Extract a flat array of indices for this send operation.

        Also take care of rotating the indices to account for axis orientation
        """
        key = str(
            (
                slices,
                exchange_data.pack_clockwise_rotation,
                exchange_data.specification.shape,
                exchange_data.specification.strides,
                exchange_data.specification.itemsize,
            )
        )

        if key not in INDICES_GPU_CACHE.keys():
            _build_flatten_indices(
                key,
                exchange_data.specification.shape,
                slices,
                exchange_data.specification.dims,
                exchange_data.specification.strides,
                exchange_data.specification.itemsize,
                rotate,
                exchange_data.pack_clockwise_rotation,
            )

        # We don't return a copy since the indices are read-only in the algorithm
        return INDICES_GPU_CACHE[key]

    def compile(self):
        # Super to get buffer allocation
        super().compile()
        # Allocate the streams & build the indices arrays
        if len(self._infos_y) == 0:
            for info_x in self._infos_x:
                self._cu_kernel_args[info_x._id] = HaloDataTransformerGPU._CuKernelArgs(
                    stream=_pop_stream(),
                    x_send_indices=self._flatten_indices(
                        info_x, info_x.pack_slices, True
                    ),
                    x_recv_indices=self._flatten_indices(
                        info_x, info_x.unpack_slices, False
                    ),
                    y_send_indices=None,
                    y_recv_indices=None,
                )
        else:
            for info_x, info_y in zip(self._infos_x, self._infos_y):
                self._cu_kernel_args[info_x._id] = HaloDataTransformerGPU._CuKernelArgs(
                    stream=_pop_stream(),
                    x_send_indices=self._flatten_indices(
                        info_x, info_x.pack_slices, True
                    ),
                    x_recv_indices=self._flatten_indices(
                        info_x, info_x.unpack_slices, False
                    ),
                    y_send_indices=self._flatten_indices(
                        info_y, info_y.pack_slices, True
                    ),
                    y_recv_indices=self._flatten_indices(
                        info_y, info_y.unpack_slices, False
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
        """Pack the quantities into a single buffer via streamed cuda kernels
        
        Writes into send_buffer.

        Args:
            quantities_x: list of quantities to pack. Must fit the specifications given
            at init time.
            quantities_y: Same as above but optional, used only for Vector transfer.
        """

        # Unpack per type
        if self._type == _HaloDataTransformerType.SCALAR:
            self._opt_pack_scalar(quantities_x)
        elif self._type == _HaloDataTransformerType.VECTOR:
            assert quantities_y is not None
            self._opt_pack_vector(quantities_x, quantities_y)
        else:
            raise RuntimeError(f"Unimplemented {self._type} pack")

    def _opt_pack_scalar(
        self, quantities: List[Quantity],
    ):
        if __debug__:
            if len(quantities) != len(self._infos_x):
                raise RuntimeError(
                    f"Quantities count ({len(quantities)}"
                    f" is different that specifications count {len(self._infos_x)}"
                )
            # TODO Per quantity check

        assert isinstance(self._pack_buffer, Buffer)  # e.g. allocate happened
        offset = 0
        for info_x, quantity in zip(self._infos_x, quantities):
            cu_kernel_args = self._cu_kernel_args[info_x._id]

            # Use private stream
            self._use_stream(cu_kernel_args.stream)

            if quantity.metadata.dtype != np.float64:
                raise RuntimeError(f"Kernel requires f64 given {np.float64}")

            # Launch kernel
            blocks = 128
            grid_x = (info_x._pack_buffer_size // blocks) + 1
            pack_scalar_f64_kernel(
                (blocks,),
                (grid_x,),
                (
                    quantity.data[:],  # source_array
                    cu_kernel_args.x_send_indices,  # indices
                    info_x._pack_buffer_size,  # nIndex
                    offset,
                    self._pack_buffer.array,
                ),
            )

            # Next transformer offset into send buffer
            offset += info_x._pack_buffer_size

    def _opt_pack_vector(
        self, quantities_x: List[Quantity], quantities_y: List[Quantity]
    ):
        """Unpack the quantities from a single buffer via streamed cuda kernels
        
        Reads from recv_buffer.

        Args:
            quantities_x: list of quantities to unpack. Must fit the specifications given
            at init time.
            quantities_y: Same as above but optional, used only for Vector transfer.
        """
        if __debug__:
            if len(quantities_x) != len(self._infos_x) and len(quantities_y) != len(
                self._infos_y
            ):
                raise RuntimeError(
                    f"Quantities count (x: {len(quantities_x)}, y: {len(quantities_y)}"
                    f" is different that specifications count (x: {len(self._infos_x)}, y: {len(self._infos_y)}"
                )
            # TODO Per quantity check
        assert isinstance(self._pack_buffer, Buffer)  # e.g. allocate happened
        assert len(self._infos_x) == len(self._infos_y)
        assert len(quantities_x) == len(quantities_y)
        offset = 0
        for quantity_x, quantity_y, info_x, info_y, in zip(
            quantities_x, quantities_y, self._infos_x, self._infos_y
        ):
            cu_kernel_args = self._cu_kernel_args[info_x._id]

            # Use private stream
            self._use_stream(cu_kernel_args.stream)

            # Buffer sizes
            transformer_size = info_x._pack_buffer_size + info_y._pack_buffer_size

            if quantity_x.metadata.dtype != np.float64:
                raise RuntimeError(f"Kernel requires f64 given {np.float64}")

            # Launch kernel
            blocks = 128
            grid_x = (transformer_size // blocks) + 1
            pack_vector_f64_kernel(
                (blocks,),
                (grid_x,),
                (
                    quantity_x.data[:],  # source_array_x
                    quantity_y.data[:],  # source_array_y
                    cu_kernel_args.x_send_indices,  # indices_x
                    cu_kernel_args.y_send_indices,  # indices_y
                    info_x._pack_buffer_size,  # nIndex_x
                    info_y._pack_buffer_size,  # nIndex_y
                    offset,
                    (-info_x.pack_clockwise_rotation) % 4,  # rotation
                    self._pack_buffer.array,
                ),
            )

            # Next transformer offset into send buffer
            offset += transformer_size

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
            raise RuntimeError(f"Unimplemented {self._type} unpack")

    def _opt_unpack_scalar(self, quantities: List[Quantity]):
        if __debug__:
            if len(quantities) != len(self._infos_x):
                raise RuntimeError(
                    f"Quantities count ({len(quantities)})"
                    f" is different that specifications count ({len(self._infos_x)})"
                )
            # TODO Per quantity check
        assert isinstance(self._unpack_buffer, Buffer)  # e.g. allocate happened
        offset = 0
        for quantity, info_x in zip(quantities, self._infos_x):
            # Use private stream
            kernel_args = self._cu_kernel_args[info_x._id]
            self._use_stream(kernel_args.stream)

            # Launch kernel
            blocks = 128
            grid_x = (info_x._unpack_buffer_size // blocks) + 1
            unpack_scalar_f64_kernel(
                (blocks,),
                (grid_x,),
                (
                    self._unpack_buffer.array,  # source_buffer
                    kernel_args.x_recv_indices,  # indices
                    info_x._unpack_buffer_size,  # nIndex
                    offset,
                    quantity.data[:],  # destination_array
                ),
            )

            # Next transformer offset into recv buffer
            offset += info_x._unpack_buffer_size

    def _opt_unpack_vector(
        self, quantities_x: List[Quantity], quantities_y: List[Quantity]
    ):
        if __debug__:
            if len(quantities_x) != len(self._infos_x) and len(quantities_y) != len(
                self._infos_y
            ):
                raise RuntimeError(
                    f"Quantities count (x: {len(quantities_x)}, y: {len(quantities_y)}"
                    f" is different that specifications count (x: {len(self._infos_x)}, y: {len(self._infos_y)}"
                )
            # TODO Per quantity check
        assert isinstance(self._unpack_buffer, Buffer)  # e.g. allocate happened
        assert len(self._infos_x) == len(self._infos_y)
        assert len(quantities_x) == len(quantities_y)
        offset = 0
        for quantity_x, quantity_y, info_x, info_y, in zip(
            quantities_x, quantities_y, self._infos_x, self._infos_y
        ):
            # We only have writte a f64 kernel
            if quantity_x.metadata.dtype != np.float64:
                raise RuntimeError(f"Kernel requires f64 given {np.float64}")

            # Use private stream
            cu_kernel_args = self._cu_kernel_args[info_x._id]
            self._use_stream(cu_kernel_args.stream)

            # Buffer sizes
            edge_size = info_x._unpack_buffer_size + info_y._unpack_buffer_size

            # Launch kernel
            blocks = 128
            grid_x = (edge_size // blocks) + 1
            unpack_vector_f64_kernel(
                (blocks,),
                (grid_x,),
                (
                    self._unpack_buffer.array,
                    cu_kernel_args.x_recv_indices,  # indices_x
                    cu_kernel_args.y_recv_indices,  # indices_y
                    info_x._unpack_buffer_size,  # nIndex_x
                    info_y._unpack_buffer_size,  # nIndex_y
                    offset,
                    quantity_x.data[:],  # destination_array_x
                    quantity_y.data[:],  # destination_array_y
                ),
            )

            # Next transformer offset into send buffer
            offset += edge_size

    def finalize(self):
        # Synchronize all work
        self.synchronize()

        # Push the buffers back in the cache
        Buffer.push_to_cache(self._pack_buffer)
        self._pack_buffer = None
        Buffer.push_to_cache(self._unpack_buffer)
        self._unpack_buffer = None

        # Push the streams back in the pool
        for cu_info in self._cu_kernel_args.values():
            _push_stream(cu_info.stream)
