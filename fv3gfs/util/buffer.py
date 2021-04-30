from typing import Callable, Generator, Iterable, Optional, Dict, Tuple, List
from ._timing import Timer, NullTimer
import numpy as np
import contextlib
from .utils import is_c_contiguous, assign_array_via_cpu, device_synchronize
from .types import Allocator

BufferKey = Tuple[Callable, Iterable[int], type]
BUFFER_CACHE: Dict[BufferKey, List["Buffer"]] = {}


class Buffer:
    """A buffer cached by default.

    _key: key into cache storage to allow easy re-caching
    array: ndarray allocated
    """

    _key: BufferKey
    array: np.ndarray

    def __init__(self, key: BufferKey, array: np.ndarray):
        """Init a cacheable buffer.

        Args:
            key: a cache key made out of tuple of allocator (behaving like np.empty), shape and dtype
            array: ndarray of actual data
        """
        self._key = key
        self.array = array

    @classmethod
    def get_from_cache(
        cls, allocator: Allocator, shape: Iterable[int], dtype: type
    ) -> "Buffer":
        """Retrieve or insert then retrieve of buffer from cache.

        Args:
            allocator: behaves like a np.empty function, used to allocate memory
            shape: shape of array
            dtype: type of array elements
        Return:
            a buffer wrapping an allocated array
        """
        key = (allocator, shape, dtype)
        if key in BUFFER_CACHE and len(BUFFER_CACHE[key]) > 0:
            return BUFFER_CACHE[key].pop()
        else:
            if key not in BUFFER_CACHE:
                BUFFER_CACHE[key] = []
            array = allocator(shape, dtype=dtype)  # type: np.ndarray
            assert is_c_contiguous(array)
            return cls(key, array)

    @staticmethod
    def push_to_cache(buffer: "Buffer"):
        """Push the buffer back into the cache.

        Args:
            buffer: buffer to push back in cache, using internal key
        """
        BUFFER_CACHE[buffer._key].append(buffer)

    def finalize_memory_transfer(self):
        """Finalize any memory transfer"""
        device_synchronize(self.array)

    def assign_to(self, destination_array):
        """Assign internal array to destination_array.

        Args:
            destination_array: target ndarray
        """
        try:
            destination_array[:] = self.array
        except ValueError:
            assign_array_via_cpu(destination_array, self.array)

    def assign_row_to(self, destination_array, row):
        """Assign given row of the internal array to the destination_array.

        Args:
            destination_array: target ndarray
        """
        try:
            destination_array[:] = self.array[row, :]
        except ValueError:
            assign_array_via_cpu(destination_array, self.array[row, :])

    def assign_from(self, source_array):
        """Assign source_array to internal array.

        Args:
            source_array: source ndarray
        """
        try:
            self.array[:] = source_array
        except TypeError:
            assign_array_via_cpu(self.array, source_array)

    def assign_row_from(self, source_array, row):
        """Assign source_array to the given row of the internal array.

        Args:
            source_array: source ndarray
            row: index of the row to assign to
        """
        try:
            self.array[row, :] = source_array
        except TypeError:
            assign_array_via_cpu(
                self.array[row, :], source_array,
            )

    def assign_from_as_contiguous(self, source_array, numpy_module):
        """Assign a contiguous copy of the source_array to the internal array.

        Args:
            source_array: source ndarray
            numpy_module: module to call ascontiguousarray from
        """
        try:
            self.array[:] = numpy_module.ascontiguousarray(source_array)
        except TypeError:
            self.array[:] = numpy_module.ascontiguousarray(source_array.get())


@contextlib.contextmanager
def array_buffer(
    allocator: Allocator, shape: Iterable[int], dtype: type
) -> Generator[Buffer, Buffer, None]:
    """
    A context manager providing a contiguous array, which may be re-used between calls.

    Args:
        allocator: a function with the same signature as numpy.zeros which returns
            an ndarray
        shape: the shape of the desired array
        dtype: the dtype of the desired array

    Yields:
        buffer_array: an ndarray created according to the specification in the args.
            May be retained and re-used in subsequent calls.
    """
    buffer = Buffer.get_from_cache(allocator, shape, dtype)
    yield buffer
    Buffer.push_to_cache(buffer)


@contextlib.contextmanager
def send_buffer(
    allocator: Callable, array: np.ndarray, timer: Optional[Timer] = None,
) -> np.ndarray:
    """A context manager ensuring that `array` is contiguous in a context where it is
    being sent as data, copying into a recycled buffer array if necessary.

    Args:
        allocator: a function behaving like numpy.empty
        array: a possibly non-contiguous array for which to provide a buffer
        timer: object to accumulate timings for "pack"

    Yields:
        buffer_array: if array is non-contiguous, a contiguous buffer array containing
            the data from array. Otherwise, yields array.
    """
    if timer is None:
        timer = NullTimer()
    if array is None or is_c_contiguous(array):
        yield array
    else:
        timer.start("pack")
        with array_buffer(allocator, array.shape, array.dtype) as sendbuf:
            sendbuf.assign_from(array)
            # this is a little dangerous, because if there is an exception in the two
            # lines above the timer may be started but never stopped. However, it
            # cannot be avoided because we cannot put those two lines in a with or
            # try block without also including the yield line.
            timer.stop("pack")
            yield sendbuf.array


@contextlib.contextmanager
def recv_buffer(
    allocator: Callable, array: np.ndarray, timer: Optional[Timer] = None,
) -> np.ndarray:
    """A context manager ensuring that array is contiguous in a context where it is
    being used to receive data, using a recycled buffer array and then copying the
    result into array if necessary.

    Args:
        allocator: a function behaving like numpy.empty
        array: a possibly non-contiguous array for which to provide a buffer
        timer: object to accumulate timings for "unpack"

    Yields:
        buffer_array: if array is non-contiguous, a contiguous buffer array which is
            copied into array when the context is exited. Otherwise, yields array.
    """
    if timer is None:
        timer = NullTimer()
    if array is None or is_c_contiguous(array):
        yield array
    else:
        timer.start("unpack")
        with array_buffer(allocator, array.shape, array.dtype) as recvbuf:
            timer.stop("unpack")
            yield recvbuf.array
            with timer.clock("unpack"):
                recvbuf.assign_to(array)
