import pytest
from fv3gfs.util.buffer import send_buffer, recv_buffer, Buffer, BUFFER_CACHE
from fv3gfs.util.utils import is_contiguous, is_c_contiguous


@pytest.fixture
def contiguous_array(numpy, backend):
    if backend == "gt4py_cupy":
        pytest.skip("gt4py gpu backend cannot produce contiguous arrays")
    array = numpy.empty([3, 4, 5])
    array[:] = numpy.random.randn(3, 4, 5)
    return array


@pytest.fixture
def non_contiguous_array(contiguous_array):
    return contiguous_array.transpose(2, 0, 1)


@pytest.fixture(params=["empty", "zeros", "ones"])
def allocator(request, numpy):
    return getattr(numpy, request.param)


def test_is_contiguous(contiguous_array):
    assert is_contiguous(contiguous_array)


def test_is_c_contiguous(contiguous_array):
    assert is_c_contiguous(contiguous_array)


def test_not_is_contiguous(non_contiguous_array):
    assert not is_contiguous(non_contiguous_array)


def test_not_is_c_contiguous(non_contiguous_array):
    assert not is_c_contiguous(non_contiguous_array)


def test_sendbuf_uses_buffer(numpy, backend, allocator, non_contiguous_array):
    with send_buffer(allocator, non_contiguous_array) as sendbuf:
        assert sendbuf is not non_contiguous_array
        assert sendbuf.data is not non_contiguous_array.data
        numpy.testing.assert_array_equal(sendbuf, non_contiguous_array)


def test_recvbuf_uses_buffer(numpy, allocator, non_contiguous_array):
    with recv_buffer(allocator, non_contiguous_array) as recvbuf:
        assert recvbuf is not non_contiguous_array
        assert recvbuf.data is not non_contiguous_array.data
        recvbuf[:] = 0.0
        assert not numpy.all(non_contiguous_array == 0.0)
    assert numpy.all(non_contiguous_array == 0.0)


def test_sendbuf_no_buffer(allocator, contiguous_array):
    with send_buffer(allocator, contiguous_array) as sendbuf:
        assert sendbuf is contiguous_array


def test_recvbuf_no_buffer(allocator, contiguous_array):
    with recv_buffer(allocator, contiguous_array) as recvbuf:
        assert recvbuf is contiguous_array


def test_buffer_cache_appends(allocator, backend):
    """Test buffer with the same key are appended while not in use for potential reuse"""
    if backend == "gt4py_cupy":
        pytest.skip("gt4py gpu backend cannot produce contiguous arrays")
    BUFFER_CACHE.clear()
    assert len(BUFFER_CACHE) == 0
    shape = (10, 10, 10)
    b0_0 = Buffer.pop_from_cache(allocator, shape, float)
    b0_0.array.fill(42)
    assert len(BUFFER_CACHE) == 1
    b0_1 = Buffer.pop_from_cache(allocator, shape, float)
    b0_1.array.fill(23)
    assert b0_0._key == b0_1._key
    assert (b0_0.array != b0_1.array).all()
    assert len(BUFFER_CACHE) == 1
    assert len(next(iter(BUFFER_CACHE.items()))[1]) == 0
    Buffer.push_to_cache(b0_0)
    Buffer.push_to_cache(b0_1)
    assert len(next(iter(BUFFER_CACHE.items()))[1]) == 2


def test_buffer_reuse(allocator, backend):
    """Test we reuse the buffer when available instead of reallocating one"""
    if backend == "gt4py_cupy":
        pytest.skip("gt4py gpu backend cannot produce contiguous arrays")
    BUFFER_CACHE.clear()
    assert len(BUFFER_CACHE) == 0
    shape = (10, 10, 10)
    b0 = Buffer.pop_from_cache(allocator, shape, float)
    fill_scalar = 42
    b0.array.fill(fill_scalar)
    assert len(BUFFER_CACHE) == 1
    Buffer.push_to_cache(b0)
    assert len(BUFFER_CACHE) == 1
    b1 = Buffer.pop_from_cache(allocator, shape, float)
    assert len(next(iter(BUFFER_CACHE.items()))[1]) == 0
    assert (b1.array == fill_scalar).all()
