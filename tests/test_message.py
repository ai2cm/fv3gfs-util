import pytest
from fv3gfs.util import (
    Quantity,
    X_DIM,
    Y_DIM,
    Z_DIM,
    Y_INTERFACE_DIM,
    X_INTERFACE_DIM,
    Z_INTERFACE_DIM,
    _boundary_utils,
    SOUTHWEST,
    NORTH,
)
from fv3gfs.util.message import MessageBundle
from fv3gfs.util.buffer import Buffer
import copy


@pytest.fixture
def nz():
    return 3


@pytest.fixture
def ny():
    return 5


@pytest.fixture
def nx():
    return 5


@pytest.fixture
def units():
    return "m"


@pytest.fixture(params=[0, 1])
def n_buffer(request):
    return request.param


@pytest.fixture
def n_points():
    return 1


@pytest.fixture
def dtype(numpy):
    return numpy.float64


@pytest.fixture
def origin(n_points, dims, n_buffer):
    return_list = []
    origin_dict = {
        X_DIM: n_points + n_buffer,
        X_INTERFACE_DIM: n_points + n_buffer,
        Y_DIM: n_points + n_buffer,
        Y_INTERFACE_DIM: n_points + n_buffer,
        Z_DIM: n_buffer,
        Z_INTERFACE_DIM: n_buffer,
    }
    for dim in dims:
        return_list.append(origin_dict[dim])
    return return_list


@pytest.fixture(
    params=[
        pytest.param((Y_DIM, X_DIM), id="center"),
        pytest.param((Z_DIM, Y_DIM, X_DIM), id="center_3d"),
        pytest.param((X_DIM, Y_DIM, Z_DIM), id="center_3d_reverse",),
        pytest.param((X_DIM, Z_DIM, Y_DIM), id="center_3d_shuffle",),
        pytest.param((Y_INTERFACE_DIM, X_INTERFACE_DIM), id="interface"),
        pytest.param(
            (Z_INTERFACE_DIM, Y_INTERFACE_DIM, X_INTERFACE_DIM,), id="interface_3d",
        ),
    ]
)
def dims(request, fast):
    if fast and request.param in (
        (X_DIM, Y_DIM, Z_DIM),
        (Z_INTERFACE_DIM, Y_INTERFACE_DIM, X_INTERFACE_DIM,),
    ):
        pytest.skip("running in fast mode")
    return request.param


@pytest.fixture
def shape(nz, ny, nx, dims, n_points, n_buffer):
    return_list = []
    length_dict = {
        X_DIM: 2 * n_points + nx + n_buffer,
        X_INTERFACE_DIM: 2 * n_points + nx + 1 + n_buffer,
        Y_DIM: 2 * n_points + ny + n_buffer,
        Y_INTERFACE_DIM: 2 * n_points + ny + 1 + n_buffer,
        Z_DIM: nz + n_buffer,
        Z_INTERFACE_DIM: nz + 1 + n_buffer,
    }
    for dim in dims:
        return_list.append(length_dict[dim])
    return return_list


@pytest.fixture
def extent(n_points, dims, nz, ny, nx):
    return_list = []
    extent_dict = {
        X_DIM: nx,
        X_INTERFACE_DIM: nx + 1,
        Y_DIM: ny,
        Y_INTERFACE_DIM: ny + 1,
        Z_DIM: nz,
        Z_INTERFACE_DIM: nz + 1,
    }
    for dim in dims:
        return_list.append(extent_dict[dim])
    return return_list


@pytest.fixture
def quantity(dims, units, origin, extent, shape, numpy, dtype):
    """A list of quantities whose values are 42.42 in the computational domain and 1
    outside of it."""
    data = numpy.ones(shape, dtype=dtype)
    quantity = Quantity(data, dims=dims, units=units, origin=origin, extent=extent)
    quantity.view[:] = 42.42
    return quantity


def test_message_allocate(quantity):
    message = MessageBundle.get_from_quantity_module(0, quantity.np)
    boundary_north = _boundary_utils.get_boundary_slice(
        quantity.dims,
        quantity.origin,
        quantity.extent,
        quantity.data.shape,
        NORTH,
        1,
        interior=False,
    )
    boundary_southwest = _boundary_utils.get_boundary_slice(
        quantity.dims,
        quantity.origin,
        quantity.extent,
        quantity.data.shape,
        SOUTHWEST,
        1,
        interior=False,
    )

    message.queue_scalar_message(quantity, boundary_north, 0, boundary_north)
    message.queue_scalar_message(quantity, boundary_southwest, 0, boundary_southwest)
    message.allocate()
    assert len(message.get_send_buffer().array.shape) == 1
    assert (
        message.get_send_buffer().array.size
        == quantity.data[boundary_north].size + quantity.data[boundary_southwest].size
    )
    assert len(message.get_recv_buffer().array.shape) == 1
    assert (
        message.get_recv_buffer().array.size
        == quantity.data[boundary_north].size + quantity.data[boundary_southwest].size
    )
    # clean up
    Buffer.push_to_cache(message._send_buffer)
    Buffer.push_to_cache(message._recv_buffer)


def test_message_scalar_pack_unpack(quantity):
    original_quantity = copy.deepcopy(quantity)
    message = MessageBundle.get_from_quantity_module(0, quantity.np)

    boundary_north = _boundary_utils.get_boundary_slice(
        quantity.dims,
        quantity.origin,
        quantity.extent,
        quantity.data.shape,
        NORTH,
        1,
        interior=False,
    )
    boundary_southwest = _boundary_utils.get_boundary_slice(
        quantity.dims,
        quantity.origin,
        quantity.extent,
        quantity.data.shape,
        SOUTHWEST,
        1,
        interior=False,
    )

    message.queue_scalar_message(quantity, boundary_north, 0, boundary_north)
    message.queue_scalar_message(quantity, boundary_southwest, 0, boundary_southwest)
    message.allocate()
    message.async_pack()
    message.synchronize()
    # Simulate data transfer
    message.get_recv_buffer().assign_from(message.get_send_buffer().array)
    message.async_unpack()
    message.synchronize()
    assert (original_quantity.data == quantity.data).all()
    assert message._send_buffer is None
    assert message._recv_buffer is None


def test_message_vector_pack_unpack(quantity):
    original_quantity = copy.deepcopy(quantity)
    x_quantity = quantity
    y_quantity = copy.deepcopy(x_quantity)
    message = MessageBundle.get_from_quantity_module(0, quantity.np)

    boundary_north = _boundary_utils.get_boundary_slice(
        quantity.dims,
        quantity.origin,
        quantity.extent,
        quantity.data.shape,
        NORTH,
        1,
        interior=False,
    )

    message.queue_vector_message(
        x_quantity,
        boundary_north,
        y_quantity,
        boundary_north,
        1,
        boundary_north,
        boundary_north,
    )
    message.allocate()
    message.async_pack()
    message.synchronize()
    # Simulate data transfer
    message.get_recv_buffer().assign_from(message.get_send_buffer().array)
    message.async_unpack()
    message.synchronize()
    original_quantity.data[boundary_north] = -1
    assert (original_quantity.data == quantity.data).all()
    assert message._send_buffer is None
    assert message._recv_buffer is None
