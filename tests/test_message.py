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
    NORTH,
    NORTHWEST,
    WEST,
    SOUTHWEST,
    SOUTH,
    SOUTHEAST,
    EAST,
    NORTHEAST,
)
from fv3gfs.util.message import MessageBundle
from fv3gfs.util.buffer import Buffer
from fv3gfs.util.rotate import rotate_scalar_data, rotate_vector_data
import copy
from typing import Tuple

try:
    import cupy as cp
except ImportError:
    cp = None


@pytest.fixture
def nz():
    return 5


@pytest.fixture
def ny():
    return 7


@pytest.fixture
def nx():
    return 7


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


@pytest.fixture(params=[1, 3])
def n_halos(request):
    return request.param


@pytest.fixture
def origin(n_halos, dims, n_buffer):
    return_list = []
    origin_dict = {
        X_DIM: n_halos + n_buffer,
        X_INTERFACE_DIM: n_halos + n_buffer,
        Y_DIM: n_halos + n_buffer,
        Y_INTERFACE_DIM: n_halos + n_buffer,
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
def shape(nz, ny, nx, dims, n_halos, n_buffer):
    return_list = []
    length_dict = {
        X_DIM: 2 * n_halos + nx + n_buffer,
        X_INTERFACE_DIM: 2 * n_halos + nx + 1 + n_buffer,
        Y_DIM: 2 * n_halos + ny + n_buffer,
        Y_INTERFACE_DIM: 2 * n_halos + ny + 1 + n_buffer,
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


def _shape_length(shape: Tuple[int]) -> int:
    """Compute linear size from slices"""
    length = 1
    for s in shape:
        length *= s
    return length


@pytest.fixture
def quantity(dims, units, origin, extent, shape, numpy, dtype, backend):
    """A list of quantities whose values are 42.42 in the computational domain and 1
    outside of it."""
    sz = _shape_length(shape)
    print(f"{shape} {sz}")
    data = numpy.arange(0, sz, dtype=dtype).reshape(shape)
    if backend == "gt4py_cupy":
        quantity = Quantity(
            cp.asnumpy(data),
            dims=dims,
            units=units,
            origin=origin,
            extent=extent,
            gt4py_backend="gtcuda",
        )
        layout_map = quantity.storage.layout_map
        print(layout_map)
    elif backend == "gt4py_numpy":
        quantity = Quantity(
            data,
            dims=dims,
            units=units,
            origin=origin,
            extent=extent,
            gt4py_backend="gtx86",
        )
        layout_map = quantity.storage.layout_map
        print(layout_map)
    else:
        quantity = Quantity(data, dims=dims, units=units, origin=origin, extent=extent)
    return quantity


@pytest.fixture(params=[-0, -1, -2, -3])
def rotation(request):
    return request.param


def test_message_allocate(quantity, n_halos):
    message = MessageBundle.get_from_quantity_module(0, quantity.np)
    boundary_north = _boundary_utils.get_boundary_slice(
        quantity.dims,
        quantity.origin,
        quantity.extent,
        quantity.data.shape,
        NORTH,
        n_halos,
        interior=False,
    )
    boundary_southwest = _boundary_utils.get_boundary_slice(
        quantity.dims,
        quantity.origin,
        quantity.extent,
        quantity.data.shape,
        SOUTHWEST,
        n_halos,
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


def _get_boundaries(quantity, n_halos):
    boundary_send_interior = True
    boundary_recv_interior = False

    send_boundaries = {}
    recv_boundaries = {}
    for direction in [
        NORTH,
        NORTHWEST,
        WEST,
        SOUTHWEST,
        SOUTH,
        SOUTHEAST,
        EAST,
        NORTHEAST,
    ]:
        send_boundaries[direction] = _boundary_utils.get_boundary_slice(
            quantity.dims,
            quantity.origin,
            quantity.extent,
            quantity.data.shape,
            direction,
            n_halos,
            interior=boundary_send_interior,
        )
        recv_boundaries[direction] = _boundary_utils.get_boundary_slice(
            quantity.dims,
            quantity.origin,
            quantity.extent,
            quantity.data.shape,
            direction,
            n_halos,
            interior=boundary_recv_interior,
        )

    return send_boundaries, recv_boundaries


def test_message_scalar_pack_unpack(quantity, rotation, n_halos):
    original_quantity: Quantity = copy.deepcopy(quantity)
    message = MessageBundle.get_from_quantity_module(0, quantity.np)

    send_boundaries, recv_boundaries = _get_boundaries(quantity, n_halos)
    N_edge_boundaries = {
        0: (send_boundaries[NORTH], recv_boundaries[SOUTH]),
        -1: (send_boundaries[NORTH], recv_boundaries[WEST]),
        -2: (send_boundaries[NORTH], recv_boundaries[NORTH]),
        -3: (send_boundaries[NORTH], recv_boundaries[EAST]),
    }

    NE_corner_boundaries = {
        0: (send_boundaries[NORTHEAST], recv_boundaries[SOUTHEAST]),
        -1: (send_boundaries[NORTHEAST], recv_boundaries[SOUTHWEST]),
        -2: (send_boundaries[NORTHEAST], recv_boundaries[NORTHWEST]),
        -3: (send_boundaries[NORTHEAST], recv_boundaries[NORTHEAST]),
    }

    message.queue_scalar_message(
        quantity,
        N_edge_boundaries[rotation][0],
        rotation,
        N_edge_boundaries[rotation][1],
    )
    message.queue_scalar_message(
        quantity,
        NE_corner_boundaries[rotation][0],
        rotation,
        NE_corner_boundaries[rotation][1],
    )

    message.allocate()
    message.async_pack()
    message.synchronize()
    # Simulate data transfer
    message.get_recv_buffer().assign_from(message.get_send_buffer().array)
    message.async_unpack()
    message.finalize()

    # From the copy of the original quantity we rotate data
    # according to the rotation & slice and insert them bak
    # this reproduce the multi-buffer strategy
    rotated = rotate_scalar_data(
        original_quantity.data[N_edge_boundaries[rotation][0]],
        original_quantity.dims,
        original_quantity.metadata.np,
        -rotation,
    )
    original_quantity.data[N_edge_boundaries[rotation][1]] = rotated
    rotated = rotate_scalar_data(
        original_quantity.data[NE_corner_boundaries[rotation][0]],
        original_quantity.dims,
        original_quantity.metadata.np,
        -rotation,
    )
    original_quantity.data[NE_corner_boundaries[rotation][1]] = rotated

    assert (original_quantity.data == quantity.data).all()
    assert message._send_buffer is None
    assert message._recv_buffer is None


def test_message_vector_pack_unpack(quantity, rotation, n_halos):
    original_quantity_x = copy.deepcopy(quantity)
    original_quantity_y = copy.deepcopy(original_quantity_x)
    x_quantity = quantity
    y_quantity = copy.deepcopy(x_quantity)
    message = MessageBundle.get_from_quantity_module(0, quantity.np)

    send_boundaries, recv_boundaries = _get_boundaries(x_quantity, n_halos)
    N_edge_boundaries = {
        0: (send_boundaries[NORTH], recv_boundaries[SOUTH]),
        -1: (send_boundaries[NORTH], recv_boundaries[WEST]),
        -2: (send_boundaries[NORTH], recv_boundaries[NORTH]),
        -3: (send_boundaries[NORTH], recv_boundaries[EAST]),
    }

    NE_corner_boundaries = {
        0: (send_boundaries[NORTHEAST], recv_boundaries[SOUTHEAST]),
        -1: (send_boundaries[NORTHEAST], recv_boundaries[SOUTHWEST]),
        -2: (send_boundaries[NORTHEAST], recv_boundaries[NORTHWEST]),
        -3: (send_boundaries[NORTHEAST], recv_boundaries[NORTHEAST]),
    }

    message.queue_vector_message(
        x_quantity,
        N_edge_boundaries[rotation][0],
        y_quantity,
        N_edge_boundaries[rotation][0],
        rotation,
        N_edge_boundaries[rotation][1],
        N_edge_boundaries[rotation][1],
    )
    message.queue_vector_message(
        x_quantity,
        NE_corner_boundaries[rotation][0],
        y_quantity,
        NE_corner_boundaries[rotation][0],
        rotation,
        NE_corner_boundaries[rotation][1],
        NE_corner_boundaries[rotation][1],
    )
    message.allocate()
    message.async_pack()
    message.synchronize()
    # Simulate data transfer
    message.get_recv_buffer().assign_from(message.get_send_buffer().array)
    message.async_unpack()
    message.finalize()

    # From the copy of the original quantity we rotate data
    # according to the rotation & slice and insert them bak
    # this reproduce the multi-buffer strategy
    rotated_x, rotated_y = rotate_vector_data(
        original_quantity_x.data[N_edge_boundaries[rotation][0]],
        original_quantity_y.data[N_edge_boundaries[rotation][0]],
        -rotation,
        original_quantity_x.dims,
        original_quantity_x.metadata.np,
    )
    original_quantity_x.data[N_edge_boundaries[rotation][1]] = rotated_x
    original_quantity_y.data[N_edge_boundaries[rotation][1]] = rotated_y
    rotated_x, rotated_y = rotate_vector_data(
        original_quantity_x.data[NE_corner_boundaries[rotation][0]],
        original_quantity_y.data[NE_corner_boundaries[rotation][0]],
        -rotation,
        original_quantity_x.dims,
        original_quantity_x.metadata.np,
    )
    original_quantity_x.data[NE_corner_boundaries[rotation][1]] = rotated_x
    original_quantity_y.data[NE_corner_boundaries[rotation][1]] = rotated_y

    assert (original_quantity_x.data == x_quantity.data).all()
    assert (original_quantity_y.data == y_quantity.data).all()
    assert message._send_buffer is None
    assert message._recv_buffer is None
