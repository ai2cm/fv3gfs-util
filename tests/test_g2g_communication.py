""" Test of the GPU to GPU communication strategy.

Those test use halo_update but are separated from the entire
"""
import pytest
import numpy as np
import fv3gfs.util
import contextlib
import functools

try:
    import cupy as cp
except ModuleNotFoundError:
    cp = None


@pytest.fixture(params=[(1, 1), (3, 3)])
def layout(request, fast):
    if fast and request.param == (1, 1):
        pytest.skip("running in fast mode")
    else:
        return request.param


@pytest.fixture
def ranks_per_tile(layout):
    return layout[0] * layout[1]


@pytest.fixture
def total_ranks(ranks_per_tile):
    return 6 * ranks_per_tile


@pytest.fixture
def tile_partitioner(layout):
    return fv3gfs.util.TilePartitioner(layout)


@pytest.fixture
def cube_partitioner(tile_partitioner):
    return fv3gfs.util.CubedSpherePartitioner(tile_partitioner)


@pytest.fixture
def communicator_list(cube_partitioner):
    shared_buffer = {}
    return_list = []
    for rank in range(cube_partitioner.total_ranks):
        return_list.append(
            fv3gfs.util.CubedSphereCommunicator(
                comm=fv3gfs.util.testing.DummyComm(
                    rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
                ),
                partitioner=cube_partitioner,
                timer=fv3gfs.util.Timer(),
            )
        )
    return return_list


@pytest.fixture
def gpu_communicators(cube_partitioner):
    shared_buffer = {}
    return_list = []
    for rank in range(cube_partitioner.total_ranks):
        return_list.append(
            fv3gfs.util.CubedSphereCommunicator(
                comm=fv3gfs.util.testing.DummyComm(
                    rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
                ),
                partitioner=cube_partitioner,
                force_cpu=False,
                timer=fv3gfs.util.Timer(),
            )
        )
    return return_list


@contextlib.contextmanager
def module_count_calls_to_empty(module):
    global n_calls
    n_calls = 0

    def count_calls(func):
        """Count np.empty call"""

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            global n_calls
            n_calls += 1
            return func(*args, **kwargs)

        return wrapped

    try:
        original = module.empty
        module.empty = count_calls(module.empty)
        yield n_calls
    finally:
        module.empty = original


@pytest.mark.parametrize("backend", ["gtcuda"])
def test_halo_update_only_communicate_on_gpu(backend, gpu_communicators):
    with module_count_calls_to_empty(np) as np_n_calls, module_count_calls_to_empty(
        cp
    ) as cp_n_calls:
        sizer = fv3gfs.util.SubtileGridSizer(
            nx=64, ny=64, nz=79, n_halo=3, extra_dim_lengths={}
        )
        quantity_factory = fv3gfs.util.QuantityFactory.from_backend(sizer, backend)
        quantity = quantity_factory.empty(
            [fv3gfs.util.Z_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM], units=""
        )

        req_list = []
        for communicator in gpu_communicators:
            req = communicator.start_halo_update(quantity, 3)
            req_list.append(req)
        for req in req_list:
            req.wait()

    # We expect no np calls and several cp calls
    assert np_n_calls == 0
    assert cp_n_calls > 9


@pytest.mark.parametrize("backend", ["gtcuda"])
def test_halo_update_communicate_though_cpu(backend, gpu_communicators):
    with module_count_calls_to_empty(np) as np_n_calls, module_count_calls_to_empty(
        cp
    ) as cp_n_calls:
        sizer = fv3gfs.util.SubtileGridSizer(
            nx=64, ny=64, nz=79, n_halo=3, extra_dim_lengths={}
        )
        quantity_factory = fv3gfs.util.QuantityFactory.from_backend(sizer, backend)
        quantity = quantity_factory.empty(
            [fv3gfs.util.Z_DIM, fv3gfs.util.Y_DIM, fv3gfs.util.X_DIM], units=""
        )

        req_list = []
        for communicator in gpu_communicators:
            req = communicator.start_halo_update(quantity, 3)
            req_list.append(req)
        for req in req_list:
            req.wait()

    # We expect no np calls and several cp calls
    assert np_n_calls > 0
    assert cp_n_calls > 9
