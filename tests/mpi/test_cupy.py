from mpi4py import MPI
try:
    import cupy
except ModuleNotFoundError:
    cupy = None


def test_cupy_send_array():
    comm = MPI.COMM_WORLD
    arr = cupy.zeros([5, 10])
    arr.device.synchronize()
    if comm.Get_rank() == 0:
        other_rank = 1
    else:
        other_rank = 0
    req = comm.Isend(arr, dest=other_rank, tag=0)
    arr_recv = cupy.empty_like(arr)
    comm.Recv(arr_recv, source=other_rank, tag=0)

def test_cupy_send_slice():
    comm = MPI.COMM_WORLD
    arr = cupy.zeros([10, 10])
    arr.ravel()[:] = cupy.arange(100)
    if comm.Get_rank() == 0:
        other_rank = 1
    else:
        other_rank = 0
    arr_slice = arr[3:-3, :3]
    arr_slice_rot = cupy.rot90(arr_slice)
    arr_contig = cupy.ascontiguousarray(arr_slice_rot)
    req = comm.Isend(arr_contig, dest=other_rank, tag=0)
    arr_recv = cupy.empty_like(arr_contig)
    comm.Recv(arr_recv, source=other_rank, tag=0)
    arr_contig.device.synchronize()
