from .types import NumpyModule, AsyncRequest
from .packed_buffer import PackedBuffer
from ._timing import Timer, NullTimer
from .boundary import Boundary
from typing import Iterable, Tuple, List, Optional, TYPE_CHECKING
from .quantity import Quantity

if TYPE_CHECKING:
    from .communicator import Communicator


class HaloUpdater:
    def __init__(
        self,
        comm: "Communicator",
        tag: int,
        packed_buffers: Iterable[Tuple[int, PackedBuffer]],
        timer: Timer,
    ):
        self._comm = comm
        self._tag = tag
        self._packed_buffers = packed_buffers
        self._timer = timer
        self._recv_requests: List[AsyncRequest] = []
        self._send_requests: List[AsyncRequest] = []

    def __del__(self):
        for _to_rank, buffer in self._packed_buffers:
            buffer.finalize()

    @classmethod
    def from_scalar_quantities(
        cls,
        comm: "Communicator",
        numpy_like_module: NumpyModule,
        quantities: Iterable[Quantity],
        boundaries: Iterable[Boundary],
        tag: int,
        n_halo_points: int,
        optional_timer: Optional[Timer] = None,
    ) -> "HaloUpdater":
        timer = optional_timer if optional_timer is not None else NullTimer()

        packed_buffers: List[Tuple[int, PackedBuffer]] = []
        for boundary in boundaries:
            for quantity in quantities:
                buffer = cls._lazy_get_packed_buffer(
                    packed_buffers, numpy_like_module, boundary
                )
                buffer.queue_scalar(
                    quantity,
                    boundary.send_slice(quantity, n_halo_points),
                    boundary.n_clockwise_rotations,
                    boundary.recv_slice(quantity, n_halo_points),
                )

        for _to_rank, buffer in packed_buffers:
            buffer.allocate()

        return cls(comm, tag, packed_buffers, timer)

    @classmethod
    def from_vector_quantities(
        cls,
        comm: "Communicator",
        numpy_like_module: NumpyModule,
        quantities_x: Iterable[Quantity],
        quantities_y: Iterable[Quantity],
        boundaries: Iterable[Boundary],
        tag: int,
        n_halo_points: int,
        optional_timer: Optional[Timer] = None,
    ) -> "HaloUpdater":
        timer = optional_timer if optional_timer is not None else NullTimer()

        packed_buffers: List[Tuple[int, PackedBuffer]] = []
        for boundary in boundaries:
            for quantity_x, quantity_y in zip(quantities_x, quantities_y):
                buffer = cls._lazy_get_packed_buffer(
                    packed_buffers, numpy_like_module, boundary
                )
                buffer.queue_vector(
                    quantity_x,
                    boundary.send_slice(quantity_x, n_halo_points),
                    quantity_y,
                    boundary.send_slice(quantity_y, n_halo_points),
                    boundary.n_clockwise_rotations,
                    boundary.recv_slice(quantity_x, n_halo_points),
                    boundary.recv_slice(quantity_y, n_halo_points),
                )

        for _to_rank, buffer in packed_buffers:
            buffer.allocate()

        return cls(comm, tag, packed_buffers, timer)

    # TODO SHARED CODE WITH COMMS
    @classmethod
    def _lazy_get_packed_buffer(
        cls,
        packed_buffer: List[Tuple[int, PackedBuffer]],
        numpy_like_module: NumpyModule,
        boundary: Boundary,
    ) -> PackedBuffer:
        to_rank_packed_buffer = [x for x in packed_buffer if x[0] == boundary.to_rank]
        assert len(to_rank_packed_buffer) <= 1
        if len(to_rank_packed_buffer) == 0:
            packed_buffer.append(
                (
                    boundary.to_rank,
                    PackedBuffer.get_from_quantity_module(numpy_like_module),
                )
            )
            to_rank_packed_buffer = [
                x for x in packed_buffer if x[0] == boundary.to_rank
            ]
        return to_rank_packed_buffer[0][1]

    def blocking_exchange(self):
        self.async_exchange_start()
        self.async_exchange_wait()

    def async_exchange_start(self):
        self._comm._device_synchronize()

        # Post recv MPI order
        with self._timer.clock("Irecv"):
            self._recv_requests = []
            for to_rank, packed_buffer in self._packed_buffers:
                self._recv_requests.append(
                    self._comm.comm.Irecv(
                        packed_buffer.get_recv_buffer().array,
                        source=to_rank,
                        tag=self._tag,
                    )
                )

        # Pack quantities halo points data into buffers
        with self._timer.clock("pack"):
            for _to_rank, buffer in self._packed_buffers:
                buffer.async_pack()
            for _to_rank, buffer in self._packed_buffers:
                buffer.synchronize()

        # Post send MPI order
        with self._timer.clock("Isend"):
            self._send_requests = []
            for to_rank, packed_buffer in self._packed_buffers:
                self._send_requests.append(
                    self._comm.comm.Isend(
                        packed_buffer.get_send_buffer().array,
                        dest=to_rank,
                        tag=self._tag,
                    )
                )

    def async_exchange_wait(self):
        # Wait message to be exchange
        with self._timer.clock("wait"):
            for send_req in self._send_requests:
                send_req.wait()
            for recv_req in self._recv_requests:
                recv_req.wait()

        # Unpack buffers (updated by MPI with neighbouring halos)
        # to proper quantities
        with self._timer.clock("unpack"):
            for _to_rank, buffer in self._packed_buffers:
                buffer.async_unpack()
            for _to_rank, buffer in self._packed_buffers:
                buffer.synchronize()
