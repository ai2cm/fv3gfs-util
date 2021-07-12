from .types import NumpyModule, AsyncRequest
from .halo_data_transformer import HaloDataTransformer, HaloUpdateSpec
from ._timing import Timer, NullTimer
from .boundary import Boundary
from typing import Iterable, Tuple, List, Optional, TYPE_CHECKING
from .quantity import Quantity

if TYPE_CHECKING:
    from .communicator import Communicator


class HaloUpdater:
    """Exchange halo information between ranks.

    The class is responsible for the entire exchange and uses the __init__
    to precompute the maximum of information to have minimum overhead at runtime.
    Therefore it should be cached for early and re-used at runtime.

    - from_scalar_specifications/from_vector_specifications are used to create an HaloUpdater
      from a list of memory specifications
    - do and start/wait trigger the halo exchange
    - the class creates a "pattern" of exchange that can fit any memory given to do/start
    - temporary reference is helf between start and wait
    """

    def __init__(
        self,
        comm: "Communicator",
        tag: int,
        packed_buffers: Iterable[Tuple[int, HaloDataTransformer]],
        timer: Timer,
    ):
        self._comm = comm
        self._tag = tag
        self._packed_buffers = packed_buffers
        self._timer = timer
        self._recv_requests: List[AsyncRequest] = []
        self._send_requests: List[AsyncRequest] = []
        self._inflight_x_quantities: Optional[List[Quantity]] = None
        self._inflight_y_quantities: Optional[List[Quantity]] = None

    def __del__(self):
        """Clean up all buffers on garbage collection"""
        assert self._inflight_x_quantities is None
        assert self._inflight_y_quantities is None
        for _to_rank, buffer in self._packed_buffers:
            buffer.finalize()

    @classmethod
    def from_scalar_specifications(
        cls,
        comm: "Communicator",
        numpy_like_module: NumpyModule,
        specifications: Iterable[HaloUpdateSpec],
        boundaries: Iterable[Boundary],
        tag: int,
        n_halo_points: int,
        optional_timer: Optional[Timer] = None,
    ) -> "HaloUpdater":
        """Create/retrieve as many packed buffer as needed and queue the slices to exchange.
        
        Args:
            comm: communicator to post network messages
            numpy_like_module: module implementing numpy API
            quantities: data to exchange.
            boundaries: informations on the exchange boundaries.
            tag: network tag (to differentiate messaging) for this node.
            n_halo_points: size of the halo to exchange.
            optional_timer: timing of operations.
        
        Returns:
            HaloUpdater ready to exchange data.
        """

        timer = optional_timer if optional_timer is not None else NullTimer()

        packed_buffers: List[Tuple[int, HaloDataTransformer]] = []
        for boundary in boundaries:
            for specification in specifications:
                buffer = cls._get_packed_buffer(
                    packed_buffers, numpy_like_module, boundary
                )
                buffer.add_scalar_specification(
                    specification,
                    boundary.send_slice(specification, n_halo_points),
                    boundary.n_clockwise_rotations,
                    boundary.recv_slice(specification, n_halo_points),
                )

        for _to_rank, buffer in packed_buffers:
            buffer.compile()

        return cls(comm, tag, packed_buffers, timer)

    @classmethod
    def from_vector_specifications(
        cls,
        comm: "Communicator",
        numpy_like_module: NumpyModule,
        specifications_x: Iterable[HaloUpdateSpec],
        specifications_y: Iterable[HaloUpdateSpec],
        boundaries: Iterable[Boundary],
        tag: int,
        n_halo_points: int,
        optional_timer: Optional[Timer] = None,
    ) -> "HaloUpdater":
        """Create/retrieve as many packed buffer as needed and queue the slices to exchange.

        Args:
            comm: communicator to post network messages
            numpy_like_module: module implementing numpy API
            quantities_x: quantities to exchange along the x axis.
                          Length must match y quantities.
            quantities_y: quantities to exchange along the y axis.
                          Length must match x quantities.
            boundaries: informations on the exchange boundaries.
            tag: network tag (to differentiate messaging) for this node.
            n_halo_points: size of the halo to exchange.
            optional_timer: timing of operations.
        
        Returns:
            HaloUpdater ready to exchange data.
        """
        timer = optional_timer if optional_timer is not None else NullTimer()

        packed_buffers: List[Tuple[int, HaloDataTransformer]] = []
        for boundary in boundaries:
            for specification_x, specification_y in zip(
                specifications_x, specifications_y
            ):
                buffer = cls._get_packed_buffer(
                    packed_buffers, numpy_like_module, boundary
                )
                buffer.add_vector_specification(
                    specification_x,
                    specification_y,
                    boundary.send_slice(specification_x, n_halo_points),
                    boundary.send_slice(specification_y, n_halo_points),
                    boundary.n_clockwise_rotations,
                    boundary.recv_slice(specification_x, n_halo_points),
                    boundary.recv_slice(specification_y, n_halo_points),
                )

        for _to_rank, buffer in packed_buffers:
            buffer.compile()

        return cls(comm, tag, packed_buffers, timer)

    @classmethod
    def _get_packed_buffer(
        cls,
        packed_buffer: List[Tuple[int, HaloDataTransformer]],
        numpy_like_module: NumpyModule,
        boundary: Boundary,
    ) -> HaloDataTransformer:
        """Returns the correct packed_buffer from the list create it first if need be.

        Args:
            packed_buffer: list of Tuple [target_rank_to_send_data, HaloDataTransformer]
            numpy_like_module: module implementing numpy API
            boundary: information on the exchange boundary

        Returns:
            Correct HaloDataTransformer for target rank
        """
        to_rank_packed_buffer = [x for x in packed_buffer if x[0] == boundary.to_rank]
        assert len(to_rank_packed_buffer) <= 1
        if len(to_rank_packed_buffer) == 0:
            packed_buffer.append(
                (
                    boundary.to_rank,
                    HaloDataTransformer.get_from_numpy_module(numpy_like_module),
                )
            )
            to_rank_packed_buffer = [
                x for x in packed_buffer if x[0] == boundary.to_rank
            ]
        return to_rank_packed_buffer[0][1]

    def do(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        """Exhange the data and blocks until finished."""
        self.start(quantities_x, quantities_y)
        self.wait()

    def start(
        self,
        quantities_x: List[Quantity],
        quantities_y: Optional[List[Quantity]] = None,
    ):
        """Start data exchange."""
        self._comm._device_synchronize()

        if (
            self._inflight_x_quantities is not None
            or self._inflight_y_quantities is not None
        ):
            raise RuntimeError(
                "Previous exchange hasn't been properly finished."
                "E.g. previous start() call didn't have a wait() call."
            )

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
                buffer.async_pack(quantities_x, quantities_y)
            for _to_rank, buffer in self._packed_buffers:
                buffer.synchronize()

        self._inflight_x_quantities = quantities_x
        self._inflight_y_quantities = quantities_y

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

    def wait(self):
        """Finalize data exchange."""
        if __debug__ and self._inflight_x_quantities is None:
            raise RuntimeError('Halo update "wait" call before "start"')

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
                buffer.async_unpack(
                    self._inflight_x_quantities, self._inflight_y_quantities
                )
            for _to_rank, buffer in self._packed_buffers:
                buffer.synchronize()

        self._inflight_x_quantities = None
        self._inflight_y_quantities = None
