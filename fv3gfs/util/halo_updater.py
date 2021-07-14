from .types import NumpyModule, AsyncRequest
from .halo_data_transformer import HaloDataTransformer, HaloUpdateSpec
from ._timing import Timer, NullTimer
from .boundary import Boundary
from typing import Dict, Iterable, List, Optional, TYPE_CHECKING
from .quantity import Quantity

if TYPE_CHECKING:
    from .communicator import Communicator


class HaloTransformerDict(dict):
    """Dict that knows how to construct HaloDataTransformer"""

    def __init__(self, numpy_module):
        self._numpy_module = numpy_module

    def __missing__(self, key):
        ret = self[key] = HaloDataTransformer.get_from_numpy_module(self._numpy_module)
        return ret


class HaloUpdater:
    """Exchange halo information between ranks.

    The class is responsible for the entire exchange and uses the __init__
    to precompute the maximum of information to have minimum overhead at runtime.
    Therefore it should be cached for early and re-used at runtime.

    - from_scalar_specifications/from_vector_specifications are used to create an HaloUpdater
      from a list of memory specifications
    - update and start/wait trigger the halo exchange
    - the class creates a "pattern" of exchange that can fit any memory given to do/start
    - temporary reference is helf between start and wait
    """

    def __init__(
        self,
        comm: "Communicator",
        tag: int,
        transformers: Dict[int, HaloDataTransformer],
        timer: Timer,
    ):
        self._comm = comm
        self._tag = tag
        self._transformers = transformers
        self._timer = timer
        self._recv_requests: List[AsyncRequest] = []
        self._send_requests: List[AsyncRequest] = []
        self._inflight_x_quantities: Optional[List[Quantity]] = None
        self._inflight_y_quantities: Optional[List[Quantity]] = None

    def __del__(self):
        """Clean up all buffers on garbage collection"""
        if (
            self._inflight_x_quantities is not None
            or self._inflight_y_quantities is not None
        ):
            raise RuntimeError(
                "An halo exchange wasn't completed and a wait() call was expected"
            )
        for _to_rank, buffer in self._transformers:
            buffer.finalize()

    @classmethod
    def from_scalar_specifications(
        cls,
        comm: "Communicator",
        numpy_like_module: NumpyModule,
        specifications: Iterable[HaloUpdateSpec],
        boundaries: Iterable[Boundary],
        tag: int,
        optional_timer: Optional[Timer] = None,
    ) -> "HaloUpdater":
        """Create/retrieve as many packed buffer as needed and queue the slices to exchange.
        
        Args:
            comm: communicator to post network messages
            numpy_like_module: module implementing numpy API
            specifications: data specifications to exchange, including number of halo points
            boundaries: informations on the exchange boundaries.
            tag: network tag (to differentiate messaging) for this node.
            optional_timer: timing of operations.
        
        Returns:
            HaloUpdater ready to exchange data.
        """

        timer = optional_timer if optional_timer is not None else NullTimer()

        transformers: Dict[int, HaloDataTransformer] = HaloTransformerDict(
            numpy_like_module
        )
        for boundary in boundaries:
            for specification in specifications:
                transformers[boundary.to_rank].add_scalar_specification(
                    specification,
                    boundary.send_slice(specification),
                    boundary.n_clockwise_rotations,
                    boundary.recv_slice(specification),
                )

        for transformer in transformers.values():
            transformer.compile()

        return cls(comm, tag, transformers, timer)

    @classmethod
    def from_vector_specifications(
        cls,
        comm: "Communicator",
        numpy_like_module: NumpyModule,
        specifications_x: Iterable[HaloUpdateSpec],
        specifications_y: Iterable[HaloUpdateSpec],
        boundaries: Iterable[Boundary],
        tag: int,
        optional_timer: Optional[Timer] = None,
    ) -> "HaloUpdater":
        """Create/retrieve as many packed buffer as needed and queue the slices to exchange.

        Args:
            comm: communicator to post network messages
            numpy_like_module: module implementing numpy API
            specifications_x: specifications to exchange along the x axis.
                          Length must match y specifications.
            specifications_y: specifications to exchange along the y axis.
                          Length must match x specifications.
            boundaries: informations on the exchange boundaries.
            tag: network tag (to differentiate messaging) for this node.
            n_halo_points: size of the halo to exchange.
            optional_timer: timing of operations.
        
        Returns:
            HaloUpdater ready to exchange data.
        """
        timer = optional_timer if optional_timer is not None else NullTimer()

        transformers: Dict[int, HaloDataTransformer] = HaloTransformerDict(
            numpy_like_module
        )
        for boundary in boundaries:
            for specification_x, specification_y in zip(
                specifications_x, specifications_y
            ):
                transformers[boundary.to_rank].add_vector_specification(
                    specification_x,
                    specification_y,
                    boundary.send_slice(specification_x),
                    boundary.send_slice(specification_y),
                    boundary.n_clockwise_rotations,
                    boundary.recv_slice(specification_x),
                    boundary.recv_slice(specification_y),
                )

        for transformer in transformers.values():
            transformer.compile()

        return cls(comm, tag, transformers, timer)

    def update(
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
            for to_rank, transformer in self._transformers.items():
                self._recv_requests.append(
                    self._comm.comm.Irecv(
                        transformer.get_unpack_buffer().array,
                        source=to_rank,
                        tag=self._tag,
                    )
                )

        # Pack quantities halo points data into buffers
        with self._timer.clock("pack"):
            for transformer in self._transformers.values():
                transformer.async_pack(quantities_x, quantities_y)
            for transformer in self._transformers.values():
                transformer.synchronize()

        self._inflight_x_quantities = quantities_x
        self._inflight_y_quantities = quantities_y

        # Post send MPI order
        with self._timer.clock("Isend"):
            self._send_requests = []
            for to_rank, transformer in self._transformers.items():
                self._send_requests.append(
                    self._comm.comm.Isend(
                        transformer.get_pack_buffer().array,
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
            for buffer in self._transformers.values():
                buffer.async_unpack(
                    self._inflight_x_quantities, self._inflight_y_quantities
                )
            for buffer in self._transformers.values():
                buffer.synchronize()

        self._inflight_x_quantities = None
        self._inflight_y_quantities = None
