from typing import Tuple, Mapping, Optional, Sequence, cast, List, Dict
from .quantity import Quantity, QuantityMetadata
from .partitioner import CubedSpherePartitioner, TilePartitioner, Partitioner
from . import constants
from .boundary import Boundary
from .rotate import rotate_scalar_data
from .buffer import array_buffer, send_buffer, recv_buffer, Buffer
from ._timing import Timer, NullTimer
from .types import AsyncRequest, NumpyModule
from .utils import device_synchronize
from .message import MessageBundle
import logging
import numpy as np

__all__ = [
    "TileCommunicator",
    "CubedSphereCommunicator",
    "Communicator",
    "HaloUpdateRequest",
]

logger = logging.getLogger("fv3gfs.util")

_HaloSendTuple = Tuple[AsyncRequest, Buffer]
_HaloRequestSendList = List[_HaloSendTuple]
_HaloRecvTuple = Tuple[AsyncRequest, Buffer, np.ndarray]
_HaloRequestRecvList = List[_HaloRecvTuple]


def bcast_metadata_list(comm, quantity_list):
    is_root = comm.Get_rank() == constants.ROOT_RANK
    if is_root:
        metadata_list = []
        for quantity in quantity_list:
            metadata_list.append(quantity.metadata)
    else:
        metadata_list = None
    return comm.bcast(metadata_list, root=constants.ROOT_RANK)


def bcast_metadata(comm, array):
    return bcast_metadata_list(comm, [array])[0]


class HaloUpdateRequest:
    """Asynchronous request object for halo updates."""

    def __init__(
        self,
        send_data: _HaloRequestSendList,
        recv_data: _HaloRequestRecvList,
        timer: Optional[Timer] = None,
    ):
        """Build a halo request.
        Args:
            send_data: a tuple of the MPI request and the buffer sent
            recv_data: a tuple of the MPI request, the temporary message buffer and
            the destination buffer
            timer: optional, time the wait & unpack of a halo exchange
        """
        self._send_data = send_data
        self._recv_data = recv_data
        self._timer: Timer = timer if timer is not None else NullTimer()

    def wait(self):
        """Wait & unpack data into destination buffers
        Clean up by inserting back all buffers back in cache
        for potential reuse
        """
        for request, transfer_buffer in self._send_data:
            with self._timer.clock("wait"):
                request.wait()
            with self._timer.clock("unpack"):
                Buffer.push_to_cache(transfer_buffer)
        for request, transfer_buffer, destination_array in self._recv_data:
            with self._timer.clock("wait"):
                request.wait()
            with self._timer.clock("unpack"):
                transfer_buffer.assign_to(destination_array)
                Buffer.push_to_cache(transfer_buffer)


class HaloUpdateRequestMessage:
    """Asynchronous request object for halo updates."""

    def __init__(
        self,
        send_requests: List[AsyncRequest],
        recv_requests: List[AsyncRequest],
        messages: Dict[int, MessageBundle],
        timer: Optional[Timer] = None,
    ):
        """Build a halo request.

        Args:
            send_data: a tuple of the MPI request and the buffer sent
            recv_data: a tuple of the MPI request, the temporary message buffer and
            the destination buffer
            timer: optional, time the wait & unpack of a halo exchange

        """
        self._send_requests = send_requests
        self._recv_requests = recv_requests
        self._messages = messages
        self._timer: Timer = timer if timer is not None else NullTimer()

    def wait(self):
        """Wait & unpack data into destination buffers

        Clean up by inserting back all buffers back in cache
        for potential reuse
        """
        with self._timer.clock("wait"):
            for send_req in self._send_requests:
                send_req.wait()
            for recv_req in self._recv_requests:
                recv_req.wait()
        with self._timer.clock("unpack"):
            for _to_rank, message in self._messages.items():
                message.async_unpack()
            for _to_rank, message in self._messages.items():
                message.synchronize()


class Communicator:
    def __init__(self, comm, partitioner, force_cpu: bool = False):
        self.comm = comm
        self.partitioner: Partitioner = partitioner
        self._force_cpu = force_cpu

    @property
    def rank(self) -> int:
        """rank of the current process within this communicator"""
        return self.comm.Get_rank()

    def _maybe_force_cpu(self, module: NumpyModule) -> NumpyModule:
        """Get a numpy-like module depending on configuration and Quantity original allocator"""
        if self._force_cpu:
            return np
        return module

    def _Scatter(self, numpy_module, sendbuf, recvbuf, **kwargs):
        with send_buffer(numpy_module.empty, sendbuf) as send, recv_buffer(
            numpy_module.empty, recvbuf
        ) as recv:
            self.comm.Scatter(send, recv, **kwargs)

    def _Gather(self, numpy_module, sendbuf, recvbuf, **kwargs):
        with send_buffer(numpy_module.empty, sendbuf) as send, recv_buffer(
            numpy_module.empty, recvbuf
        ) as recv:
            self.comm.Gather(send, recv, **kwargs)

    def scatter(
        self,
        send_quantity: Optional[Quantity] = None,
        recv_quantity: Optional[Quantity] = None,
    ) -> Quantity:
        """Transfer subtile regions of a full-tile quantity
        from the tile root rank to all subtiles.

        Args:
            send_quantity: quantity to send, only required/used on the tile root rank
            recv_quantity: if provided, assign received data into this Quantity.
        Returns:
            recv_quantity
        """
        if self.rank == constants.ROOT_RANK and send_quantity is None:
            raise TypeError("send_quantity is a required argument on the root rank")
        if self.rank == constants.ROOT_RANK:
            send_quantity = cast(Quantity, send_quantity)
            metadata = self.comm.bcast(send_quantity.metadata, root=constants.ROOT_RANK)
        else:
            metadata = self.comm.bcast(None, root=constants.ROOT_RANK)
        shape = self.partitioner.subtile_extent(metadata)
        if recv_quantity is None:
            recv_quantity = self._get_scatter_recv_quantity(shape, metadata)
        if self.rank == constants.ROOT_RANK:
            send_quantity = cast(Quantity, send_quantity)
            with array_buffer(
                self._maybe_force_cpu(metadata.np).empty,
                (self.partitioner.total_ranks,) + shape,
                dtype=metadata.dtype,
            ) as sendbuf:
                for rank in range(0, self.partitioner.total_ranks):
                    subtile_slice = self.partitioner.subtile_slice(
                        rank,
                        global_dims=metadata.dims,
                        global_extent=metadata.extent,
                        overlap=True,
                    )
                    sendbuf.assign_from(
                        send_quantity.view[subtile_slice],
                        buffer_slice=np.index_exp[rank, :],
                    )
                self._Scatter(
                    metadata.np,
                    sendbuf.array,
                    recv_quantity.view[:],
                    root=constants.ROOT_RANK,
                )
        else:
            self._Scatter(
                metadata.np, None, recv_quantity.view[:], root=constants.ROOT_RANK,
            )
        return recv_quantity

    def _get_gather_recv_quantity(
        self, global_extent: Sequence[int], send_metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving global data during gather"""
        recv_quantity = Quantity(
            send_metadata.np.empty(global_extent, dtype=send_metadata.dtype),
            dims=send_metadata.dims,
            units=send_metadata.units,
            origin=tuple([0 for dim in send_metadata.dims]),
            extent=global_extent,
            gt4py_backend=send_metadata.gt4py_backend,
        )
        return recv_quantity

    def _get_scatter_recv_quantity(
        self, shape: Sequence[int], send_metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving subtile data during scatter"""
        recv_quantity = Quantity(
            send_metadata.np.empty(shape, dtype=send_metadata.dtype),
            dims=send_metadata.dims,
            units=send_metadata.units,
            gt4py_backend=send_metadata.gt4py_backend,
        )
        return recv_quantity

    def gather(
        self, send_quantity: Quantity, recv_quantity: Quantity = None
    ) -> Optional[Quantity]:
        """Transfer subtile regions of a full-tile quantity
        from each rank to the tile root rank.

        Args:
            send_quantity: quantity to send
            recv_quantity: if provided, assign received data into this Quantity (only
                used on the tile root rank)
        Returns:
            recv_quantity: quantity if on root rank, otherwise None
        """
        result: Optional[Quantity]
        if self.rank == constants.ROOT_RANK:
            with array_buffer(
                send_quantity.np.empty,
                (self.partitioner.total_ranks,) + tuple(send_quantity.extent),
                dtype=send_quantity.data.dtype,
            ) as recvbuf:
                self._Gather(
                    send_quantity.np,
                    send_quantity.view[:],
                    recvbuf.array,
                    root=constants.ROOT_RANK,
                )
                if recv_quantity is None:
                    global_extent = self.partitioner.global_extent(
                        send_quantity.metadata
                    )
                    recv_quantity = self._get_gather_recv_quantity(
                        global_extent, send_quantity.metadata
                    )
                for rank in range(self.partitioner.total_ranks):
                    to_slice = self.partitioner.subtile_slice(
                        rank,
                        global_dims=recv_quantity.dims,
                        global_extent=recv_quantity.extent,
                        overlap=True,
                    )
                    recvbuf.assign_to(
                        recv_quantity.view[to_slice], buffer_slice=np.index_exp[rank, :]
                    )
                result = recv_quantity
        else:
            self._Gather(
                send_quantity.np, send_quantity.view[:], None, root=constants.ROOT_RANK,
            )
            result = None
        return result

    def gather_state(self, send_state=None, recv_state=None):
        """Transfer a state dictionary from subtile ranks to the tile root rank.

        'time' is assumed to be the same on all ranks, and its value will be set
        to the value from the root rank.

        Args:
            send_state: the model state to be sent containing the subtile data
            recv_state: the pre-allocated state in which to recieve the full tile
                state. Only variables which are scattered will be written to.
        Returns:
            recv_state: on the root rank, the state containing the entire tile
        """
        if self.rank == constants.ROOT_RANK and recv_state is None:
            recv_state = {}
        for name, quantity in send_state.items():
            if name == "time":
                if self.rank == constants.ROOT_RANK:
                    recv_state["time"] = send_state["time"]
            else:
                if recv_state is not None and name in recv_state:
                    tile_quantity = self.gather(
                        quantity, recv_quantity=recv_state[name]
                    )
                else:
                    tile_quantity = self.gather(quantity)
                if self.rank == constants.ROOT_RANK:
                    recv_state[name] = tile_quantity
        return recv_state

    def scatter_state(self, send_state=None, recv_state=None):
        """Transfer a state dictionary from the tile root rank to all subtiles.

        Args:
            send_state: the model state to be sent containing the entire tile,
                required only from the root rank
            recv_state: the pre-allocated state in which to recieve the scattered
                state. Only variables which are scattered will be written to.
        Returns:
            rank_state: the state corresponding to this rank's subdomain
        """

        def scatter_root():
            if send_state is None:
                raise TypeError("send_state is a required argument on the root rank")
            name_list = list(send_state.keys())
            while "time" in name_list:
                name_list.remove("time")
            name_list = self.comm.bcast(name_list, root=constants.ROOT_RANK)
            array_list = [send_state[name] for name in name_list]
            for name, array in zip(name_list, array_list):
                if name in recv_state:
                    self.scatter(send_quantity=array, recv_quantity=recv_state[name])
                else:
                    recv_state[name] = self.scatter(send_quantity=array)
            recv_state["time"] = self.comm.bcast(
                send_state.get("time", None), root=constants.ROOT_RANK
            )

        def scatter_client():
            name_list = self.comm.bcast(None, root=constants.ROOT_RANK)
            for name in name_list:
                if name in recv_state:
                    self.scatter(recv_quantity=recv_state[name])
                else:
                    recv_state[name] = self.scatter()
            recv_state["time"] = self.comm.bcast(None, root=constants.ROOT_RANK)

        if recv_state is None:
            recv_state = {}
        if self.rank == constants.ROOT_RANK:
            scatter_root()
        else:
            scatter_client()
        if recv_state["time"] is None:
            recv_state.pop("time")
        return recv_state


class TileCommunicator(Communicator):
    """Performs communications within a single tile or region of a tile"""

    def __init__(self, comm, partitioner: TilePartitioner, force_cpu: bool = False):
        """Initialize a TileCommunicator.

        Args:
            comm: communication object behaving like mpi4py.Comm
            partitioner: tile partitioner
            force_cpu: force all communication to go through central memory
        """
        super(TileCommunicator, self).__init__(comm, partitioner, force_cpu=force_cpu)
        self.partitioner: TilePartitioner = partitioner


class CubedSphereCommunicator(Communicator):
    """Performs communications within a cubed sphere"""

    timer: Timer
    partitioner: CubedSpherePartitioner

    def __init__(
        self,
        comm,
        partitioner: CubedSpherePartitioner,
        force_cpu: bool = False,
        timer: Optional[Timer] = None,
    ):
        """Initialize a CubedSphereCommunicator.

        Args:
            comm: mpi4py.Comm object
            partitioner: cubed sphere partitioner
            force_cpu: Force all communication to go through central memory. Optional.
            timer: Time communication operations. Optional.
        """
        self.timer: Timer = timer if timer is not None else NullTimer()
        self._tile_communicator: Optional[TileCommunicator] = None
        self._boundaries: Optional[Mapping[int, Boundary]] = None
        self._last_halo_tag = 0
        self._force_cpu = force_cpu
        super(CubedSphereCommunicator, self).__init__(comm, partitioner, force_cpu)
        self.partitioner: CubedSpherePartitioner = partitioner

    def _get_halo_tag(self) -> int:
        self._last_halo_tag += 1
        return self._last_halo_tag

    @property
    def boundaries(self) -> Mapping[int, Boundary]:
        """boundaries of this tile with neighboring tiles"""
        if self._boundaries is None:
            self._boundaries = {}
            for boundary_type in constants.BOUNDARY_TYPES:
                boundary = self.partitioner.boundary(boundary_type, self.rank)
                if boundary is not None:
                    self._boundaries[boundary_type] = boundary
        return self._boundaries

    @property
    def tile(self) -> TileCommunicator:
        """communicator for within a tile"""
        if self._tile_communicator is None:
            self._initialize_tile_communicator()
        return cast(TileCommunicator, self._tile_communicator)

    def _initialize_tile_communicator(self):
        tile_comm = self.comm.Split(
            color=self.partitioner.tile_index(self.rank), key=self.rank
        )
        self._tile_communicator = TileCommunicator(tile_comm, self.partitioner.tile)

    def _get_gather_recv_quantity(
        self, global_extent: Sequence[int], metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving global data during gather

        Args:
            shape: ndarray shape, numpy-style
            metadata: metadata to the created Quantity
        """
        # needs to change the quantity dimensions since we add a "tile" dimension,
        # unlike for tile scatter/gather which retains the same dimensions
        recv_quantity = Quantity(
            metadata.np.empty(global_extent, dtype=metadata.dtype),
            dims=(constants.TILE_DIM,) + metadata.dims,
            units=metadata.units,
            origin=(0,) + tuple([0 for dim in metadata.dims]),
            extent=global_extent,
            gt4py_backend=metadata.gt4py_backend,
        )
        return recv_quantity

    def _get_scatter_recv_quantity(
        self, shape: Sequence[int], metadata: QuantityMetadata
    ) -> Quantity:
        """Initialize a Quantity for use when receiving subtile data during scatter

        Args:
            shape: ndarray shape, numpy-style
            metadata: metadata to the created Quantity
        """
        # needs to change the quantity dimensions since we remove a "tile" dimension,
        # unlike for tile scatter/gather which retains the same dimensions
        recv_quantity = Quantity(
            metadata.np.empty(shape, dtype=metadata.dtype),
            dims=metadata.dims[1:],
            units=metadata.units,
            gt4py_backend=metadata.gt4py_backend,
        )
        return recv_quantity

    def halo_update(self, quantity: Quantity, n_points: int):
        self.halo_update_aggregate([quantity], n_points)

    def halo_update_aggregate(self, quantities: List[Quantity], n_points: int):
        """Perform a halo update on a quantity.

        Args:
            quantity: the quantity to be updated
            n_points: how many halo points to update, starting from the interior
        """
        req = self.start_halo_update(quantities, n_points)
        req.wait()

    @staticmethod
    def _device_synchronize():
        """Wait for all work that could be in-flight to finish."""
        # this is a method so we can profile it separately from other device syncs
        device_synchronize()

    def start_halo_update(self, quantity: Quantity, n_points: int) -> HaloUpdateRequest:
        return self.start_halo_update_aggregate([quantity], n_points)

    def start_halo_update_aggregate(
        self, quantities: List[Quantity], n_points: int
    ) -> HaloUpdateRequest:
        """Start an asynchronous halo update on a quantity.

        Args:
            quantity: the quantity to be updated
            n_points: how many halo points to update, starting from the interior

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if n_points == 0:
            raise ValueError("cannot perform a halo update on zero halo points")
        CubedSphereCommunicator._device_synchronize()
        tag = self._get_halo_tag()

        # Prepare messages
        # - message dict index on rank of receiver
        # - queue for packing
        # - allocate internal message memory
        messages = {}
        with self.timer.clock("pack"):
            for boundary in self.boundaries.values():
                for quantity in quantities:
                    message = self._lazy_get_messages(messages, boundary, quantity)
                    self._queue_scalar(message, quantity, boundary, n_points)
            self._allocate_messages(messages)

        # Issue asynchroneous transfer commands
        # Includes pre-network call message packing
        return self._Isend_Irecv_halos(messages, tag)

    def _lazy_get_messages(
        self, messages: Dict[int, MessageBundle], boundary: Boundary, quantity: Quantity
    ) -> MessageBundle:
        if boundary.to_rank not in messages:
            messages[boundary.to_rank] = MessageBundle.get_from_quantity_module(
                self._maybe_force_cpu(quantity.np)
            )
        return messages[boundary.to_rank]

    def _queue_scalar(
        self,
        message: MessageBundle,
        quantity: Quantity,
        boundary: Boundary,
        n_points: int,
    ):
        message.queue_scalar_message(
            quantity,
            boundary.send_slice(quantity, n_points),
            boundary.n_clockwise_rotations,
            boundary.recv_slice(quantity, n_points),
        )

    def _allocate_messages(self, messages: Dict[int, MessageBundle]):
        for _to_rank, message in messages.items():
            message.allocate()

    def _Isend_Irecv_halos(
        self, messages: Dict[int, MessageBundle], tag: int
    ) -> HaloUpdateRequest:
        with self.timer.clock("Irecv"):
            recv_requests = []
            for to_rank, message in messages.items():
                recv_requests.append(
                    self.comm.Irecv(
                        message.get_recv_buffer().array, source=to_rank, tag=tag,
                    )
                )
        send_requests = []
        for to_rank, message in messages.items():
            with self.timer.clock("pack"):
                message.async_pack()
        for to_rank, message in messages.items():
            with self.timer.clock("Isend"):
                message.synchronize()
                send_requests.append(
                    self.comm.Isend(
                        message.get_send_buffer().array, dest=to_rank, tag=tag
                    )
                )
        return HaloUpdateRequestMessage(
            send_requests, recv_requests, messages, self.timer
        )

    def finish_halo_update(self, quantity: Quantity, n_points: int):
        """Deprecated, do not use."""
        raise NotImplementedError(
            "finish_halo_update has been removed, use .wait() on the request object "
            "returned by start_halo_update"
        )

    def vector_halo_update(
        self, x_quantity: Quantity, y_quantity: Quantity, n_points: int,
    ):
        self.vector_halo_update_aggregate([x_quantity], [y_quantity], n_points)

    def vector_halo_update_aggregate(
        self, x_quantities: List[Quantity], y_quantities: List[Quantity], n_points: int,
    ):
        """Perform a halo update of a horizontal vector quantity.

        Assumes the x and y dimension indices are the same between the two quantities.

        Args:
            x_quantity: the x-component quantity to be halo updated
            y_quantity: the y-component quantity to be halo updated
            n_points: how many halo points to update, starting at the interior
        """
        req = self.start_vector_halo_update_aggregate(
            x_quantities, y_quantities, n_points
        )
        req.wait()

    def start_synchronize_vector_interfaces(
        self, x_quantity: Quantity, y_quantity: Quantity
    ):
        """
        Synchronize shared points at the edges of a vector interface variable.

        Sends the values on the south and west edges to overwrite the values on adjacent
        subtiles. Vector must be defined on the Arakawa C grid.

        For interface variables, the edges of the tile are computed on both ranks
        bordering that edge. This routine copies values across those shared edges
        so that both ranks have the same value for that edge. It also handles any
        rotation of vector quantities needed to move data across the edge.

        Args:
            x_quantity: the x-component quantity to be synchronized
            y_quantity: the y-component quantity to be synchronized

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if not on_c_grid(x_quantity, y_quantity):
            raise ValueError("vector must be defined on Arakawa C-grid")
        CubedSphereCommunicator._device_synchronize()
        tag = self._get_halo_tag()
        send_requests = self._Isend_vector_shared_boundary(
            x_quantity, y_quantity, tag=tag
        )
        recv_requests = self._Irecv_vector_shared_boundary(
            x_quantity, y_quantity, tag=tag
        )
        return HaloUpdateRequest(send_requests, recv_requests, self.timer)

    def synchronize_vector_interfaces(self, x_quantity: Quantity, y_quantity: Quantity):
        """
        Synchronize shared points at the edges of a vector interface variable.

        Sends the values on the south and west edges to overwrite the values on adjacent
        subtiles. Vector must be defined on the Arakawa C grid.

        For interface variables, the edges of the tile are computed on both ranks
        bordering that edge. This routine copies values across those shared edges
        so that both ranks have the same value for that edge. It also handles any
        rotation of vector quantities needed to move data across the edge.

        Args:
            x_quantity: the x-component quantity to be synchronized
            y_quantity: the y-component quantity to be synchronized
        """
        req = self.start_synchronize_vector_interfaces(x_quantity, y_quantity)
        req.wait()

    def start_vector_halo_update(
        self, x_quantity: Quantity, y_quantity: Quantity, n_points: int,
    ):
        return self.start_vector_halo_update_aggregate(
            [x_quantity], [y_quantity], n_points
        )

    def start_vector_halo_update_aggregate(
        self, x_quantities: List[Quantity], y_quantities: List[Quantity], n_points: int,
    ) -> HaloUpdateRequest:
        """Start an asynchronous halo update of a horizontal vector quantity.

        Assumes the x and y dimension indices are the same between the two quantities.

        Args:
            x_quantity: the x-component quantity to be halo updated
            y_quantity: the y-component quantity to be halo updated
            n_points: how many halo points to update, starting at the interior

        Returns:
            request: an asynchronous request object with a .wait() method
        """
        if n_points == 0:
            raise ValueError("cannot perform a halo update on zero halo points")
        CubedSphereCommunicator._device_synchronize()
        tag = self._get_halo_tag()

        # Prepare messages
        # - message dict index on rank of receiver
        # - queue for packing
        # - allocate internal message memory
        messages = {}
        with self.timer.clock("pack"):
            for boundary in self.boundaries.values():
                for x_quantity, y_quantity in zip(x_quantities, y_quantities):
                    message = self._lazy_get_messages(messages, boundary, x_quantity)
                    self._queue_vector(
                        message, x_quantity, y_quantity, boundary, n_points
                    )
            self._allocate_messages(messages)

        # Issue asynchroneous transfer commands
        # Includes pre-network call message packing
        return self._Isend_Irecv_halos(messages, tag)

    def _queue_vector(
        self,
        message: MessageBundle,
        x_quantity: Quantity,
        y_quantity: Quantity,
        boundary: Boundary,
        n_points: int,
    ):
        message.queue_vector_message(
            x_quantity,
            boundary.send_slice(x_quantity, n_points),
            y_quantity,
            boundary.send_slice(y_quantity, n_points),
            boundary.n_clockwise_rotations,
            boundary.recv_slice(x_quantity, n_points),
            boundary.recv_slice(y_quantity, n_points),
        )

    def _Isend_vector_shared_boundary(
        self, x_quantity, y_quantity, tag=0
    ) -> _HaloRequestSendList:
        south_boundary = self.boundaries[constants.SOUTH]
        west_boundary = self.boundaries[constants.WEST]
        south_data = x_quantity.view.southwest.sel(
            **{
                constants.Y_INTERFACE_DIM: 0,
                constants.X_DIM: slice(
                    0, x_quantity.extent[x_quantity.dims.index(constants.X_DIM)]
                ),
            }
        )
        south_data = rotate_scalar_data(
            south_data,
            [constants.X_DIM],
            x_quantity.np,
            -south_boundary.n_clockwise_rotations,
        )
        if south_boundary.n_clockwise_rotations in (3, 2):
            south_data = -south_data
        west_data = y_quantity.view.southwest.sel(
            **{
                constants.X_INTERFACE_DIM: 0,
                constants.Y_DIM: slice(
                    0, y_quantity.extent[y_quantity.dims.index(constants.Y_DIM)]
                ),
            }
        )
        west_data = rotate_scalar_data(
            west_data,
            [constants.Y_DIM],
            y_quantity.np,
            -west_boundary.n_clockwise_rotations,
        )
        if west_boundary.n_clockwise_rotations in (1, 2):
            west_data = -west_data
        send_requests = [
            self._Isend(
                self._maybe_force_cpu(x_quantity.np),
                south_data,
                dest=south_boundary.to_rank,
                tag=tag,
            ),
            self._Isend(
                self._maybe_force_cpu(y_quantity.np),
                west_data,
                dest=west_boundary.to_rank,
                tag=tag,
            ),
        ]
        return send_requests

    def _Irecv_vector_shared_boundary(
        self, x_quantity, y_quantity, tag=0
    ) -> _HaloRequestRecvList:
        north_rank = self.boundaries[constants.NORTH].to_rank
        east_rank = self.boundaries[constants.EAST].to_rank
        north_data = x_quantity.view.northwest.sel(
            **{
                constants.Y_INTERFACE_DIM: -1,
                constants.X_DIM: slice(
                    0, x_quantity.extent[x_quantity.dims.index(constants.X_DIM)]
                ),
            }
        )
        east_data = y_quantity.view.southeast.sel(
            **{
                constants.X_INTERFACE_DIM: -1,
                constants.Y_DIM: slice(
                    0, y_quantity.extent[y_quantity.dims.index(constants.Y_DIM)]
                ),
            }
        )
        recv_requests = [
            self._Irecv(
                self._maybe_force_cpu(x_quantity.np),
                north_data,
                source=north_rank,
                tag=tag,
            ),
            self._Irecv(
                self._maybe_force_cpu(y_quantity.np),
                east_data,
                source=east_rank,
                tag=tag,
            ),
        ]
        return recv_requests

    def _Isend(self, numpy_module, in_array, **kwargs) -> _HaloSendTuple:
        # copy the resulting view in a contiguous array for transfer
        with self.timer.clock("pack"):
            buffer = Buffer.pop_from_cache(
                numpy_module.empty, in_array.shape, in_array.dtype
            )
            buffer.assign_from(in_array)
            buffer.finalize_memory_transfer()
        with self.timer.clock("Isend"):
            request = self.comm.Isend(buffer.array, **kwargs)
        return (request, buffer)

    def _Irecv(self, numpy_module, out_array, **kwargs) -> _HaloRecvTuple:
        # Prepare a contiguous buffer to receive data
        with self.timer.clock("Irecv"):
            buffer = Buffer.pop_from_cache(
                numpy_module.empty, out_array.shape, out_array.dtype
            )
            recv_request = self.comm.Irecv(buffer.array, **kwargs)
        return (recv_request, buffer, out_array)

    def finish_vector_halo_update(
        self, x_quantity: Quantity, y_quantity: Quantity, n_points: int,
    ):
        """Deprecated, do not use."""
        raise NotImplementedError(
            "finish_vector_halo_update has been removed, use .wait() on the request object "
            "returned by start_vector_halo_update"
        )


def on_c_grid(x_quantity, y_quantity):
    if (
        constants.X_DIM not in x_quantity.dims
        or constants.Y_INTERFACE_DIM not in x_quantity.dims
    ):
        return False
    if (
        constants.Y_DIM not in y_quantity.dims
        or constants.X_INTERFACE_DIM not in y_quantity.dims
    ):
        return False
    else:
        return True
