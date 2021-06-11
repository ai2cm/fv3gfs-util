from dataclasses import dataclass
from .buffer import Buffer
from .quantity import Quantity
from .types import NumpyModule
from typing import List, Tuple
from enum import Enum
from .rotate import rotate_scalar_data, rotate_vector_data
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


def _slices_length(slices: Tuple[slice]) -> int:
    length = 1
    for s in slices:
        assert s.step is None
        length *= abs(s.start - s.stop)
    return length


@dataclass
class MessageMetadata:
    """Description of a single message."""

    quantity: Quantity
    send_slices: Tuple[slice]
    send_clockwise_rotation: int
    recv_slices: Tuple[slice]
    linear_size: int


class MessageBundleType(Enum):
    """Dimensionality of the message bundle."""

    UNKNOWN = 0
    SCALAR = 1
    VECTOR = 2


class MessageBundle:
    """Pack/unpack multiple nD messages in/from a single linear buffer."""

    _send_buffer: Buffer = None
    _recv_buffer: Buffer = None

    _x_infos: List[MessageMetadata]
    _y_infos: List[MessageMetadata]

    _type: MessageBundleType = MessageBundleType.UNKNOWN

    def __init__(self, np_module: NumpyModule) -> None:
        self._np_module = np_module
        self._x_infos = []
        self._y_infos = []
        assert self._send_buffer is None
        assert self._recv_buffer is None

    def __del__(self):
        assert self._send_buffer is None
        assert self._recv_buffer is None

    @staticmethod
    def get_from_quantity_module(qty_module: NumpyModule) -> "MessageBundle":
        if qty_module is np:
            return MessageBundleCPU(np)
        elif qty_module is cp:
            return MessageBundleGPU(cp)

        raise NotImplementedError(
            f"Quantity module {qty_module} has no MessageBundle implemented"
        )

    def queue_scalar_message(
        self,
        quantity: Quantity,
        send_slice: Tuple[slice],
        n_clockwise_rotation: int,
        recv_slice: Tuple[slice],
    ) -> None:
        """Add message information to the bundle.
        
        Send array & rotation parameter for later packing.
        Recv array for later unpacking. Pop a recv Buffer for transfer.
        Recv buffer is held by the class.
        The function will also compute & store the slicing of the arrays.
        """
        assert (
            self._type == MessageBundleType.SCALAR
            or self._type == MessageBundleType.UNKNOWN
        )
        if self._recv_buffer is not None and self._send_buffer is not None:
            raise RuntimeError(
                "Buffer previously allocated are not longer correct."
                "Make sure to queue all message before calling allocate."
            )

        self._type = MessageBundleType.SCALAR
        self._x_infos.append(
            MessageMetadata(
                quantity=quantity,
                send_slices=send_slice,
                linear_size=0,
                send_clockwise_rotation=n_clockwise_rotation,
                recv_slices=recv_slice,
            )
        )

    def queue_vector_message(
        self,
        x_quantity: Quantity,
        x_send_slice: List[slice],
        y_quantity: Quantity,
        y_send_slice: List[slice],
        n_clockwise_rotation: int,
        x_recv_slice: List[slice],
        y_recv_slice: List[slice],
    ) -> None:
        assert (
            self._type == MessageBundleType.VECTOR
            or self._type == MessageBundleType.UNKNOWN
        )
        if self._recv_buffer is not None and self._send_buffer is not None:
            raise RuntimeError(
                "Buffer previously allocated are not longer correct."
                "Make sure to queue all message before calling allocate."
            )
        self._type = MessageBundleType.VECTOR
        self._x_infos.append(
            MessageMetadata(
                quantity=x_quantity,
                send_slices=x_send_slice,
                linear_size=0,
                send_clockwise_rotation=n_clockwise_rotation,
                recv_slices=x_recv_slice,
            )
        )
        self._y_infos.append(
            MessageMetadata(
                quantity=y_quantity,
                send_slices=y_send_slice,
                linear_size=0,
                send_clockwise_rotation=-n_clockwise_rotation,
                recv_slices=y_recv_slice,
            )
        )

    def get_recv_buffer(self):
        if self._recv_buffer is None:
            raise RuntimeError("Recv buffer can't be retrieved before allocate()")
        return self._recv_buffer

    def get_send_buffer(self):
        if self._send_buffer is None:
            raise RuntimeError("Send buffer can't be retrieved before allocate()")
        return self._send_buffer

    def allocate(self):
        # Compute required size
        buffer_size = 0
        dtype = None
        for x_info in self._x_infos:
            x_info.linear_size = _slices_length(x_info.send_slices)
            buffer_size += x_info.linear_size
            dtype = x_info.quantity.metadata.dtype
        if self._type is MessageBundleType.VECTOR:
            for y_info in self._y_infos:
                y_info.linear_size = _slices_length(y_info.send_slices)
                buffer_size += y_info.linear_size

        # Demand a properly size buffers
        self._send_buffer = Buffer.pop_from_cache(
            self._np_module.empty, (buffer_size), dtype
        )
        self._recv_buffer = Buffer.pop_from_cache(
            self._np_module.empty, (buffer_size), dtype
        )

    def async_pack(self):
        """Pack all queued messages into a send Buffer.

        This guarantees the send Buffer is ready for MPI communication.
        The send Buffer is held by the class."""
        raise NotImplementedError()

    def async_unpack(self):
        """Unpack the buffer into destination arrays.

        This guarantees transfer is done and data ready to use.
        Once unpacking is done, recv & send Buffers are
        reinserted into the cache for reuse."""
        raise NotImplementedError()

    def synchronize(self):
        raise NotImplementedError()


class MessageBundleCPU(MessageBundle):
    def synchronize(self):
        # CPU doesn't do true async (for now)
        pass

    def async_pack(self):
        # Unpack per type
        if self._type == MessageBundleType.SCALAR:
            self._pack_scalar()
        elif self._type == MessageBundleType.VECTOR:
            self._pack_vector()
        else:
            raise RuntimeError(f"Unimplemented {self._type} message pack")

        self._send_buffer.finalize_memory_transfer()

    def _pack_scalar(self):
        offset = 0
        for x_info in self._x_infos:
            # sending data across the boundary will rotate the data
            # n_clockwise_rotations times, due to the difference in axis orientation.\
            # Thus we rotate that number of times counterclockwise before sending,
            # to get the right final orientation
            source_view = rotate_scalar_data(
                x_info.quantity.data[x_info.send_slices],
                x_info.quantity.dims,
                x_info.quantity.np,
                -x_info.send_clockwise_rotation,
            )
            contiguous_view = x_info.quantity.np.ascontiguousarray(source_view)
            self._send_buffer.assign_from(
                x_info.quantity.np.reshape(contiguous_view, x_info.linear_size),
                buffer_slice=np.index_exp[offset : offset + x_info.linear_size],
            )
            offset += x_info.linear_size

    def _pack_vector(self):
        assert len(self._x_infos) == len(self._y_infos)
        offset = 0
        for x_info, y_info in zip(self._x_infos, self._y_infos):
            # sending data across the boundary will rotate the data
            # n_clockwise_rotations times, due to the difference in axis orientation
            # Thus we rotate that number of times counterclockwise before sending,
            # to get the right final orientation
            x_view, y_view = rotate_vector_data(
                x_info.quantity.data[x_info.send_slices],
                y_info.quantity.data[y_info.send_slices],
                -x_info.send_clockwise_rotation,
                x_info.quantity.dims,
                x_info.quantity.np,
            )
            contiguous_view = x_info.quantity.np.ascontiguousarray(x_view)
            self._send_buffer.assign_from(
                x_info.quantity.np.reshape(contiguous_view, x_info.linear_size),
                buffer_slice=np.index_exp[offset : offset + x_info.linear_size],
            )
            offset += x_info.linear_size
            contiguous_view = y_info.quantity.np.ascontiguousarray(y_view)
            self._send_buffer.assign_from(
                y_info.quantity.np.reshape(contiguous_view, y_info.linear_size),
                buffer_slice=np.index_exp[offset : offset + y_info.linear_size],
            )
            offset += y_info.linear_size

    def async_unpack(self):
        # Unpack per type
        if self._type == MessageBundleType.SCALAR:
            self._unpack_scalar()
        elif self._type == MessageBundleType.VECTOR:
            self._unpack_vector()
        else:
            raise RuntimeError(f"Unimplemented {self._type} message unpack")

        self._recv_buffer.finalize_memory_transfer()

        # Pop the buffers back in the cache
        Buffer.push_to_cache(self._send_buffer)
        self._send_buffer = None
        Buffer.push_to_cache(self._recv_buffer)
        self._recv_buffer = None

    def _unpack_scalar(self):
        offset = 0
        for x_info in self._x_infos:
            quantity_view = x_info.quantity.data[x_info.recv_slices]
            self._recv_buffer.assign_to(
                quantity_view,
                buffer_slice=np.index_exp[offset : offset + x_info.linear_size],
                buffer_reshape=quantity_view.shape,
            )
            offset += x_info.linear_size

    def _unpack_vector(self):
        offset = 0
        for x_info, y_info in zip(self._x_infos, self._y_infos):
            quantity_view = x_info.quantity.data[x_info.recv_slices]
            self._recv_buffer.assign_to(
                quantity_view,
                buffer_slice=np.index_exp[offset : offset + x_info.linear_size],
                buffer_reshape=quantity_view.shape,
            )
            offset += x_info.linear_size
            quantity_view = y_info.quantity.data[y_info.recv_slices]
            self._recv_buffer.assign_to(
                quantity_view,
                buffer_slice=np.index_exp[offset : offset + y_info.linear_size],
                buffer_reshape=quantity_view.shape,
            )
            offset += y_info.linear_size


class MessageBundleGPU(MessageBundle):
    def pack(self):
        return NotImplementedError()

    def unpack(self):
        return NotImplementedError()
