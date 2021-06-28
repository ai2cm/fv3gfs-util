import dataclasses
from .quantity import Quantity
from ._boundary_utils import get_boundary_slice
from typing import Tuple


@dataclasses.dataclass
class Boundary:
    """Maps part of a subtile domain to another rank which shares halo points."""

    from_rank: int
    to_rank: int
    n_clockwise_rotations: int
    """
    number of clockwise rotations data undergoes if it moves from the from_rank
    to the to_rank. The same as the number of clockwise rotations to get from the
    orientation of the axes in from_rank to the orientation of the axes in to_rank.
    """

    def send_view(self, quantity: Quantity, n_points: int):
        """Return a sliced view of points which should be sent at this boundary.

        Args:
            quantity: quantity for which to return a slice
            n_points: the width of boundary to include
        """
        return self._view(quantity, n_points, interior=True)

    def recv_view(self, quantity: Quantity, n_points: int):
        """Return a sliced view of points which should be recieved at this boundary.

        Args:
            quantity: quantity for which to return a slice
            n_points: the width of boundary to include
        """
        return self._view(quantity, n_points, interior=False)

    def send_slice(self, quantity: Quantity, n_points: int) -> Tuple(slice):
        """Return the index slices which shoud be sent at this boundary.

        Args:
            quantity: quantity for which to return slices
            n_points: the width of boundary to include
        
        Returns:
            A tuple of slices (one per dimensions)
        """
        return self._slice(quantity, n_points, interior=True)

    def recv_slice(self, quantity: Quantity, n_points: int) -> Tuple(slice):
        """Return the index slices which shoud be received at this boundary.

        Args:
            quantity: quantity for which to return slices
            n_points: the width of boundary to include
        
        Returns:
            A tuple of slices (one per dimensions)
        """
        return self._slice(quantity, n_points, interior=False)

    def _slice(self, quantity: Quantity, n_points: int, interior: bool) -> Tuple(slice):
        """Abstract function to be reimplemented by child class.
        
        Return:
            A tuple of slices (one per dimensions)
        """
        raise NotImplementedError()

    def _view(self, quantity: Quantity, n_points: int, interior: bool):
        """Return a sliced view of points in the given quantity at this boundary.

        Args:
            quantity: quantity for which to return a slice
            n_points: the width of boundary to include
            interior: if True, give points inside the computational domain (default),
                otherwise give points in the halo
        """
        raise NotImplementedError()


@dataclasses.dataclass
class SimpleBoundary(Boundary):
    """A boundary representing an edge or corner of a subtile."""

    boundary_type: int

    def _view(self, quantity: Quantity, n_points: int, interior: bool):
        boundary_slice = self._slice(quantity, n_points, interior)
        return quantity.data[tuple(boundary_slice)]

    def _slice(self, quantity: Quantity, n_points: int, interior: bool):
        return get_boundary_slice(
            quantity.dims,
            quantity.origin,
            quantity.extent,
            quantity.data.shape,
            self.boundary_type,
            n_points,
            interior,
        )
