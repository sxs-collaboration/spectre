// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions logical_coordinates and interface_grid_coordinates

#include "Domain/LogicalCoordinates.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Direction.hpp"
#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"

template <size_t VolumeDim>
tnsr::I<DataVector, VolumeDim, Frame::Logical> logical_coordinates(
    const Index<VolumeDim>& extents) {
  tnsr::I<DataVector, VolumeDim, Frame::Logical> logical_x(extents.product());
  for (size_t d = 0; d < VolumeDim; ++d) {
    const auto& collocation_points_in_this_dim =
        Basis::lgl::collocation_points(extents[d]);
    for (IndexIterator<VolumeDim> index(extents); index; ++index) {
      logical_x.get(d)[index.offset()] =
          collocation_points_in_this_dim[index()[d]];
    }
  }
  return logical_x;
}

template <size_t VolumeDim>
tnsr::I<DataVector, VolumeDim, Frame::Logical> interface_logical_coordinates(
    const Index<VolumeDim - 1>& extents,
    const Direction<VolumeDim>& direction) {
  const size_t sliced_away_dim = direction.dimension();
  tnsr::I<DataVector, VolumeDim, Frame::Logical> logical_x(extents.product());

  std::array<DataVector, VolumeDim - 1> collocation_points_in_each_dim{};
  for (size_t d = 0; d < VolumeDim - 1; ++d) {
    gsl::at(collocation_points_in_each_dim, d) =
        Basis::lgl::collocation_points(extents[d]);
  }

  for (IndexIterator<VolumeDim - 1> index(extents); index; ++index) {
    for (size_t d = 0; d < sliced_away_dim; ++d) {
      logical_x.get(d)[index.offset()] =
          gsl::at(collocation_points_in_each_dim, d)[index()[d]];
    }
    logical_x[sliced_away_dim] =
        (Side::Lower == direction.side() ? DataVector(extents.product(), -1.0)
                                         : DataVector(extents.product(), 1.0));
    for (size_t d = sliced_away_dim; d < VolumeDim - 1; ++d) {
      logical_x.get(d + 1)[index.offset()] =
          gsl::at(collocation_points_in_each_dim, d)[index()[d]];
    }
  }
  return logical_x;
}

// Explicit instantiations
template tnsr::I<DataVector, 1, Frame::Logical> logical_coordinates(
    const Index<1>& extents);
template tnsr::I<DataVector, 2, Frame::Logical> logical_coordinates(
    const Index<2>& extents);
template tnsr::I<DataVector, 3, Frame::Logical> logical_coordinates(
    const Index<3>& extents);

template tnsr::I<DataVector, 1, Frame::Logical> interface_logical_coordinates(
    const Index<0>& extents,
    const Direction<1>& direction);
template tnsr::I<DataVector, 2, Frame::Logical> interface_logical_coordinates(
    const Index<1>& extents,
    const Direction<2>& direction);
template tnsr::I<DataVector, 3, Frame::Logical> interface_logical_coordinates(
    const Index<2>& extents,
    const Direction<3>& direction);
