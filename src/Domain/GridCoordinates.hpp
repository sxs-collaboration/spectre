// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions logical_coordinates and interface_grid_coordinates

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Direction.hpp"
#include "Numerical/Spectral/LegendreGaussLobatto.hpp"

/*!
 * \ingroup ComputationalDomain
 * \brief Compute the Legendre-Gauss-Lobatto coordinates in an Element
 *
 * \returns logical-frame vector holding coordinates
 *
 * \example
 * \snippet Test_GridCoordinates.cpp logical_coordinates_example
 */
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

/*!
 * \ingroup ComputationalDomain
 * \brief Compute the grid coordinates on a face of an Element
 *
 * \returns grid-frame vector holding coordinates
 *
 * \remarks assumes that the grid-points lies at the Legendre-Gauss-Lobatto
 * points of the reference cell of the face
 *
 * \details
 * Computes the grid coordinates by applying the coordinate_map at the
 * Legendre-Gauss-Lobatto points of the reference cell of the face lying
 * in the given direction.
 *
 * \example
 * \snippet Test_GridCoordinates.cpp interface_grid_coordinates_example
 */
template <size_t VolumeDim, typename... Maps>
tnsr::I<DataVector, VolumeDim, Frame::Grid> interface_grid_coordinates(
    const Index<VolumeDim - 1>& extents,
    const CoordinateMap<Frame::Logical, Frame::Grid, Maps...>& coordinate_map,
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
  return coordinate_map(logical_x);
}
