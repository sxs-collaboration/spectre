// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/InterfaceLogicalCoordinates.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

template <size_t VolumeDim>
tnsr::I<DataVector, VolumeDim, Frame::ElementLogical>
interface_logical_coordinates(const Mesh<VolumeDim - 1>& mesh,
                              const Direction<VolumeDim>& direction) {
  const auto num_grid_points = mesh.number_of_grid_points();
  const size_t sliced_away_dim = direction.dimension();
  tnsr::I<DataVector, VolumeDim, Frame::ElementLogical> logical_x(
      num_grid_points);

  std::array<DataVector, VolumeDim - 1> collocation_points_in_each_dim{};
  for (size_t d = 0; d < VolumeDim - 1; ++d) {
    collocation_points_in_each_dim.at(d) =
        Spectral::collocation_points(mesh.slice_through(d));
  }

  for (IndexIterator<VolumeDim - 1> index(mesh.extents()); index; ++index) {
    for (size_t d = 0; d < sliced_away_dim; ++d) {
      logical_x.get(d)[index.collapsed_index()] =
          collocation_points_in_each_dim.at(d)[index()[d]];
    }
    logical_x[sliced_away_dim] =
        (Side::Lower == direction.side() ? DataVector(num_grid_points, -1.0)
                                         : DataVector(num_grid_points, 1.0));
    for (size_t d = sliced_away_dim; d < VolumeDim - 1; ++d) {
      logical_x.get(d + 1)[index.collapsed_index()] =
          collocation_points_in_each_dim.at(d)[index()[d]];
    }
  }
  return logical_x;
}

// We need this specialisation since Mesh<Dim>::slice_through() is available
// only for Dim > 0
template <>
tnsr::I<DataVector, 1, Frame::ElementLogical> interface_logical_coordinates<1>(
    const Mesh<0>& mesh, const Direction<1>& direction) {
  return tnsr::I<DataVector, 1, Frame::ElementLogical>{
      mesh.number_of_grid_points(), direction.sign()};
}

// Explicit instantiations
template tnsr::I<DataVector, 2, Frame::ElementLogical>
interface_logical_coordinates(const Mesh<1>&, const Direction<2>&);
template tnsr::I<DataVector, 3, Frame::ElementLogical>
interface_logical_coordinates(const Mesh<2>&, const Direction<3>&);
