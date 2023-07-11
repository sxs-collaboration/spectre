// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Direction.hpp"    // IWYU pragma: keep
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

template <size_t VolumeDim>
void logical_coordinates(
    const gsl::not_null<tnsr::I<DataVector, VolumeDim, Frame::ElementLogical>*>
        logical_coords,
    const Mesh<VolumeDim>& mesh) {
  set_number_of_grid_points(logical_coords, mesh.number_of_grid_points());
  for (size_t d = 0; d < VolumeDim; ++d) {
    const auto& collocation_points_in_this_dim =
        Spectral::collocation_points(mesh.slice_through(d));
    for (IndexIterator<VolumeDim> index(mesh.extents()); index; ++index) {
      logical_coords->get(d)[index.collapsed_index()] =
          collocation_points_in_this_dim[index()[d]];
    }
  }
}

template <size_t VolumeDim>
tnsr::I<DataVector, VolumeDim, Frame::ElementLogical> logical_coordinates(
    const Mesh<VolumeDim>& mesh) {
  tnsr::I<DataVector, VolumeDim, Frame::ElementLogical> result{};
  logical_coordinates(make_not_null(&result), mesh);
  return result;
}

// Explicit instantiations
template tnsr::I<DataVector, 1, Frame::ElementLogical> logical_coordinates(
    const Mesh<1>&);
template tnsr::I<DataVector, 2, Frame::ElementLogical> logical_coordinates(
    const Mesh<2>&);
template tnsr::I<DataVector, 3, Frame::ElementLogical> logical_coordinates(
    const Mesh<3>&);
template void logical_coordinates(
    const gsl::not_null<tnsr::I<DataVector, 1, Frame::ElementLogical>*>,
    const Mesh<1>&);
template void logical_coordinates(
    const gsl::not_null<tnsr::I<DataVector, 2, Frame::ElementLogical>*>,
    const Mesh<2>&);
template void logical_coordinates(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::ElementLogical>*>,
    const Mesh<3>&);
