// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/SizeOfElement.hpp"

#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Mesh.hpp"  // IWYU pragma: keep
#include "Domain/Side.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdArrayHelpers.hpp"  // IWYU pragma: keep

template <size_t VolumeDim>
std::array<double, VolumeDim> size_of_element(
    const Mesh<VolumeDim>& mesh,
    const tnsr::I<DataVector, VolumeDim>& inertial_coords) noexcept {
  ASSERT(mesh.quadrature() ==
             make_array<VolumeDim>(Spectral::Quadrature::GaussLobatto),
         "Implementation assumes that grid points extend to the faces of the\n"
         "element, but not sure if this is true for quadrature: "
             << mesh.quadrature());
  auto result = make_array<VolumeDim>(0.0);
  // inertial-coord vector between lower face center and upper face center
  auto center_to_center = make_array<VolumeDim>(0.0);
  for (size_t logical_dim = 0; logical_dim < VolumeDim; ++logical_dim) {
    for (size_t inertial_dim = 0; inertial_dim < VolumeDim; ++inertial_dim) {
      const double center_upper = mean_value_on_boundary(
          inertial_coords.get(inertial_dim), mesh, logical_dim, Side::Upper);
      const double center_lower = mean_value_on_boundary(
          inertial_coords.get(inertial_dim), mesh, logical_dim, Side::Lower);
      center_to_center.at(inertial_dim) = center_upper - center_lower;
    }
    result.at(logical_dim) = magnitude(center_to_center);
  }
  return result;
}

// Explicit instantiations
template std::array<double, 1> size_of_element(
    const Mesh<1>&, const tnsr::I<DataVector, 1>&) noexcept;
template std::array<double, 2> size_of_element(
    const Mesh<2>&, const tnsr::I<DataVector, 2>&) noexcept;
template std::array<double, 3> size_of_element(
    const Mesh<3>&, const tnsr::I<DataVector, 3>&) noexcept;
