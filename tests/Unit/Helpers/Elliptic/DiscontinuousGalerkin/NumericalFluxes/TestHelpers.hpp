// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/NumericalFluxHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "Utilities/TMPL.hpp"

namespace EllipticNumericalFluxesTestHelpers {

/// Test that the flux is single-valued on the interface, i.e. that the elements
/// on either side of the interface are working with the same numerical flux
/// data
template <size_t Dim, typename VariablesTags, typename FluxType>
void test_conservation(const FluxType& flux_computer,
                       const DataVector& used_for_size) {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.0, 1.0);
  const size_t num_points = used_for_size.size();

  using PackagedData =
      dg::SimpleBoundaryData<typename FluxType::package_field_tags,
                             typename FluxType::package_extra_tags>;
  PackagedData packaged_data_interior{};
  packaged_data_interior.field_data =
      make_with_random_values<Variables<typename FluxType::package_field_tags>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  PackagedData packaged_data_exterior{};
  packaged_data_exterior.field_data =
      make_with_random_values<Variables<typename FluxType::package_field_tags>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);

  Variables<VariablesTags> n_dot_num_flux_interior(
      num_points, std::numeric_limits<double>::signaling_NaN());
  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      make_not_null(&n_dot_num_flux_interior), flux_computer,
      packaged_data_interior, packaged_data_exterior);

  Variables<VariablesTags> n_dot_num_flux_exterior(
      num_points, std::numeric_limits<double>::signaling_NaN());
  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      make_not_null(&n_dot_num_flux_exterior), flux_computer,
      packaged_data_exterior, packaged_data_interior);

  CHECK_VARIABLES_APPROX(n_dot_num_flux_interior, -n_dot_num_flux_exterior);
}

}  // namespace EllipticNumericalFluxesTestHelpers
