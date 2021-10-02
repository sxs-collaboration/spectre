// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/NumericalFluxHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "Utilities/TMPL.hpp"

namespace EllipticNumericalFluxesTestHelpers {

/// Test that the flux is single-valued on the interface, i.e. that the elements
/// on either side of the interface are working with the same numerical flux
/// data
template <size_t Dim, typename NumericalFluxComputer, typename MakeRandomArgs>
void test_conservation(const NumericalFluxComputer& numerical_flux_computer,
                       MakeRandomArgs&& make_random_args,
                       const DataVector& used_for_size) {
  const size_t num_points = used_for_size.size();
  using PackagedData = dg::SimpleBoundaryData<
      typename NumericalFluxComputer::package_field_tags,
      typename NumericalFluxComputer::package_extra_tags>;
  const auto make_random_packaged_data = [&numerical_flux_computer,
                                          &make_random_args, &num_points]() {
    return std::apply(
        [&numerical_flux_computer, &num_points](const auto... args) {
          PackagedData packaged_data{num_points};
          dg::NumericalFluxes::package_data(make_not_null(&packaged_data),
                                            numerical_flux_computer, args...);
          return packaged_data;
        },
        make_random_args());
  };
  const auto packaged_data_interior = make_random_packaged_data();
  const auto packaged_data_exterior = make_random_packaged_data();

  using Vars = Variables<typename NumericalFluxComputer::variables_tags>;
  Vars n_dot_num_flux_interior(num_points,
                               std::numeric_limits<double>::signaling_NaN());
  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      make_not_null(&n_dot_num_flux_interior), numerical_flux_computer,
      packaged_data_interior, packaged_data_exterior);
  Vars n_dot_num_flux_exterior(num_points,
                               std::numeric_limits<double>::signaling_NaN());
  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      make_not_null(&n_dot_num_flux_exterior), numerical_flux_computer,
      packaged_data_exterior, packaged_data_interior);

  CHECK_VARIABLES_APPROX(n_dot_num_flux_interior, -n_dot_num_flux_exterior);
}

}  // namespace EllipticNumericalFluxesTestHelpers
