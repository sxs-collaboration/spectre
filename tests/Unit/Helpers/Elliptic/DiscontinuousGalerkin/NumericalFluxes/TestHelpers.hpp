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
#include "Utilities/TMPL.hpp"

namespace EllipticNumericalFluxesTestHelpers {

template <class... PackageDataTags, class FluxType,
          class... NormalDotNumericalFluxTypes>
void apply_numerical_flux(
    const FluxType& flux,
    const Variables<tmpl::list<PackageDataTags...>>& packaged_data_int,
    const Variables<tmpl::list<PackageDataTags...>>& packaged_data_ext,
    NormalDotNumericalFluxTypes&&... normal_dot_numerical_flux) noexcept {
  flux(std::forward<NormalDotNumericalFluxTypes>(normal_dot_numerical_flux)...,
       get<PackageDataTags>(packaged_data_int)...,
       get<PackageDataTags>(packaged_data_ext)...);
}

namespace detail {

/// Test that the flux is single-valued on the interface, i.e. that the elements
/// on either side of the interface are working with the same numerical flux
/// data
template <size_t Dim, typename FluxType, typename... VariablesTags>
void test_conservation_impl(const FluxType& flux_computer,
                            const DataVector& used_for_size,
                            const tmpl::list<VariablesTags...> /*meta*/) {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.0, 1.0);

  using PackagedData = Variables<typename FluxType::package_tags>;
  const auto packaged_data_interior = make_with_random_values<PackagedData>(
      make_not_null(&gen), make_not_null(&dist), used_for_size);
  const auto packaged_data_exterior = make_with_random_values<PackagedData>(
      make_not_null(&gen), make_not_null(&dist), used_for_size);

  Variables<tmpl::list<VariablesTags...>> n_dot_num_flux_interior(
      used_for_size.size(), std::numeric_limits<double>::signaling_NaN());
  apply_numerical_flux(flux_computer, packaged_data_interior,
                       packaged_data_exterior,
                       &get<VariablesTags>(n_dot_num_flux_interior)...);

  Variables<tmpl::list<VariablesTags...>> n_dot_num_flux_exterior(
      used_for_size.size(), std::numeric_limits<double>::signaling_NaN());
  apply_numerical_flux(flux_computer, packaged_data_exterior,
                       packaged_data_interior,
                       &get<VariablesTags>(n_dot_num_flux_exterior)...);

  const auto check = [](const auto& int_flux, const auto& ext_flux) noexcept {
    for (size_t i = 0; i < int_flux.size(); ++i) {
      CHECK_ITERABLE_APPROX(int_flux[i], -ext_flux[i]);
    }
    return nullptr;
  };

  EXPAND_PACK_LEFT_TO_RIGHT(check(get<VariablesTags>(n_dot_num_flux_interior),
                                  get<VariablesTags>(n_dot_num_flux_exterior)));
}

}  // namespace detail

template <size_t Dim, typename VariablesTags, typename FluxType>
void test_conservation(const FluxType& flux_computer,
                       const DataVector& used_for_size) {
  detail::test_conservation_impl<Dim>(flux_computer, used_for_size,
                                      VariablesTags{});
}

}  // namespace EllipticNumericalFluxesTestHelpers
