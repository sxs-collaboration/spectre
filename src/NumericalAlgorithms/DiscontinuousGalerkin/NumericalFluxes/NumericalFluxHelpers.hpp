// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace dg {
namespace NumericalFluxes {

namespace detail {
template <typename NumericalFluxType, typename... AllFieldTags,
          typename... AllExtraTags, typename... Args,
          typename... PackageFieldTags, typename... PackageExtraTags>
void package_data_impl(
    const gsl::not_null<dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                               tmpl::list<AllExtraTags...>>*>
        packaged_data,
    const NumericalFluxType& numerical_flux_computer,
    tmpl::list<PackageFieldTags...> /*meta*/,
    tmpl::list<PackageExtraTags...> /*meta*/, const Args&... args) noexcept {
  numerical_flux_computer.package_data(
      make_not_null(&get<PackageFieldTags>(packaged_data->field_data))...,
      make_not_null(&get<PackageExtraTags>(packaged_data->extra_data))...,
      args...);
}

template <typename NumericalFluxType, typename... AllFieldTags,
          typename... AllExtraTags, typename... NormalDotNumericalFluxTypes,
          typename... PackageFieldTags, typename... PackageExtraTags>
void normal_dot_numerical_fluxes_impl(
    const NumericalFluxType& numerical_flux_computer,
    const dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                 tmpl::list<AllExtraTags...>>&
        packaged_data_int,
    const dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                 tmpl::list<AllExtraTags...>>&
        packaged_data_ext,
    tmpl::list<PackageFieldTags...> /*meta*/,
    tmpl::list<PackageExtraTags...> /*meta*/,
    // Taking output arguments last in this detail implementation so the
    // `NormalDotNumericalFluxTypes` can be inferred
    const gsl::not_null<
        NormalDotNumericalFluxTypes*>... n_dot_num_fluxes) noexcept {
  numerical_flux_computer(
      n_dot_num_fluxes...,
      get<PackageFieldTags>(packaged_data_int.field_data)...,
      get<PackageExtraTags>(packaged_data_int.extra_data)...,
      get<PackageFieldTags>(packaged_data_ext.field_data)...,
      get<PackageExtraTags>(packaged_data_ext.extra_data)...);
}
}  // namespace detail

// @{
/// Helper function to unpack arguments when invoking the numerical flux
template <typename NumericalFluxType, typename... AllFieldTags,
          typename... AllExtraTags, typename... Args>
void package_data(
    const gsl::not_null<dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                               tmpl::list<AllExtraTags...>>*>
        packaged_data,
    const NumericalFluxType& numerical_flux_computer,
    const Args&... args) noexcept {
  static_assert(
      tt::conforms_to_v<NumericalFluxType, protocols::NumericalFlux>,
      "The 'NumericalFluxType' must conform to 'dg::protocol::NumericalFlux'.");
  detail::package_data_impl(packaged_data, numerical_flux_computer,
                            typename NumericalFluxType::package_field_tags{},
                            typename NumericalFluxType::package_extra_tags{},
                            args...);
}

template <typename NumericalFluxType, typename... AllFieldTags,
          typename... AllExtraTags, typename... NormalDotNumericalFluxTags>
void normal_dot_numerical_fluxes(
    const gsl::not_null<Variables<tmpl::list<NormalDotNumericalFluxTags...>>*>
        n_dot_num_fluxes,
    const NumericalFluxType& numerical_flux_computer,
    const dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                 tmpl::list<AllExtraTags...>>&
        packaged_data_int,
    const dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                 tmpl::list<AllExtraTags...>>&
        packaged_data_ext) noexcept {
  static_assert(
      tt::conforms_to_v<NumericalFluxType, protocols::NumericalFlux>,
      "The 'NumericalFluxType' must conform to 'dg::protocol::NumericalFlux'.");
  detail::normal_dot_numerical_fluxes_impl(
      numerical_flux_computer, packaged_data_int, packaged_data_ext,
      typename NumericalFluxType::package_field_tags{},
      typename NumericalFluxType::package_extra_tags{},
      make_not_null(&get<NormalDotNumericalFluxTags>(*n_dot_num_fluxes))...);
}

template <typename NumericalFluxType, typename... AllFieldTags,
          typename... AllExtraTags, typename... NormalDotNumericalFluxes>
void normal_dot_numerical_fluxes(
    const NumericalFluxType& numerical_flux_computer,
    const dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                 tmpl::list<AllExtraTags...>>&
        packaged_data_int,
    const dg::SimpleBoundaryData<tmpl::list<AllFieldTags...>,
                                 tmpl::list<AllExtraTags...>>&
        packaged_data_ext,
    // Need to take these return-by-reference arguments last so the template
    // parameter pack deduction works
    gsl::not_null<NormalDotNumericalFluxes*>... n_dot_num_fluxes) noexcept {
  static_assert(
      tt::conforms_to_v<NumericalFluxType, protocols::NumericalFlux>,
      "The 'NumericalFluxType' must conform to 'dg::protocol::NumericalFlux'.");
  detail::normal_dot_numerical_fluxes_impl(
      numerical_flux_computer, packaged_data_int, packaged_data_ext,
      typename NumericalFluxType::package_field_tags{},
      typename NumericalFluxType::package_extra_tags{}, n_dot_num_fluxes...);
}
// @}

}  // namespace NumericalFluxes
}  // namespace dg
