// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace TestHelpers {
namespace NumericalFluxes {

namespace Tags {
struct Variable1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Variable2 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};

template <size_t Dim>
struct Variable3 : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim>;
};

template <size_t Dim>
struct Variable4 : db::SimpleTag {
  using type = tnsr::Ij<DataVector, Dim>;
};

template <size_t Dim>
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, (Dim + 1) * (Dim + 1)>;
};

}  // namespace Tags

template <size_t Dim>
std::array<DataVector, (Dim + 1) * (Dim + 1)> characteristic_speeds(
    const Scalar<DataVector>& var_1, const tnsr::I<DataVector, Dim>& var_2,
    const tnsr::i<DataVector, Dim>& var_3) noexcept {
  std::array<DataVector, (Dim + 1) * (Dim + 1)> result;
  // Any expression for the characteristic speeds is fine.
  for (size_t i = 0; i < result.size(); ++i) {
    gsl::at(result, i) =
        cos(static_cast<double>(i)) * get(var_1) -
        (1.0 - sin(static_cast<double>(i))) * get(dot_product(var_2, var_3));
  }
  return result;
}

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using variables_tag =
      ::Tags::Variables<tmpl::list<Tags::Variable1, Tags::Variable2<Dim>,
                                   Tags::Variable3<Dim>, Tags::Variable4<Dim>>>;
  using char_speeds_tag = Tags::CharacteristicSpeeds<Dim>;
};

template <typename Var>
using n_dot_f = ::Tags::NormalDotFlux<Var>;

template <size_t Dim>
using n_dot_f_tags =
    tmpl::list<n_dot_f<Tags::Variable1>, n_dot_f<Tags::Variable2<Dim>>,
               n_dot_f<Tags::Variable3<Dim>>, n_dot_f<Tags::Variable4<Dim>>>;

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
template <size_t Dim, typename FluxType, typename... VariablesTags>
void test_conservation(const FluxType& flux_computer,
                       const DataVector& used_for_size,
                       const tmpl::list<VariablesTags...> /*meta*/) noexcept {
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.0, 1.0);

  const auto variables_interior =
      make_with_random_values<Variables<tmpl::list<VariablesTags...>>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  const auto variables_exterior =
      make_with_random_values<Variables<tmpl::list<VariablesTags...>>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  const auto n_dot_f_interior =
      make_with_random_values<Variables<tmpl::list<VariablesTags...>>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  const auto n_dot_f_exterior =
      make_with_random_values<Variables<tmpl::list<VariablesTags...>>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);

  Variables<typename FluxType::package_tags> packaged_data_interior(
      used_for_size.size(), std::numeric_limits<double>::signaling_NaN());
  flux_computer.package_data(
      make_not_null(&packaged_data_interior),
      get<VariablesTags>(n_dot_f_interior)...,
      get<VariablesTags>(variables_interior)...,
      TestHelpers::NumericalFluxes::characteristic_speeds(
          get<Tags::Variable1>(variables_interior),
          get<Tags::Variable2<Dim>>(variables_interior),
          get<Tags::Variable3<Dim>>(variables_interior)));

  Variables<typename FluxType::package_tags> packaged_data_exterior(
      used_for_size.size(), std::numeric_limits<double>::signaling_NaN());
  flux_computer.package_data(
      make_not_null(&packaged_data_exterior),
      get<VariablesTags>(n_dot_f_exterior)...,
      get<VariablesTags>(variables_exterior)...,
      TestHelpers::NumericalFluxes::characteristic_speeds(
          get<Tags::Variable1>(variables_exterior),
          get<Tags::Variable2<Dim>>(variables_exterior),
          get<Tags::Variable3<Dim>>(variables_exterior)));

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

  auto check = [](const auto& int_flux, const auto& ext_flux) noexcept {
    for (size_t i = 0; i < int_flux.size(); ++i) {
      CHECK_ITERABLE_APPROX(int_flux[i], -ext_flux[i]);
    }
    return nullptr;
  };

  expand_pack(check(get<VariablesTags>(n_dot_num_flux_interior),
                    get<VariablesTags>(n_dot_num_flux_exterior))...);
}
}  // namespace detail

template <size_t Dim, typename FluxType>
void test_conservation(const FluxType& flux_computer,
                       const DataVector& used_for_size) noexcept {
  detail::test_conservation<Dim>(
      flux_computer, used_for_size,
      typename System<Dim>::variables_tag::tags_list{});
}

}  // namespace NumericalFluxes
}  // namespace TestHelpers
