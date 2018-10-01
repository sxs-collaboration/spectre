// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace TestHelpers {
namespace NumericalFluxes {

namespace Tags {
struct Variable1 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Variable1"; }
};

template <size_t Dim>
struct Variable2 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
  static std::string name() noexcept { return "Variable2"; }
};

template <size_t Dim>
struct Variable3 : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim>;
  static std::string name() noexcept { return "Variable3"; }
};

template <size_t Dim>
struct Variable4 : db::SimpleTag {
  using type = tnsr::Ij<DataVector, Dim>;
  static std::string name() noexcept { return "Variable4"; }
};

template <size_t Dim>
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, (Dim + 1) * (Dim + 1)>;
  static std::string name() noexcept { return "CharacteristicSpeeds"; }
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

}  // namespace NumericalFluxes
}  // namespace TestHelpers
