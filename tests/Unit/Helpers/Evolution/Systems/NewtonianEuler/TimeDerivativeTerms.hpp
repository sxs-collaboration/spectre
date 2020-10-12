// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::NewtonianEuler {
struct FirstArg : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct SecondArg : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};

struct ThirdArg : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct FourthArg : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim>;
};

// Some source term where all three conservative quantities are sourced.
template <size_t Dim>
struct SomeSourceType {
  static constexpr size_t volume_dim = Dim;
  using sourced_variables =
      tmpl::list<::NewtonianEuler::Tags::MassDensityCons,
                 ::NewtonianEuler::Tags::MomentumDensity<Dim>,
                 ::NewtonianEuler::Tags::EnergyDensity>;

  using argument_tags =
      tmpl::list<FirstArg, SecondArg<Dim>, ThirdArg, FourthArg<Dim>>;

  void apply(
      const gsl::not_null<Scalar<DataVector>*> source_mass_density_cons,
      const gsl::not_null<tnsr::I<DataVector, Dim>*> source_momentum_density,
      const gsl::not_null<Scalar<DataVector>*> source_energy_density,
      const Scalar<DataVector>& first_arg,
      const tnsr::I<DataVector, Dim>& second_arg,
      const Scalar<DataVector>& third_arg,
      const tnsr::i<DataVector, Dim>& fourth_arg) const noexcept {
    get(*source_mass_density_cons) = exp(get(first_arg));
    for (size_t i = 0; i < Dim; ++i) {
      source_momentum_density->get(i) =
          (get(first_arg) - 1.5 * get(third_arg)) * second_arg.get(i);
    }
    get(*source_energy_density) =
        get(dot_product(second_arg, fourth_arg)) + 3.0 * get(third_arg);
  }
};

// Some other source term where the mass density is not sourced (this is by
// far the most common type of non-trivial source term for NewtonianEuler.)
template <size_t Dim>
struct SomeOtherSourceType {
  static constexpr size_t volume_dim = Dim;
  using sourced_variables =
      tmpl::list<::NewtonianEuler::Tags::MomentumDensity<Dim>,
                 ::NewtonianEuler::Tags::EnergyDensity>;

  using argument_tags =
      tmpl::list<FirstArg, SecondArg<Dim>, ThirdArg, FourthArg<Dim>>;

  void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim>*> source_momentum_density,
      const gsl::not_null<Scalar<DataVector>*> source_energy_density,
      const Scalar<DataVector>& first_arg,
      const tnsr::I<DataVector, Dim>& second_arg,
      const Scalar<DataVector>& third_arg,
      const tnsr::i<DataVector, Dim>& fourth_arg) const noexcept {
    for (size_t i = 0; i < Dim; ++i) {
      source_momentum_density->get(i) =
          (get(first_arg) - 1.5 * get(third_arg)) * second_arg.get(i);
    }
    get(*source_energy_density) =
        get(dot_product(second_arg, fourth_arg)) + 3.0 * get(third_arg);
  }
};

template <typename SourceTermType>
struct TestInitialData {
  static constexpr size_t volume_dim = SourceTermType::volume_dim;
  using source_term_type = SourceTermType;
};
}  // namespace TestHelpers::NewtonianEuler
