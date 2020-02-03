// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {

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
      tmpl::list<NewtonianEuler::Tags::MassDensityCons<DataVector>,
                 NewtonianEuler::Tags::MomentumDensity<DataVector, Dim>,
                 NewtonianEuler::Tags::EnergyDensity<DataVector>>;

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
      tmpl::list<NewtonianEuler::Tags::MomentumDensity<DataVector, Dim>,
                 NewtonianEuler::Tags::EnergyDensity<DataVector>>;

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
struct SomeInitialData {
  static constexpr size_t volume_dim = SourceTermType::volume_dim;
  using source_term_type = SourceTermType;
};

// We need this proxy, as pypp can't work with arguments other than doubles
// and SpECTRE tensors.
template <typename InitialDataType>
struct ComputeSourcesProxy {
  static constexpr size_t dim = InitialDataType::volume_dim;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_mass_density_cons,
      const gsl::not_null<tnsr::I<DataVector, dim>*> source_momentum_density,
      const gsl::not_null<Scalar<DataVector>*> source_energy_density,
      const Scalar<DataVector>& first_arg,
      const tnsr::I<DataVector, dim>& second_arg,
      const Scalar<DataVector>& third_arg,
      const tnsr::i<DataVector, dim>& fourth_arg) noexcept {
    typename InitialDataType::source_term_type source_computer;
    NewtonianEuler::ComputeSources<InitialDataType>::apply(
        source_mass_density_cons, source_momentum_density,
        source_energy_density, source_computer, first_arg, second_arg,
        third_arg, fourth_arg);
  }
};

// Pypp needs explicit args for it to work, so we need a new proxy for testing
// second source term with different number of sourced variables.
template <typename InitialDataType>
struct ComputeSourcesProxy2 {
  static constexpr size_t dim = InitialDataType::volume_dim;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, dim>*> source_momentum_density,
      const gsl::not_null<Scalar<DataVector>*> source_energy_density,
      const Scalar<DataVector>& first_arg,
      const tnsr::I<DataVector, dim>& second_arg,
      const Scalar<DataVector>& third_arg,
      const tnsr::i<DataVector, dim>& fourth_arg) noexcept {
    typename InitialDataType::source_term_type source_computer;
    NewtonianEuler::ComputeSources<InitialDataType>::apply(
        source_momentum_density, source_energy_density, source_computer,
        first_arg, second_arg, third_arg, fourth_arg);
  }
};

template <size_t Dim>
void test_sources(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(
      &ComputeSourcesProxy<SomeInitialData<SomeSourceType<Dim>>>::apply,
      "TestFunctions",
      {"source_mass_density_cons", "source_momentum_density",
       "source_energy_density"},
      {{{-1.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &ComputeSourcesProxy2<SomeInitialData<SomeOtherSourceType<Dim>>>::apply,
      "TestFunctions", {"source_momentum_density", "source_energy_density"},
      {{{-1.0, 1.0}}}, used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Sources",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianEuler"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_sources, (1, 2, 3))
}
