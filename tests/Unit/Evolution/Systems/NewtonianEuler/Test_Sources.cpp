// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/Evolution/Systems/NewtonianEuler/TimeDerivativeTerms.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// We need a wrapper around the time derivative call because pypp cannot forward
// the source terms, only Tensor<DataVector>s and doubles.
template <typename InitialDataType, size_t Dim = InitialDataType::volume_dim>
void wrap_sources(
    const gsl::not_null<Scalar<DataVector>*> source_mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> source_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> source_energy_density,

    const Scalar<DataVector>& first_arg,
    const tnsr::I<DataVector, Dim>& second_arg,
    const Scalar<DataVector>& third_arg,
    const tnsr::i<DataVector, Dim>& fourth_arg) {
  typename InitialDataType::source_term_type source_computer;
  if constexpr (std::is_same_v<
                    TestHelpers::NewtonianEuler::TestInitialData<
                        TestHelpers::NewtonianEuler::SomeOtherSourceType<Dim>>,
                    InitialDataType>) {
    get(*source_mass_density_cons) = -1.0;
    NewtonianEuler::ComputeSources<InitialDataType>::apply(
        source_momentum_density, source_energy_density, source_computer,
        first_arg, second_arg, third_arg, fourth_arg);
  } else {
    NewtonianEuler::ComputeSources<InitialDataType>::apply(
        source_mass_density_cons, source_momentum_density,
        source_energy_density, source_computer, first_arg, second_arg,
        third_arg, fourth_arg);
  }
}

template <size_t Dim>
void test_sources(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(
      &wrap_sources<TestHelpers::NewtonianEuler::TestInitialData<
          TestHelpers::NewtonianEuler::SomeSourceType<Dim>>>,
      "TimeDerivative",
      {"source_mass_density_cons_impl", "source_momentum_density_impl",
       "source_energy_density_impl"},
      {{{-1.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &wrap_sources<TestHelpers::NewtonianEuler::TestInitialData<
          TestHelpers::NewtonianEuler::SomeOtherSourceType<Dim>>>,
      "TimeDerivative",
      {"minus_one_mass_density_impl", "source_momentum_density_impl",
       "source_energy_density_impl"},
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
