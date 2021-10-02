// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Evolution/Systems/NewtonianEuler/TimeDerivativeTerms.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/Evolution/Systems/NewtonianEuler/TimeDerivativeTerms.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// We need a wrapper around the time derivative call because pypp cannot forward
// the source terms, only Tensor<DataVector>s and doubles.
template <typename InitialDataType, size_t Dim = InitialDataType::volume_dim>
void wrap_time_derivative(
    const gsl::not_null<Scalar<DataVector>*>
        non_flux_terms_dt_mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        non_flux_terms_dt_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_energy_density,

    const gsl::not_null<tnsr::I<DataVector, Dim>*> mass_density_cons_flux,
    const gsl::not_null<tnsr::IJ<DataVector, Dim>*> momentum_density_flux,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> energy_density_flux,

    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& pressure,

    const Scalar<DataVector>& first_arg,
    const tnsr::I<DataVector, Dim>& second_arg,
    const Scalar<DataVector>& third_arg,
    const tnsr::i<DataVector, Dim>& fourth_arg) {
  typename InitialDataType::source_term_type source_computer;
  Scalar<DataVector> enthalpy_density{get(energy_density).size()};
  if constexpr (std::is_same_v<
                    TestHelpers::NewtonianEuler::TestInitialData<
                        TestHelpers::NewtonianEuler::SomeOtherSourceType<Dim>>,
                    InitialDataType>) {
    get(*non_flux_terms_dt_mass_density_cons) = -1.0;
  }
  NewtonianEuler::TimeDerivativeTerms<Dim, InitialDataType>::apply(
      non_flux_terms_dt_mass_density_cons, non_flux_terms_dt_momentum_density,
      non_flux_terms_dt_energy_density, mass_density_cons_flux,
      momentum_density_flux, energy_density_flux,
      make_not_null(&enthalpy_density), momentum_density, energy_density,
      velocity, pressure, source_computer, first_arg, second_arg, third_arg,
      fourth_arg);
}

template <size_t Dim>
void test_time_derivative(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(
      &wrap_time_derivative<TestHelpers::NewtonianEuler::TestInitialData<
          TestHelpers::NewtonianEuler::SomeSourceType<Dim>>>,
      "TimeDerivative",
      {"source_mass_density_cons", "source_momentum_density",
       "source_energy_density", "mass_density_cons_flux",
       "momentum_density_flux", "energy_density_flux"},
      {{{-1.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &wrap_time_derivative<TestHelpers::NewtonianEuler::TestInitialData<
          TestHelpers::NewtonianEuler::SomeOtherSourceType<Dim>>>,
      "TimeDerivative",
      {"minus_one_mass_density", "source_momentum_density",
       "source_energy_density", "mass_density_cons_flux",
       "momentum_density_flux", "energy_density_flux"},
      {{{-1.0, 1.0}}}, used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.TimeDerivative",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianEuler"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_time_derivative, (1, 2, 3))
}
