// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Sources/UniformAcceleration.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/VortexPerturbation.hpp"
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
template <typename SourceTerm, size_t Dim>
void wrap_time_derivative(
    const gsl::not_null<Scalar<DataVector>*>
        non_flux_terms_dt_mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        non_flux_terms_dt_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_energy_density,

    const gsl::not_null<tnsr::I<DataVector, Dim>*> mass_density_cons_flux,
    const gsl::not_null<tnsr::IJ<DataVector, Dim>*> momentum_density_flux,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> energy_density_flux,

    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& pressure,
    const tnsr::I<DataVector, Dim>& coords) {
  const double time = 1.38;
  const EquationsOfState::IdealFluid<false> eos{5.0 / 3.0};
  const Scalar<DataVector> specific_internal_energy =
      eos.specific_internal_energy_from_density_and_pressure(mass_density_cons,
                                                             pressure);
  const SourceTerm source = []() {
    if constexpr (std::is_same_v<
                      SourceTerm,
                      ::NewtonianEuler::Sources::UniformAcceleration<Dim>>) {
      std::array<double, Dim> accel{};
      for (size_t i = 0; i < Dim; ++i) {
        gsl::at(accel, i) = 0.3 + static_cast<double>(i);
      }
      return ::NewtonianEuler::Sources::UniformAcceleration<Dim>{accel};
    } else if constexpr (std::is_same_v<
                             SourceTerm,
                             ::NewtonianEuler::Sources::VortexPerturbation>) {
      return ::NewtonianEuler::Sources::VortexPerturbation{0.1};
    } else {
      return SourceTerm{};
    }
  }();
  Scalar<DataVector> enthalpy_density{get(energy_density).size()};
  if constexpr (std::is_same_v<
                    TestHelpers::NewtonianEuler::SomeOtherSourceType<Dim>,
                    SourceTerm>) {
    get(*non_flux_terms_dt_mass_density_cons) = -1.0;
  }
  NewtonianEuler::TimeDerivativeTerms<Dim>::apply(
      non_flux_terms_dt_mass_density_cons, non_flux_terms_dt_momentum_density,
      non_flux_terms_dt_energy_density, mass_density_cons_flux,
      momentum_density_flux, energy_density_flux,

      make_not_null(&enthalpy_density),

      mass_density_cons, momentum_density, energy_density,

      velocity, pressure, specific_internal_energy,

      eos, coords, time, source);
}

template <size_t Dim>
void test_time_derivative(const DataVector& used_for_size) {
  if constexpr (Dim == 3) {
    pypp::check_with_random_values<1>(
        &wrap_time_derivative<::NewtonianEuler::Sources::VortexPerturbation,
                              Dim>,
        "TimeDerivative",
        {"source_mass_density_cons_vortex_perturbation",
         "source_momentum_density_cons_vortex_perturbation",
         "source_energy_density_cons_vortex_perturbation",
         "mass_density_cons_flux", "momentum_density_flux",
         "energy_density_flux"},
        {{{-1.0, 1.0}}}, used_for_size);
  }
  pypp::check_with_random_values<1>(
      &wrap_time_derivative<::NewtonianEuler::Sources::UniformAcceleration<Dim>,
                            Dim>,
      "TimeDerivative",
      {"source_mass_density_cons_uniform_acceleration",
       "source_momentum_density_cons_uniform_acceleration",
       "source_energy_density_cons_uniform_acceleration",
       "mass_density_cons_flux", "momentum_density_flux",
       "energy_density_flux"},
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
