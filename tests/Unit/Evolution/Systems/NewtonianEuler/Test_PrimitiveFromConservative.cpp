// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/PrimitiveFromConservative.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

template <size_t Dim>
struct PrimitiveFromConservativeProxyThermoDim1 {
  static void apply_helper(
      gsl::not_null<Scalar<DataVector>*> mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim>*> velocity,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<Scalar<DataVector>*> pressure,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density) noexcept {
    NewtonianEuler::PrimitiveFromConservative<Dim, 1>::apply(
        mass_density, velocity, specific_internal_energy, pressure,
        mass_density_cons, momentum_density, energy_density,
        EquationsOfState::PolytropicFluid<false>(1.4, 5.0 / 3.0));
  }
};

template <size_t Dim>
struct PrimitiveFromConservativeProxyThermoDim2 {
  static void apply_helper(
      gsl::not_null<Scalar<DataVector>*> mass_density,
      gsl::not_null<tnsr::I<DataVector, Dim>*> velocity,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<Scalar<DataVector>*> pressure,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density) noexcept {
    NewtonianEuler::PrimitiveFromConservative<Dim, 2>::apply(
        mass_density, velocity, specific_internal_energy, pressure,
        mass_density_cons, momentum_density, energy_density,
        EquationsOfState::IdealFluid<false>(5.0 / 3.0));
  }
};

template <size_t Dim>
void test_primitive_from_conservative(const DataVector& used_for_size) {
  pypp::check_with_random_values<3>(
      &PrimitiveFromConservativeProxyThermoDim1<Dim>::apply_helper,
      "TestFunctions",
      {"mass_density", "velocity", "specific_internal_energy", "pressure_1d"},
      {{{0.0, 1.0}, {-2.0, 2.0}, {0.0, 3.0}}}, used_for_size);

  pypp::check_with_random_values<3>(
      &PrimitiveFromConservativeProxyThermoDim2<Dim>::apply_helper,
      "TestFunctions",
      {"mass_density", "velocity", "specific_internal_energy", "pressure_2d"},
      {{{0.0, 1.0}, {-2.0, 2.0}, {0.0, 3.0}}}, used_for_size);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.PrimitiveFromConservative",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianEuler"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_primitive_from_conservative, (1, 2, 3))
}
