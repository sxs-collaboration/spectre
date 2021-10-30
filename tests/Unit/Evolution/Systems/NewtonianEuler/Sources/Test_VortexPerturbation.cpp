// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/VortexPerturbation.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/IsentropicVortex.hpp"

// IWYU pragma: no_include <string>

// IWYU pragma: no_include "DataStructures/Tensor/Tensor.hpp"
// IWYU pragma: no_include "Utilities/Gsl.hpp"

namespace {
// Need this proxy in order for pypp to evaluate function whose arguments
// include a variable of type `NewtonianEuler::Solutions::IsentropicVortex`
struct VortexPerturbation3dProxy
    : NewtonianEuler::Sources::VortexPerturbation {
  using NewtonianEuler::Sources::VortexPerturbation::VortexPerturbation;
  VortexPerturbation3dProxy(const double adiabatic_index,
                            const double perturbation_amplitude,
                            const std::array<double, 3>& vortex_center,
                            const std::array<double, 3>& vortex_mean_velocity,
                            const double vortex_strength)
      : vortex_(adiabatic_index, vortex_center, vortex_mean_velocity,
                vortex_strength, perturbation_amplitude) {}
  void apply_helper(
      const gsl::not_null<Scalar<DataVector>*> source_mass_density_cons,
      const gsl::not_null<tnsr::I<DataVector, 3>*> source_momentum_density,
      const gsl::not_null<Scalar<DataVector>*> source_energy_density,
      const tnsr::I<DataVector, 3>& x, const double t) const {
    this->apply(source_mass_density_cons, source_momentum_density,
                source_energy_density, vortex_, x, t);
  }

 private:
  NewtonianEuler::Solutions::IsentropicVortex<3> vortex_{};
};

void test_sources() {
  const double perturbation_amplitude = 0.1987;
  const double adiabatic_index = 1.2;
  const std::array<double, 3> vortex_center = {{0.7, 0.5, -5.5}};
  const std::array<double, 3> vortex_mean_velocity = {{1.8, 0.6, -0.57}};
  const double vortex_strength = 2.0;
  VortexPerturbation3dProxy source_proxy(adiabatic_index,
                                         perturbation_amplitude, vortex_center,
                                         vortex_mean_velocity, vortex_strength);
  pypp::check_with_random_values<1>(
      &VortexPerturbation3dProxy::apply_helper, source_proxy,
      "Evolution.Systems.NewtonianEuler.Sources.VortexPerturbation",
      {"source_mass_density_cons", "source_momentum_density",
       "source_energy_density"},
      {{{0.0, 1.0}}},
      std::make_tuple(adiabatic_index, perturbation_amplitude, vortex_center,
                      vortex_mean_velocity, vortex_strength),
      DataVector(5));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Sources.VortexPerturb",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  test_sources();
}
