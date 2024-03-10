// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/VortexPerturbation.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"

namespace {
// Need this proxy in order for pypp to evaluate function whose arguments
// include a variable of type `NewtonianEuler::Solutions::IsentropicVortex`
struct VortexPerturbation3dProxy
    : NewtonianEuler::Sources::VortexPerturbation {
  using NewtonianEuler::Sources::VortexPerturbation::VortexPerturbation;
  explicit VortexPerturbation3dProxy(const double perturbation_amplitude)
      : NewtonianEuler::Sources::VortexPerturbation(perturbation_amplitude){};

  void apply(
      const gsl::not_null<Scalar<DataVector>*> source_mass_density_cons,
      const gsl::not_null<tnsr::I<DataVector, 3>*> source_momentum_density,
      const gsl::not_null<Scalar<DataVector>*> source_energy_density,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, 3>& momentum_density,
      const Scalar<DataVector>& energy_density_cons,
      const Scalar<DataVector>& pressure,
      const tnsr::I<DataVector, 3>& coords) const {
    tnsr::I<DataVector, 3> velocity = momentum_density;
    for (size_t i = 0; i < 3; ++i) {
      velocity.get(i) = momentum_density.get(i) / get(mass_density_cons);
    }

    get(*source_mass_density_cons) = 0.0;
    get(*source_energy_density) = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      source_momentum_density->get(i) = 0.0;
    }
    // EOS is unused
    const EquationsOfState::IdealFluid<false> eos{5.0 / 3.0};
    static_cast<const NewtonianEuler::Sources::VortexPerturbation&>(*this)(
        source_mass_density_cons, source_momentum_density,
        source_energy_density, mass_density_cons, momentum_density,
        energy_density_cons, velocity, pressure, {}, eos, coords, 0.0);
  }
};

void test_sources() {
  const double perturbation_amplitude = 0.1987;
  VortexPerturbation3dProxy source_proxy(perturbation_amplitude);
  pypp::check_with_random_values<1>(
      &VortexPerturbation3dProxy::apply, source_proxy,
      "Evolution.Systems.NewtonianEuler.Sources.VortexPerturbation",
      {"source_mass_density_cons", "source_momentum_density",
       "source_energy_density"},
      {{{0.0, 1.0}}}, std::make_tuple(perturbation_amplitude), DataVector(5));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Sources.VortexPerturb",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  test_sources();
}
