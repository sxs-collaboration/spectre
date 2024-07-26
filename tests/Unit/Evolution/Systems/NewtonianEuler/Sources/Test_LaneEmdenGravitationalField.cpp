// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/LaneEmdenGravitationalField.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/Gsl.hpp"

namespace {
// Need this proxy in order for pypp to evaluate function whose arguments
// include a variable of type `NewtonianEuler::Solutions::LaneEmdenStar`
struct LaneEmdenGravitationalFieldProxy
    : NewtonianEuler::Sources::LaneEmdenGravitationalField {
  using NewtonianEuler::Sources::LaneEmdenGravitationalField::
      LaneEmdenGravitationalField;
  LaneEmdenGravitationalFieldProxy(const double central_mass_density,
                                   const double polytropic_index)
      : LaneEmdenGravitationalField(central_mass_density, polytropic_index) {}

  void apply(
      const gsl::not_null<tnsr::I<DataVector, 3>*> source_momentum_density,
      const gsl::not_null<Scalar<DataVector>*> source_energy_density,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, 3>& momentum_density,
      const tnsr::I<DataVector, 3>& coords) const {
    get(*source_energy_density) = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      source_momentum_density->get(i) = 0.0;
    }
    // EOS is unused
    const EquationsOfState::IdealFluid<false> eos{5.0 / 3.0};
    Scalar<DataVector> source_mass_density_cons{};
    static_cast<const NewtonianEuler::Sources::LaneEmdenGravitationalField&>(
        *this)(make_not_null(&source_mass_density_cons),
               source_momentum_density, source_energy_density,
               mass_density_cons, momentum_density, {}, {}, {}, {}, eos, coords,
               0.0);
  }
};

void test_sources() {
  // Nothing special about these values, we just want them to be non-unity and
  // to be different from each other:
  const double central_mass_density = 0.7;
  const double polytropic_constant = 2.0;
  LaneEmdenGravitationalFieldProxy source_proxy(central_mass_density,
                                                polytropic_constant);
  pypp::check_with_random_values<3>(
      &LaneEmdenGravitationalFieldProxy::apply, source_proxy,
      "Evolution.Systems.NewtonianEuler.Sources.LaneEmdenGravitationalField",
      {"source_momentum_density", "source_energy_density"},
      {{{0.0, 3.0},
        {-1.0, 1.0},
        // with polytropic_constant == 2, star has outer radius ~ 1.77
        {-2.0, 2.0}}},
      std::make_tuple(central_mass_density, polytropic_constant),
      DataVector(5));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.NewtonianEuler.Sources.LaneEmdenGravitationalField",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  test_sources();
}
