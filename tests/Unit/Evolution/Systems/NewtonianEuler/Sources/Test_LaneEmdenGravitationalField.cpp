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
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/LaneEmdenStar.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {
// Need this proxy in order for pypp to evaluate function whose arguments
// include a variable of type `NewtonianEuler::Solutions::LaneEmdenStar`
struct LaneEmdenGravitationalFieldProxy
    : NewtonianEuler::Sources::LaneEmdenGravitationalField {
  using NewtonianEuler::Sources::LaneEmdenGravitationalField::
      LaneEmdenGravitationalField;
  LaneEmdenGravitationalFieldProxy(const double central_mass_density,
                                   const double polytropic_index)
      : star_(central_mass_density, polytropic_index) {}

  void apply_helper(
      const gsl::not_null<tnsr::I<DataVector, 3>*> source_momentum_density,
      const gsl::not_null<Scalar<DataVector>*> source_energy_density,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, 3>& momentum_density,
      const tnsr::I<DataVector, 3>& x) const noexcept {
    this->apply(source_momentum_density, source_energy_density,
                mass_density_cons, momentum_density, star_, x);
  }

 private:
  NewtonianEuler::Solutions::LaneEmdenStar star_{};
};

void test_sources() noexcept {
  // Nothing special about these values, we just want them to be non-unity and
  // to be different from each other:
  const double central_mass_density = 0.7;
  const double polytropic_constant = 2.0;
  LaneEmdenGravitationalFieldProxy source_proxy(central_mass_density,
                                                polytropic_constant);
  pypp::check_with_random_values<3>(
      &LaneEmdenGravitationalFieldProxy::apply_helper, source_proxy,
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
