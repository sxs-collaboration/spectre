// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "Evolution/Systems/CurvedScalarWave/Worldtube/SelfForce.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave::Worldtube {
namespace {

void test_self_force_acceleration() {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/CurvedScalarWave/Worldtube"};
  pypp::check_with_random_values<1>(
      static_cast<tnsr::I<double, 3> (*)(
          const Scalar<double>&, const tnsr::i<double, 3>&,
          const tnsr::I<double, 3>&, const double, const double,
          const tnsr::AA<double, 3>&, const Scalar<double>&)>(
          self_force_acceleration<3>),
      "SelfForce", "self_force_acceleration", {{{-2.0, 2.0}}}, 1);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CSW.Worldtube.SelfForce",
                  "[Unit][Evolution]") {
  test_self_force_acceleration();
}
}  // namespace
}  // namespace CurvedScalarWave::Worldtube
