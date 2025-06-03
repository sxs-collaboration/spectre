// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeodesicEquation.hpp"

namespace gr {

SPECTRE_TEST_CASE("Unit.GeneralRelativity.GeodesicEquation",
                  "[Unit][PointwiseFunctions]") {
  MAKE_GENERATOR(generator);
  const pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity");
  pypp::check_with_random_values<1>(
      &geodesic_equation<double, 3, Frame::Inertial>, "GeodesicEquation",
      {"dt_x", "dt_pi", "dt_lnp0"}, {{{-1., 1.}}}, DataVector(5));
}

}  // namespace gr
