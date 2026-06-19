// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/NullGeodesic.hpp"

namespace gr {

namespace {

}  // namespace

SPECTRE_TEST_CASE("Unit.GeneralRelativity.NullGeodesic",
                  "[Unit][PointwiseFunctions]") {
  MAKE_GENERATOR(generator);
  const pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/GeneralRelativity");

  pypp::check_with_random_values<1>(
      &photon_geodesic_equation_with_constraint<double, 3, Frame::Inertial>,
      "NullGeodesic",
      {"dt_x", "dt_pi", "current_p0", "current_dt_lnp0"},
      {{{0.1, 1.}}},
      DataVector(5));
}

}  // namespace gr
