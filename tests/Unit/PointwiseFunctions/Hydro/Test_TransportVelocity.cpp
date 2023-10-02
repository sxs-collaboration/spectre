// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/TransportVelocity.hpp"

namespace hydro {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.TransportVelocity",
                  "[Unit][Hydro]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Hydro/");
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(
      &transport_velocity<DataVector, 1, Frame::Inertial>, "TransportVelocity",
      {"transport_velocity"}, {{{0.0, 1.0}}}, used_for_size);
  pypp::check_with_random_values<1>(
      &transport_velocity<DataVector, 3, Frame::Inertial>, "TransportVelocity",
      {"transport_velocity"}, {{{0.0, 1.0}}}, used_for_size);
  TestHelpers::db::test_compute_tag<
      Tags::TransportVelocityCompute<DataVector, 3, Frame::Inertial>>(
      "TransportVelocity");
}

}  // namespace hydro
