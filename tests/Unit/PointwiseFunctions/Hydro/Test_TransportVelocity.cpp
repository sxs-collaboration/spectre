// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/TransportVelocity.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace hydro {
namespace {
template <size_t Dim, typename Frame, typename DataType>
void test_transport_velocity(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(&transport_velocity<DataType, Dim, Frame>,
                                    "TestFunctions", "transport_velocity",
                                    {{{0.0, 1.0 / sqrt(Dim)}}}, used_for_size);

  pypp::check_with_random_values<1>(&transport_velocity<DataType, Dim, Frame>,
                                    "TestFunctions", "transport_velocity",
                                    {{{-1.0 / sqrt(Dim), 0.0}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.TransportVelocity",
                  "[Unit][Hydro]") {
  pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/Hydro/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(test_transport_velocity, (1, 2, 3),
                                    (Frame::Inertial, Frame::Grid));
}
}  // namespace hydro
