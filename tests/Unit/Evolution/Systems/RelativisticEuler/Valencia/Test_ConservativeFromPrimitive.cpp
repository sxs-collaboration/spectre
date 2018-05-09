// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/ConservativeFromPrimitive.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {

template <size_t Dim, typename DataType>
void test_conservative_from_primitive(const DataType& used_for_size) noexcept {
  pypp::check_with_random_values<1>(
      &RelativisticEuler::Valencia::conservative_from_primitive<DataType, Dim>,
      "TestFunctions", {"tilde_d", "tilde_tau", "tilde_s"}, {{{0.0, 1.0}}},
      used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.ConservativeFromPrimitive",
                  "[Unit][RelativisticEuler]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/RelativisticEuler/Valencia"};

  const double d = std::numeric_limits<double>::signaling_NaN();
  test_conservative_from_primitive<1>(d);
  test_conservative_from_primitive<2>(d);
  test_conservative_from_primitive<3>(d);

  const DataVector dv(5);
  test_conservative_from_primitive<1>(dv);
  test_conservative_from_primitive<2>(dv);
  test_conservative_from_primitive<3>(dv);
}
