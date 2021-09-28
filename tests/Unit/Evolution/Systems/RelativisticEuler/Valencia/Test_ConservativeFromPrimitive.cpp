// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/ConservativeFromPrimitive.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

namespace {

template <size_t Dim, typename DataType>
void test_conservative_from_primitive(const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      &RelativisticEuler::Valencia::ConservativeFromPrimitive<Dim>::apply,
      "TestFunctions", {"tilde_d", "tilde_tau", "tilde_s"}, {{{0.0, 1.0}}},
      used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.ConservativeFromPrimitive",
                  "[Unit][RelativisticEuler]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/RelativisticEuler/Valencia"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_conservative_from_primitive, (1, 2, 3))
}
