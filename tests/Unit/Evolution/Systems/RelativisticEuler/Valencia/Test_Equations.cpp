// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RelativisticEuler/Valencia/Equations.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

namespace {

template <size_t Dim>
void test_source_terms(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(
      &RelativisticEuler::Valencia::compute_source_terms_of_u<Dim>,
      "TestFunctions", {"source_tilde_d", "source_tilde_tau", "source_tilde_s"},
      {{{0.0, 1.0}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.Equations",
                  "[Unit][RelativisticEuler]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/RelativisticEuler/Valencia"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_source_terms, (1, 2, 3))
}
