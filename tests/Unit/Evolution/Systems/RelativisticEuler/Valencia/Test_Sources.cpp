// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RelativisticEuler/Valencia/Sources.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

namespace {

template <size_t Dim>
void test_sources(const DataVector& used_for_size) {
  pypp::check_with_random_values<1>(
      &RelativisticEuler::Valencia::ComputeSources<Dim>::apply, "TestFunctions",
      {"source_tilde_tau", "source_tilde_s"}, {{{0.0, 1.0}}}, used_for_size);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.Sources",
                  "[Unit][RelativisticEuler]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/RelativisticEuler/Valencia"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_sources, (1, 2, 3))
}
