// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"

// IWYU pragma: no_include "DataStructures/Tensor/Tensor.hpp"
// IWYU pragma: no_include "Utilities/Gsl.hpp"
// IWYU pragma: no_include <string>

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Sources", "[Unit][GrMhd]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  pypp::check_with_random_values<1>(
      &grmhd::ValenciaDivClean::ComputeSources::apply, "TestFunctions",
      {"source_tilde_tau", "source_tilde_s", "source_tilde_b",
       "source_tilde_phi"},
      {{{0.0, 1.0}}}, DataVector{5});
}
