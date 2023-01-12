// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ComovingMagneticFieldMagnitude.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.ComovingMagneticFieldMagnitude",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  pypp::check_with_random_values<1>(
      &grmhd::ValenciaDivClean::Tags::ComovingMagneticFieldMagnitudeCompute::
          function,
      "TestFunctions", {"comoving_b_magnitude"}, {{{0.0, 1.0}}}, DataVector{5});
}
