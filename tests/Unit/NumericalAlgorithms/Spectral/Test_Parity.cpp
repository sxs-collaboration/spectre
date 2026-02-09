// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Spectral.Parity", "[NumericalAlgorithms][Unit]") {
  CHECK(get_output(Spectral::Parity::Uninitialized) == "Uninitialized");
  CHECK(get_output(Spectral::Parity::Even) == "Even");
  CHECK(get_output(Spectral::Parity::Odd) == "Odd");
}
