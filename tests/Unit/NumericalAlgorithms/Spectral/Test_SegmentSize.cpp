// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Spectral.SegmentSize", "[NumericalAlgorithms][Unit]") {
  CHECK(get_output(Spectral::SegmentSize::Uninitialized) == "Uninitialized");
  CHECK(get_output(Spectral::SegmentSize::Full) == "Full");
  CHECK(get_output(Spectral::SegmentSize::UpperHalf) == "UpperHalf");
  CHECK(get_output(Spectral::SegmentSize::LowerHalf) == "LowerHalf");
}
