// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "ApparentHorizons/ObjectLabel.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.ApparentHorizons.ObjectLabel",
                  "[ApparentHorizons][Unit]") {
  CHECK(name(ah::ObjectLabel::A) == "A");
  CHECK(get_output(ah::ObjectLabel::A) == "A");
  CHECK(name(ah::ObjectLabel::B) == "B");
  CHECK(get_output(ah::ObjectLabel::B) == "B");
}
