// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/Structure/ObjectLabel.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Domain.ObjectLabel", "[Domain][Unit]") {
  CHECK(name(domain::ObjectLabel::A) == "A");
  CHECK(get_output(domain::ObjectLabel::A) == "A");
  CHECK(name(domain::ObjectLabel::B) == "B");
  CHECK(get_output(domain::ObjectLabel::B) == "B");
  CHECK(name(domain::ObjectLabel::None) == "");
  CHECK(get_output(domain::ObjectLabel::None) == "");
}
