// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>

#include "Utilities/IsNan.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.IsNan", "[Unit][Utilities]") {
  CHECK_FALSE(isnan_safe(0.));
  CHECK_FALSE(isnan_safe(0.f));
  CHECK(isnan_safe(std::numeric_limits<double>::signaling_NaN()));
  CHECK(isnan_safe(std::numeric_limits<float>::signaling_NaN()));
  CHECK(isnan_safe(std::numeric_limits<double>::quiet_NaN()));
  CHECK(isnan_safe(std::numeric_limits<float>::quiet_NaN()));
}
