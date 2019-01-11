// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>

#include "Utilities/IsNan.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.IsNan", "[Unit][Utilities]") {
  CHECK_FALSE(is_nan(0.));
  CHECK_FALSE(is_nan(0.f));
  CHECK(is_nan(std::numeric_limits<double>::signaling_NaN()));
  CHECK(is_nan(std::numeric_limits<float>::signaling_NaN()));
}
