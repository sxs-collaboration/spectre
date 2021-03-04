// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Comparator.hpp"

namespace {
void check(const std::string& name, const bool expected_less,
           const bool expected_equal, const bool expected_greater) noexcept {
  const auto comparator = TestHelpers::test_creation<Options::Comparator>(name);
  CHECK(comparator(2.0, 3.0) == expected_less);
  CHECK(comparator(2.0, 2.0) == expected_equal);
  CHECK(comparator(2.0, 1.0) == expected_greater);

  const auto copy = serialize_and_deserialize(comparator);
  CHECK(copy(2.0, 3.0) == expected_less);
  CHECK(copy(2.0, 2.0) == expected_equal);
  CHECK(copy(2.0, 1.0) == expected_greater);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Options.Comparator", "[Unit][Options]") {
  check("EqualTo", false, true, false);
  check("NotEqualTo", true, false, true);
  check("LessThan", true, false, false);
  check("GreaterThan", false, false, true);
  check("LessThanOrEqualTo", true, true, false);
  check("GreaterThanOrEqualTo", false, true, true);
}
