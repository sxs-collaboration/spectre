// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <optional>

#include "Utilities/OptionalHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.OptionalHelpers", "[Unit][Utilities]") {
  const int t = 10;
  std::optional<int> t_opt{12};
  std::optional<int> t_no_opt{std::nullopt};

  CHECK(has_value(t));
  CHECK(has_value(t_opt));
  CHECK(not has_value(t_no_opt));

  CHECK(value(t) == 10);
  int t_mutable = 0;
  value(t_mutable) = value(t) + 10;
  CHECK(value(t_mutable) == 20);

  CHECK(value(t_opt) == 12);
  value(t_opt) = 15;
  CHECK(value(t_opt) == 15);
}
