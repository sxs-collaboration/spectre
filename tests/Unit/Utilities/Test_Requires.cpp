// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <list>
#include <string>
#include <vector>

#include "Utilities/Requires.hpp"

namespace {
/// [function_definitions]
/// [foo_definition]
template <typename T, Requires<tt::is_a_v<std::vector, T>> = nullptr>
std::string foo(const T& /*unused*/) {
  return "vector";
}
/// [foo_definition]

template <typename T, Requires<tt::is_a_v<std::list, T>> = nullptr>
std::string foo(const T& /*unused*/) {
  return "list";
}
/// [function_definitions]
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.Requires", "[Unit][Utilities]") {
  CHECK("vector" == foo(std::vector<double>{}));
  CHECK("list" == foo(std::list<double>{}));
}
