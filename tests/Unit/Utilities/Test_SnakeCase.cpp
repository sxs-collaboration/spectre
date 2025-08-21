// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Utilities/SnakeCase.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.SnakeCase", "[Unit][Utilities]") {
  CHECK(camel_case_to_snake_case("CamelCaseString") == "camel_case_string");
  CHECK(snake_case_to_camel_case("snake_case_string") == "SnakeCaseString");
}
