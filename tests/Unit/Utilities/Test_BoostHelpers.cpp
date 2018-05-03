// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <string>

#include "Utilities/BoostHelpers.hpp"  // IWYU pragma: associated
#include "Utilities/TMPL.hpp"
#include "tests/Unit/TestHelpers.hpp"

static_assert(
    cpp17::is_same_v<boost::variant<double, int, char>,
                     make_boost_variant_over<tmpl::list<double, int, char>>>,
    "Failed testing make_variant_over");

SPECTRE_TEST_CASE("Unit.Utilities.BoostHelpers.Variant.Pup",
                  "[Unit][Utilities]") {
  // The order of checks is arbitrary but jumps around to make sure that it is
  // general enough to catch bugs.
  boost::variant<int, double, char, std::string> var{"aoeu"};
  boost::variant<int, double, char, std::string> var2 =
      serialize_and_deserialize(var);
  CHECK(boost::get<std::string>(var) == boost::get<std::string>(var2));

  var = 5;
  var2 = serialize_and_deserialize(var);
  CHECK(boost::get<int>(var) == boost::get<int>(var2));

  var = 'A';
  var2 = serialize_and_deserialize(var);
  CHECK(boost::get<char>(var) == boost::get<char>(var2));


  var = 7.2341125387;
  var2 = serialize_and_deserialize(var);
  CHECK(boost::get<double>(var) == boost::get<double>(var2));
}

SPECTRE_TEST_CASE("Unit.Utilities.BoostHelpers.Variant.Names",
                  "[Unit][Utilities]") {
  boost::variant<int, double, char, std::string> var{"aoeu"};
  CHECK(type_of_current_state(var) == "std::string");
  CHECK(type_of_current_state(var = 1) == "int");
  CHECK(type_of_current_state(var = 'A') == "char");
  CHECK(type_of_current_state(var = 2.8) == "double");
}

SPECTRE_TEST_CASE("Unit.Utilities.BoostHelpers.Optional.Pup",
                  "[Unit][Utilities]") {
  test_serialization(boost::optional<double>{});
  test_serialization(boost::optional<double>{1.2});
}
