// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/ValidateSelection.hpp"
#include "Options/Context.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct Tag1 : db::SimpleTag {
  using type = int;
};

struct Tag2 : db::SimpleTag {
  using type = int;
  static std::string name() { return "CustomTag2"; }
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Options.ValidateSelection", "[Unit][Options]") {
  db::validate_selection<tmpl::list<Tag1, Tag2>>({"Tag1"}, Options::Context{});
  CHECK_THROWS_WITH((db::validate_selection<tmpl::list<Tag1, Tag2>>(
                        {"Tag2"}, Options::Context{})),
                    Catch::Matchers::ContainsSubstring("Invalid selection"));
  db::validate_selection<tmpl::list<Tag1, Tag2>>({"CustomTag2", "Tag1"},
                                                 Options::Context{});
}
