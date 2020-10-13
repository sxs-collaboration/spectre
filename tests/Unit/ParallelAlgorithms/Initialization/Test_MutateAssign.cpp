// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Tag0 : db::SimpleTag {
  using type = double;
};

struct Tag1 : db::SimpleTag {
  using type = std::vector<double>;
};

struct Tag2Base : db::BaseTag {};

struct Tag2 : db::SimpleTag, Tag2Base {
  using type = std::string;
};

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Initialization.MutateAssign",
                  "[Unit][ParallelAlgorithms]") {
  auto box = db::create<db::AddSimpleTags<Tag0, Tag1, Tag2>>(
      0.5, std::vector<double>{1.0, 2.0}, std::string("original string value"));
  Initialization::mutate_assign<tmpl::list<Tag0>>(make_not_null(&box), 1.2);
  CHECK(db::get<Tag0>(box) == 1.2);
  CHECK(db::get<Tag1>(box) == std::vector<double>{1.0, 2.0});
  Initialization::mutate_assign<tmpl::list<Tag0, Tag1>>(
      make_not_null(&box), 1.5, std::vector<double>{1.6, 2.5, 3.0});
  CHECK(db::get<Tag0>(box) == 1.5);
  CHECK(db::get<Tag1>(box) == std::vector<double>{1.6, 2.5, 3.0});
  Initialization::mutate_assign<tmpl::list<Tag2Base>>(make_not_null(&box),
                                                      "new string value");
  CHECK(db::get<Tag2>(box) == "new string value");
  CHECK(db::get<Tag2Base>(box) == "new string value");
}
}  // namespace
