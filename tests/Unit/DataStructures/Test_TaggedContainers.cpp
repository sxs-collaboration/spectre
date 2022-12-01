// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
namespace Tags {
struct MyDouble : db::SimpleTag {
  using type = double;
};
struct MyScalar : db::SimpleTag {
  using type = Scalar<DataVector>;
};
struct MyDouble2 : db::SimpleTag {
  using type = double;
};
}  // namespace Tags

// Test that desired tags are retrieved.
void test_tagged_containers() {
  const auto box = db::create<db::AddSimpleTags<Tags::MyDouble>>(1.2);
  tuples::TaggedTuple<Tags::MyDouble2> tuple(4.4);
  Variables<tmpl::list<Tags::MyScalar>> vars(10ul, 3.2);

  CHECK(get<Tags::MyDouble>(box) == 1.2);
  CHECK(get<Tags::MyDouble2>(tuple) == 4.4);
  CHECK(get<Tags::MyScalar>(vars).get() == 3.2);

  // Note: tests do not pass if DataVector only has a single component!
  const auto box2 = db::create<db::AddSimpleTags<Tags::MyScalar>>(
      Scalar<DataVector>{10ul, 1.2});

  CHECK(get<Tags::MyScalar>(vars, box2).get() == 3.2);
  CHECK(get<Tags::MyScalar>(box2, vars).get() == 1.2);

  CHECK(get<Tags::MyScalar>(tuple, box2, vars).get() == 1.2);
  CHECK(get<Tags::MyScalar>(box2, tuple, vars).get() == 1.2);
  CHECK(get<Tags::MyScalar>(box2, vars, tuple).get() == 1.2);

  CHECK(get<Tags::MyDouble>(box, tuple) == 1.2);
  CHECK(get<Tags::MyDouble2>(tuple, vars) == 4.4);
  CHECK(get<Tags::MyScalar>(vars, box).get() == 3.2);

  CHECK(get<Tags::MyDouble>(tuple, box) == 1.2);
  CHECK(get<Tags::MyDouble2>(vars, tuple) == 4.4);
  CHECK(get<Tags::MyScalar>(box, vars).get() == 3.2);

  CHECK(get<Tags::MyDouble>(box, tuple, vars) == 1.2);
  CHECK(get<Tags::MyDouble2>(tuple, vars, box) == 4.4);
  CHECK(get<Tags::MyScalar>(vars, box, tuple).get() == 3.2);

  CHECK(get<Tags::MyDouble>(box, vars, tuple) == 1.2);
  CHECK(get<Tags::MyDouble2>(tuple, box, vars) == 4.4);
  CHECK(get<Tags::MyScalar>(vars, tuple, box).get() == 3.2);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.TaggedContainers",
                  "[Unit][DataStructures]") {
  test_tagged_containers();
}
