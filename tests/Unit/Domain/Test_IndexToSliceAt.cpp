// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/Index.hpp"
#include "Domain/Direction.hpp"
#include "Domain/IndexToSliceAt.hpp"

SPECTRE_TEST_CASE("Unit.Domain.IndexToSliceAt", "[Domain][Unit]") {
  const Index<2> extents{{{2, 5}}};
  CHECK(index_to_slice_at(extents, Direction<2>::lower_xi()) == 0);
  CHECK(index_to_slice_at(extents, Direction<2>::upper_xi()) == 1);
  CHECK(index_to_slice_at(extents, Direction<2>::lower_eta()) == 0);
  CHECK(index_to_slice_at(extents, Direction<2>::upper_eta()) == 4);
}
