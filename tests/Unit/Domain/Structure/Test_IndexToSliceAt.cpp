// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"

SPECTRE_TEST_CASE("Unit.Domain.Structure.IndexToSliceAt", "[Domain][Unit]") {
  const Index<2> extents{{{2, 5}}};
  CHECK(index_to_slice_at(extents, Direction<2>::lower_xi()) == 0);
  CHECK(index_to_slice_at(extents, Direction<2>::lower_xi(), 1) == 1);
  CHECK(index_to_slice_at(extents, Direction<2>::upper_xi()) == 1);
  CHECK(index_to_slice_at(extents, Direction<2>::upper_xi(), 1) == 0);
  CHECK(index_to_slice_at(extents, Direction<2>::lower_eta()) == 0);
  CHECK(index_to_slice_at(extents, Direction<2>::lower_eta(), 1) == 1);
  CHECK(index_to_slice_at(extents, Direction<2>::upper_eta()) == 4);
  CHECK(index_to_slice_at(extents, Direction<2>::upper_eta(), 1) == 3);
}
