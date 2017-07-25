// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <numeric>

#include "DataStructures/Index.hpp"
#include "Utilities/Literals.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Index", "[DataStructures][Unit]") {
  Index<0> index_0d;
  CHECK(index_0d.product() == 1);
  Index<1> index_1d(3);
  CHECK(index_1d.product() == 3);
  CHECK(index_1d[0] == 3);
  const Index<2> index_2d(4);
  CHECK(index_2d.product() == 16);
  CHECK(index_2d[0] == 4);
  CHECK(index_2d[1] == 4);
  CHECK(index_2d.slice_away(0)[0] == 4);
  CHECK(index_2d.slice_away(1)[0] == 4);
  CHECK(index_2d.data()[0] == 4);
  CHECK(index_2d.data()[1] == 4);
  Index<3> index_3d(1, 2, 3);
  CHECK(index_3d.product() == 6);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(index_3d[i] == i + 1);
    CHECK(index_3d.data()[i] == i + 1);
  }
  CHECK(index_3d.slice_away(0)[0] == 2);
  CHECK(index_3d.slice_away(0)[1] == 3);
  CHECK(index_3d.slice_away(0).product() == 6);
  CHECK(index_3d.slice_away(1)[0] == 1);
  CHECK(index_3d.slice_away(1)[1] == 3);
  CHECK(index_3d.slice_away(1).product() == 3);
  CHECK(index_3d.slice_away(2)[0] == 1);
  CHECK(index_3d.slice_away(2)[1] == 2);
  CHECK(index_3d.slice_away(2).product() == 2);

  // Check iterator
  CHECK(6 == std::accumulate(index_3d.begin(), index_3d.end(), 0));

  // Check serialization
  CHECK(index_3d == serialize_and_deserialize(index_3d));

  // Test inequivalence operator
  CHECK(index_3d != Index<3>(2, 4, 9));

  test_copy_semantics(index_3d);
  auto index_3d_copy = index_3d;
  test_move_semantics(std::move(index_3d), index_3d_copy);
}
