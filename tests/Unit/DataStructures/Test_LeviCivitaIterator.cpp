// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.LeviCivitaIterator",
                  "[DataStructures][Unit]") {
  // Test 2D
  std::array<int, 6> signs_2d = {{1, -1}};
  std::array<Index<2>, 2> indexes_2d = {{Index<2>(0, 1), Index<2>(1, 0)}};

  size_t i = 0;
  for (LeviCivitaIterator<2> it; it; ++it) {
    CHECK(it() == indexes_2d[i]);
    CHECK(it.sign() == signs_2d[i]);
    ++i;
  }

  // Test 3D
  std::array<int, 6> signs_3d = {{1, -1, -1, 1, 1, -1}};
  std::array<Index<3>, 6> indexes_3d = {{Index<3>(0, 1, 2), Index<3>(0, 2, 1),
                                         Index<3>(1, 0, 2), Index<3>(1, 2, 0),
                                         Index<3>(2, 0, 1), Index<3>(2, 1, 0)}};

  i = 0;
  for (LeviCivitaIterator<3> it; it; ++it) {
    CHECK(it() == indexes_3d[i]);
    CHECK(it.sign() == signs_3d[i]);
    ++i;
  }

  // Test 4D
  std::array<int, 24> signs_4d = {{1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1,
                                   1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1}};
  std::array<Index<4>, 24> indexes_4d = {
      {Index<4>(0, 1, 2, 3), Index<4>(0, 1, 3, 2), Index<4>(0, 2, 1, 3),
       Index<4>(0, 2, 3, 1), Index<4>(0, 3, 1, 2), Index<4>(0, 3, 2, 1),
       Index<4>(1, 0, 2, 3), Index<4>(1, 0, 3, 2), Index<4>(1, 2, 0, 3),
       Index<4>(1, 2, 3, 0), Index<4>(1, 3, 0, 2), Index<4>(1, 3, 2, 0),
       Index<4>(2, 0, 1, 3), Index<4>(2, 0, 3, 1), Index<4>(2, 1, 0, 3),
       Index<4>(2, 1, 3, 0), Index<4>(2, 3, 0, 1), Index<4>(2, 3, 1, 0),
       Index<4>(3, 0, 1, 2), Index<4>(3, 0, 2, 1), Index<4>(3, 1, 0, 2),
       Index<4>(3, 1, 2, 0), Index<4>(3, 2, 0, 1), Index<4>(3, 2, 1, 0)}};

  i = 0;
  for (LeviCivitaIterator<4> it; it; ++it) {
    CHECK(it() == indexes_4d[i]);
    CHECK(it.sign() == signs_4d[i]);
    ++i;
  }
}
