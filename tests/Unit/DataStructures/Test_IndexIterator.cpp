// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.IndexIterator",
                  "[DataStructures][Unit]") {
  /// [index_iterator_example]
  Index<3> elements(1, 2, 3);
  for (IndexIterator<3> index_it(elements); index_it; ++index_it) {
    // Use the index iterator to do something super awesome
    CHECK(Index<3>(-1, -1, -1) != *index_it);
  }
  /// [index_iterator_example]

  IndexIterator<3> index_iterator(elements);
  auto check_next = [&index_iterator,
                     call_num = 0](const Index<3>& expected) mutable noexcept {
    CHECK(index_iterator);
    CHECK(index_iterator() == expected);
    CHECK(*index_iterator == expected);
    CHECK(index_iterator->indices() == expected.indices());
    CHECK(index_iterator.collapsed_index() == call_num);
    ++index_iterator;
    ++call_num;
  };
  check_next(Index<3>(0, 0, 0));
  check_next(Index<3>(0, 1, 0));
  check_next(Index<3>(0, 0, 1));
  check_next(Index<3>(0, 1, 1));
  check_next(Index<3>(0, 0, 2));
  check_next(Index<3>(0, 1, 2));
  CHECK(not index_iterator);

  // Test 0D IndexIterator
  IndexIterator<0> index_iterator_0d(Index<0>{});
  CHECK(index_iterator_0d);
  CHECK_FALSE(++index_iterator_0d);
}
