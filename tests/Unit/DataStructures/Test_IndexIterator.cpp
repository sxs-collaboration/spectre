// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/IndexIterator.hpp"
#include "tests/Unit/TestHelpers.hpp"

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
  CHECK(index_iterator);
  CHECK((index_iterator()[0] == 0 and index_iterator()[1] == 0 and
         index_iterator()[2] == 0));
  CHECK(index_iterator.offset() == 0);
  ++index_iterator;
  CHECK(index_iterator);
  CHECK((index_iterator()[0] == 0 and index_iterator()[1] == 1 and
         index_iterator()[2] == 0));
  CHECK(index_iterator.offset() == 1);
  ++index_iterator;
  CHECK(index_iterator);
  CHECK((index_iterator()[0] == 0 and index_iterator()[1] == 0 and
         index_iterator()[2] == 1));
  CHECK(index_iterator.offset() == 2);
  ++index_iterator;
  CHECK(index_iterator);
  CHECK((index_iterator()[0] == 0 and index_iterator()[1] == 1 and
         index_iterator()[2] == 1));
  CHECK(index_iterator.offset() == 3);
  ++index_iterator;
  CHECK(index_iterator);
  CHECK((index_iterator()[0] == 0 and index_iterator()[1] == 0 and
         index_iterator()[2] == 2));
  CHECK(index_iterator.offset() == 4);
  ++index_iterator;
  CHECK(index_iterator);
  CHECK((index_iterator()[0] == 0 and index_iterator()[1] == 1 and
         index_iterator()[2] == 2));
  CHECK(index_iterator.offset() == 5);
  ++index_iterator;
  CHECK(not index_iterator);

  // Test 0D IndexIterator
  IndexIterator<0> index_iterator_0d(Index<0>{});
  CHECK(index_iterator_0d);
  CHECK_FALSE(++index_iterator_0d);
}
