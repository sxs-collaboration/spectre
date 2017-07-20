// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "tests/Unit/TestHelpers.hpp"

TEST_CASE("Test.TestHelpers", "[Unit]") {
  std::vector<double> vector{0, 1, 2, 3};
  test_iterators(vector);
  test_reverse_iterators(vector);

  std::set<double> set;
  set.insert(0);
  set.insert(1);
  set.insert(2);
  set.insert(3);
  test_iterators(set);
  test_reverse_iterators(set);

  std::unordered_set<int> u_set;
  u_set.insert(3);
  u_set.insert(2);
  u_set.insert(1);
  u_set.insert(0);
  test_iterators(u_set);
}
