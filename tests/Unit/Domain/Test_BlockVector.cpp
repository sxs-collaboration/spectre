// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "Domain/BlockId.hpp"
#include "Domain/BlockVector.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.BlockVector", "[Domain][Unit]") {
  using domain::BlockId;
  domain::BlockVector<double> block_vector{1.3, 4.8, 7.8};
  CHECK(block_vector[BlockId{0}] == 1.3);
  CHECK(block_vector[BlockId{1}] == 4.8);
  CHECK(block_vector[BlockId{2}] == 7.8);
  CHECK(block_vector == domain::BlockVector<double>{1.3, 4.8, 7.8});
  CHECK(get_output(block_vector) == "(1.3,4.8,7.8)");

  domain::BlockVector<double> block_vector2{8.3, 2.8, -7.8};
  using std::swap;
  swap(block_vector, block_vector2);
  CHECK(block_vector == domain::BlockVector<double>{8.3, 2.8, -7.8});
  CHECK(block_vector2 == domain::BlockVector<double>{1.3, 4.8, 7.8});

  check_cmp(block_vector2, block_vector);

  // Test PUP
  test_serialization(block_vector);
}
