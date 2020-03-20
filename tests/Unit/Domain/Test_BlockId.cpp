// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Domain/BlockId.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.Domain.BlockId", "[Domain][Unit]") {
  domain::BlockId id{10};
  CHECK(id.get_index() == 10);
  CHECK(id == domain::BlockId{10});
  CHECK(id != domain::BlockId{7});
  CHECK(get_output(id) == "[10]");
  CHECK(id++ == domain::BlockId{10});
  CHECK(id == domain::BlockId{11});
  CHECK(id-- == domain::BlockId{11});
  CHECK(id == domain::BlockId{10});
  CHECK(&id == &(++id));
  CHECK(id == domain::BlockId{11});
  CHECK(&id == &(--id));
  CHECK(id == domain::BlockId{10});

  // Test PUP
  test_serialization(id);
}
