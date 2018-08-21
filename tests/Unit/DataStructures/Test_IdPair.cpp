// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>
#include <vector>

#include "DataStructures/IdPair.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.IdPair", "[Unit][DataStructures]") {
  auto id_pair = make_id_pair(5, std::vector<double>{1.2, 7.8});
  static_assert(cpp17::is_same_v<decltype(id_pair.id), int>,
                "Failed checking type decay in make_id_pair");
  static_assert(cpp17::is_same_v<decltype(id_pair.data), std::vector<double>>,
                "Failed checking type decay in make_id_pair");
  CHECK(id_pair.id == 5);
  CHECK(id_pair.data == std::vector<double>{1.2, 7.8});
  CHECK(id_pair == make_id_pair(5, std::vector<double>{1.2, 7.8}));
  CHECK(id_pair != make_id_pair(6, std::vector<double>{1.2, 7.8}));
  CHECK(id_pair != make_id_pair(5, std::vector<double>{1.3, 7.8}));
  CHECK(id_pair != make_id_pair(7, std::vector<double>{1.3, 7.8}));

  // Test stream operator
  CHECK(get_output(id_pair) == "(5,(1.2,7.8))");

  // Test PUP
  test_serialization(id_pair);
}
