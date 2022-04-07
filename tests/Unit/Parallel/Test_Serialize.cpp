// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "Parallel/Serialize.hpp"

SPECTRE_TEST_CASE("Unit.Parallel.Serialize", "[Unit][Parallel]") {
  CHECK(size_of_object_in_bytes(std::array<double, 4>{}) == 4 * sizeof(double));
  CHECK(
      size_of_object_in_bytes(std::vector<double>(10)) ==
      (10 * sizeof(double) + sizeof(decltype(std::vector<double>(10).size()))));
}
