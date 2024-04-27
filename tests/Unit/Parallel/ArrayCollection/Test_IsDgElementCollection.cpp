// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"

namespace Parallel {
SPECTRE_TEST_CASE("Unit.Parallel.ArrayCollection.IsDgElementCollection",
                  "[Unit][Parallel]") {
  CHECK(is_dg_element_collection_v<DgElementCollection<3, void, void>>);
  CHECK_FALSE(is_dg_element_collection_v<int>);
}
}  // namespace Parallel
