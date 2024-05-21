// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/ArrayCollection/Tags/ElementCollection.hpp"
#include "Parallel/ArrayCollection/Tags/ElementLocations.hpp"
#include "Parallel/ArrayCollection/Tags/ElementLocationsReference.hpp"
#include "Parallel/ArrayCollection/Tags/NumberOfElementsTerminated.hpp"

namespace Parallel {
SPECTRE_TEST_CASE("Unit.Parallel.ArrayCollection.Tags", "[Unit][Parallel]") {
  TestHelpers::db::test_simple_tag<
      Tags::ElementCollection<3, void, void, void>>("ElementCollection");
  TestHelpers::db::test_simple_tag<Tags::ElementLocations<3>>(
      "ElementLocations");
  TestHelpers::db::test_reference_tag<
      Tags::ElementLocationsReference<3, void, void>>("ElementLocations");
  TestHelpers::db::test_simple_tag<Tags::NumberOfElementsTerminated>(
      "NumberOfElementsTerminated");
}

}  // namespace Parallel
