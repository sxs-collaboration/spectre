// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/Tags/Section.hpp"

namespace Parallel {
namespace {
struct ParallelComponent;
struct SectionIdTag {
  using type = int;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.Tags.Section", "[Unit][Parallel]") {
  TestHelpers::db::test_simple_tag<
      Tags::Section<ParallelComponent, SectionIdTag>>("Section(SectionIdTag)");
}

}  // namespace Parallel
