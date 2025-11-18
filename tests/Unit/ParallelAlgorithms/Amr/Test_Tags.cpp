// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "ParallelAlgorithms/Amr/Tags.hpp"

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Amr.Tags",
                  "[ParallelAlgorithms][Unit]") {
  TestHelpers::db::test_simple_tag<amr::Tags::AmrBlocks<1>>("AmrBlocks");
  TestHelpers::db::test_simple_tag<amr::Tags::AllElementIds<1>>(
      "AllElementIds");
  TestHelpers::db::test_simple_tag<amr::Tags::ParentId<1>>("ParentId");
  TestHelpers::db::test_simple_tag<amr::Tags::ChildIds<1>>("ChildIds");
  TestHelpers::db::test_simple_tag<amr::Tags::ParentMesh<1>>("ParentMesh");
  TestHelpers::db::test_compute_tag<
      amr::Tags::GridIndexObservationKeyCompute<1>>(
      "ObservationKey(GridIndex)");
  TestHelpers::db::test_compute_tag<
      amr::Tags::IsFinestGridObservationKeyCompute<1>>(
      "ObservationKey(IsFinestGrid)");
}
