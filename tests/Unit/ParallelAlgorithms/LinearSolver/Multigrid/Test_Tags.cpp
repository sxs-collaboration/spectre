// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"

namespace LinearSolver::multigrid {

namespace {
struct Tag : db::SimpleTag {
  using type = int;
};
struct TestSolver {};
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.LinearSolver.Multigrid.Tags",
                  "[Unit][ParallelAlgorithms][LinearSolver]") {
  TestHelpers::db::test_simple_tag<Tags::ChildrenRefinementLevels<1>>(
      "ChildrenRefinementLevels");
  TestHelpers::db::test_simple_tag<Tags::ParentRefinementLevels<1>>(
      "ParentRefinementLevels");
  TestHelpers::db::test_simple_tag<Tags::MaxLevels<TestSolver>>(
      "MaxLevels(TestSolver)");
  TestHelpers::db::test_simple_tag<Tags::OutputVolumeData<TestSolver>>(
      "OutputVolumeData(TestSolver)");
  TestHelpers::db::test_simple_tag<Tags::MultigridLevel>("MultigridLevel");
  TestHelpers::db::test_simple_tag<Tags::IsFinestGrid>("IsFinestGrid");
  TestHelpers::db::test_simple_tag<Tags::ParentId<1>>("ParentId");
  TestHelpers::db::test_simple_tag<Tags::ChildIds<1>>("ChildIds");
  TestHelpers::db::test_simple_tag<Tags::ParentMesh<1>>("ParentMesh");
  TestHelpers::db::test_simple_tag<Tags::ObservationId<TestSolver>>(
      "ObservationId(TestSolver)");
  TestHelpers::db::test_prefix_tag<Tags::PreSmoothingInitial<Tag>>(
      "PreSmoothingInitial(Tag)");
  TestHelpers::db::test_prefix_tag<Tags::PreSmoothingSource<Tag>>(
      "PreSmoothingSource(Tag)");
  TestHelpers::db::test_prefix_tag<Tags::PreSmoothingResult<Tag>>(
      "PreSmoothingResult(Tag)");
  TestHelpers::db::test_prefix_tag<Tags::PreSmoothingResidual<Tag>>(
      "PreSmoothingResidual(Tag)");
  TestHelpers::db::test_prefix_tag<Tags::PostSmoothingInitial<Tag>>(
      "PostSmoothingInitial(Tag)");
  TestHelpers::db::test_prefix_tag<Tags::PostSmoothingSource<Tag>>(
      "PostSmoothingSource(Tag)");
  TestHelpers::db::test_prefix_tag<Tags::PostSmoothingResult<Tag>>(
      "PostSmoothingResult(Tag)");
  TestHelpers::db::test_prefix_tag<Tags::PostSmoothingResidual<Tag>>(
      "PostSmoothingResidual(Tag)");
  TestHelpers::db::test_simple_tag<Tags::VolumeDataForOutput<
      TestSolver, ::Tags::Variables<tmpl::list<::Tags::TempScalar<0>>>>>(
      "VolumeDataForOutput(TestSolver)");
}

}  // namespace LinearSolver::multigrid
