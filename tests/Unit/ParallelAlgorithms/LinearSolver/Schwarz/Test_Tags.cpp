// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"

namespace {
struct DummyOptionsGroup {};
struct DummySubdomainSolver {};
template <size_t N>
struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace

namespace LinearSolver::Schwarz {

SPECTRE_TEST_CASE("Unit.ParallelSchwarz.Tags",
                  "[Unit][ParallelAlgorithms][LinearSolver]") {
  TestHelpers::db::test_simple_tag<Tags::MaxOverlap<DummyOptionsGroup>>(
      "MaxOverlap(DummyOptionsGroup)");
  TestHelpers::db::test_base_tag<Tags::SubdomainSolverBase<DummyOptionsGroup>>(
      "SubdomainSolver(DummyOptionsGroup)");
  TestHelpers::db::test_simple_tag<
      Tags::SubdomainSolver<DummySubdomainSolver, DummyOptionsGroup>>(
      "SubdomainSolver(DummyOptionsGroup)");
  TestHelpers::db::test_simple_tag<
      Tags::IntrudingExtents<1, DummyOptionsGroup>>(
      "IntrudingExtents(DummyOptionsGroup)");
  TestHelpers::db::test_simple_tag<
      Tags::IntrudingOverlapWidths<1, DummyOptionsGroup>>(
      "IntrudingOverlapWidths(DummyOptionsGroup)");
  TestHelpers::db::test_simple_tag<Tags::Weight<DummyOptionsGroup>>(
      "Weight(DummyOptionsGroup)");
  TestHelpers::db::test_simple_tag<
      Tags::SummedIntrudingOverlapWeights<DummyOptionsGroup>>(
      "SummedIntrudingOverlapWeights(DummyOptionsGroup)");

  {
    TestHelpers::db::test_simple_tag<
        Tags::Overlaps<ScalarFieldTag<0>, 1, DummyOptionsGroup>>(
        "Overlaps(ScalarFieldTag, DummyOptionsGroup)");
    // Test subitems on overlaps
    const OverlapId<1> overlap_id{Direction<1>::lower_xi(), ElementId<1>{0}};
    auto box = db::create<db::AddSimpleTags<Tags::Overlaps<
        ::Tags::Variables<tmpl::list<ScalarFieldTag<0>, ScalarFieldTag<1>>>, 1,
        DummyOptionsGroup>>>(
        OverlapMap<1,
                   Variables<tmpl::list<ScalarFieldTag<0>, ScalarFieldTag<1>>>>{
            {overlap_id, {3, 0.}}});
    // Test retrieving the individual tags
    CHECK(get(get<Tags::Overlaps<ScalarFieldTag<0>, 1, DummyOptionsGroup>>(box)
                  .at(overlap_id)) == DataVector(3, 0.));
    CHECK(get(get<Tags::Overlaps<ScalarFieldTag<1>, 1, DummyOptionsGroup>>(box)
                  .at(overlap_id)) == DataVector(3, 0.));
    // Test mutating the individual tags
    db::mutate<Tags::Overlaps<ScalarFieldTag<0>, 1, DummyOptionsGroup>>(
        make_not_null(&box), [&overlap_id](const auto scalar0_overlaps) {
          get(scalar0_overlaps->at(overlap_id)) = 1.;
        });
    CHECK(get(get<Tags::Overlaps<ScalarFieldTag<0>, 1, DummyOptionsGroup>>(box)
                  .at(overlap_id)) == DataVector(3, 1.));
    CHECK(
        get(get<ScalarFieldTag<0>>(
            get<Tags::Overlaps<::Tags::Variables<tmpl::list<ScalarFieldTag<0>,
                                                            ScalarFieldTag<1>>>,
                               1, DummyOptionsGroup>>(box)
                .at(overlap_id))) == DataVector(3, 1.));
  }
}

}  // namespace LinearSolver::Schwarz
