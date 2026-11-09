// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/RefineAtBoundary.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace amr::Criteria {

namespace {

// All valid instantiations for a given volume dimension
template <size_t Dim>
struct AllRefineAtBoundary;
template <>
struct AllRefineAtBoundary<1> {
  using type = tmpl::list<RefineAtBoundary<1, 0>>;
};
template <>
struct AllRefineAtBoundary<2> {
  using type = tmpl::list<RefineAtBoundary<2, 0>, RefineAtBoundary<2, 1>>;
};
template <>
struct AllRefineAtBoundary<3> {
  using type = tmpl::list<RefineAtBoundary<3, 0>, RefineAtBoundary<3, 1>,
                          RefineAtBoundary<3, 2>>;
};

template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<amr::Criterion, typename AllRefineAtBoundary<Dim>::type>>;
  };
};

template <size_t Dim, size_t DimToRefine>
void test_criterion(const std::string& option_name) {
  CAPTURE(Dim);
  CAPTURE(DimToRefine);
  const auto created =
      TestHelpers::test_creation<std::unique_ptr<amr::Criterion>,
                                 Metavariables<Dim>>(option_name);
  REQUIRE(dynamic_cast<const RefineAtBoundary<Dim, DimToRefine>*>(
              created.get()) != nullptr);
  const auto& criterion = serialize_and_deserialize(
      dynamic_cast<const RefineAtBoundary<Dim, DimToRefine>&>(*created));

  // Single-block domain with refinement level 2 in every dimension, i.e. 4
  // segments per dimension. In each dimension, segment indices 0 and 3 touch
  // the lower and upper external boundaries, while 1 and 2 are interior.
  const domain::creators::Rectilinear<Dim> domain_creator{
      make_array<Dim>(0.), make_array<Dim>(1.), make_array<Dim>(size_t{2}),
      make_array<Dim>(size_t{3})};
  const auto domain = domain_creator.create_domain();
  const auto refinement_levels = domain_creator.initial_refinement_levels();
  Parallel::GlobalCache<Metavariables<Dim>> empty_cache{};

  // Loop over all elements. An element should be split in DimToRefine if and
  // only if it touches an external boundary in DimToRefine. Touching an
  // external boundary in any other dimension must not trigger refinement.
  size_t num_split = 0;
  size_t num_do_nothing = 0;
  for (const auto& element_id : initial_element_ids(refinement_levels)) {
    CAPTURE(element_id);
    auto element = domain::create_initial_element(element_id, domain.blocks(),
                                                  refinement_levels);
    auto databox =
        db::create<tmpl::list<domain::Tags::Element<Dim>>>(std::move(element));
    const ObservationBox<tmpl::list<>, decltype(databox)> box{
        make_not_null(&databox)};

    const size_t index = element_id.segment_id(DimToRefine).index();
    auto expected = make_array<Dim>(amr::Flag::DoNothing);
    if (index == 0 or index == 3) {
      gsl::at(expected, DimToRefine) = amr::Flag::Split;
      ++num_split;
    } else {
      ++num_do_nothing;
    }
    CHECK(criterion.evaluate(box, empty_cache, element_id) == expected);
  }
  // Make sure both branches were exercised
  CHECK(num_split > 0);
  CHECK(num_do_nothing > 0);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Criteria.RefineAtBoundary",
                  "[Unit][ParallelAlgorithms]") {
  register_factory_classes_with_charm<Metavariables<1>>();
  register_factory_classes_with_charm<Metavariables<2>>();
  register_factory_classes_with_charm<Metavariables<3>>();

  test_criterion<1, 0>("RefineAtBoundaryX");
  test_criterion<2, 0>("RefineAtBoundaryX");
  test_criterion<2, 1>("RefineAtBoundaryY");
  test_criterion<3, 0>("RefineAtBoundaryX");
  test_criterion<3, 1>("RefineAtBoundaryY");
  test_criterion<3, 2>("RefineAtBoundaryZ");
}

}  // namespace amr::Criteria
