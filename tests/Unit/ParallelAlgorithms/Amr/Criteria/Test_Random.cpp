// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Random.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t VolumeDim>
struct Metavariables {
  static constexpr size_t volume_dim = VolumeDim;
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<amr::Criterion, tmpl::list<amr::Criteria::Random>>>;
  };
};
struct TestComponent {};

template <size_t VolumeDim>
void test_criterion(const amr::Criterion& criterion) {
  Parallel::GlobalCache<Metavariables<VolumeDim>> empty_cache{};
  auto databox = db::create<db::AddSimpleTags<>>();
  auto empty_box =
      make_observation_box<db::AddComputeTags<>>(make_not_null(&databox));

  ElementId<VolumeDim> root_id{0};
  auto flags = criterion.evaluate(empty_box, empty_cache, root_id);
  for (size_t d = 0; d < VolumeDim; ++d) {
    CHECK((gsl::at(flags, d) == amr::Flag::Split or
           gsl::at(flags, d) == amr::Flag::DoNothing));
  }
}

template <size_t VolumeDim>
void test_always_split() {
  const amr::Criteria::Random criterion{{{amr::Flag::Split, 1}}};
  Parallel::GlobalCache<Metavariables<VolumeDim>> empty_cache{};
  auto databox = db::create<db::AddSimpleTags<>>();
  auto empty_box =
      make_observation_box<db::AddComputeTags<>>(make_not_null(&databox));

  ElementId<VolumeDim> root_id{0};
  auto flags = criterion.evaluate(empty_box, empty_cache, root_id);
  for (size_t d = 0; d < VolumeDim; ++d) {
    CHECK(gsl::at(flags, d) == amr::Flag::Split);
  }
}

template <size_t VolumeDim>
void test_always_do_nothing() {
  const amr::Criteria::Random criterion{{{amr::Flag::DoNothing, 1}}};

  Parallel::GlobalCache<Metavariables<VolumeDim>> empty_cache{};
  auto databox = db::create<db::AddSimpleTags<>>();
  auto empty_box =
      make_observation_box<db::AddComputeTags<>>(make_not_null(&databox));

  for (size_t level = 0; level <= 5; ++level) {
    ElementId<VolumeDim> id{0, make_array<VolumeDim>(SegmentId(level, 0))};
    auto flags = criterion.evaluate(empty_box, empty_cache, id);
    for (size_t d = 0; d < VolumeDim; ++d) {
      CHECK(gsl::at(flags, d) == amr::Flag::DoNothing);
    }
  }
}

template <size_t VolumeDim>
void test_h_or_p() {
  const amr::Criteria::Random criterion{{{amr::Flag::Split, 1},
                                         {amr::Flag::IncreaseResolution, 1},
                                         {amr::Flag::DecreaseResolution, 1},
                                         {amr::Flag::Join, 1}}};
  Parallel::GlobalCache<Metavariables<VolumeDim>> empty_cache{};
  auto databox = db::create<db::AddSimpleTags<>>();
  auto empty_box =
      make_observation_box<db::AddComputeTags<>>(make_not_null(&databox));

  ElementId<VolumeDim> root_id{0, make_array<VolumeDim>(SegmentId(1, 0))};
  auto flags = criterion.evaluate(empty_box, empty_cache, root_id);
  for (size_t d = 0; d < VolumeDim; ++d) {
    CHECK((gsl::at(flags, d) != amr::Flag::DoNothing and
           gsl::at(flags, d) != amr::Flag::Undefined));
  }
}

template <size_t VolumeDim>
void test() {
  register_factory_classes_with_charm<Metavariables<VolumeDim>>();
  test_always_split<VolumeDim>();
  test_always_do_nothing<VolumeDim>();
  test_h_or_p<VolumeDim>();
  const amr::Criteria::Random random_criterion{
      {{amr::Flag::Split, 4}, {amr::Flag::DoNothing, 1}}};
  test_criterion<VolumeDim>(random_criterion);
  test_criterion<VolumeDim>(serialize_and_deserialize(random_criterion));

  const auto criterion =
      TestHelpers::test_creation<std::unique_ptr<amr::Criterion>,
                                 Metavariables<VolumeDim>>(
          "Random:\n"
          "  ProbabilityWeights:\n"
          "    Split: 4\n"
          "    DoNothing: 1\n"
          "  MaximumRefinementLevel: 5\n");
  test_criterion<VolumeDim>(*criterion);
  test_criterion<VolumeDim>(*serialize_and_deserialize(criterion));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Criteria.Random", "[Unit][ParallelAlgorithms]") {
  TestHelpers::db::test_simple_tag<amr::Criteria::Tags::Criteria>("Criteria");
  test<1>();
  test<2>();
  test<3>();
}
