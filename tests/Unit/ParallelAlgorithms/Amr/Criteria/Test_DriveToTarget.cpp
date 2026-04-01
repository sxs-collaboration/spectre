// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>

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
#include "ParallelAlgorithms/Amr/Criteria/DriveToTarget.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Type.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t VolumeDim>
struct Metavariables {
  static constexpr size_t volume_dim = VolumeDim;
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        amr::Criterion,
        tmpl::list<
            amr::Criteria::DriveToTarget<VolumeDim, amr::Criteria::Type::h>,
            amr::Criteria::DriveToTarget<VolumeDim, amr::Criteria::Type::p>>>>;
  };
};
struct TestComponent {};

template <size_t VolumeDim>
void test_criterion(
    const amr::Criterion& criterion,
    const std::array<size_t, VolumeDim> mesh_extents,
    const std::array<size_t, VolumeDim>& element_refinement_levels,
    const std::array<amr::Flag, VolumeDim>& expected_flags) {
  Parallel::GlobalCache<Metavariables<VolumeDim>> empty_cache{};
  auto databox = db::create<tmpl::list<::domain::Tags::Mesh<VolumeDim>>>(
      Mesh<VolumeDim>{mesh_extents, Spectral::Basis::Legendre,
                      Spectral::Quadrature::GaussLobatto});
  ObservationBox<tmpl::list<>,
                 db::DataBox<tmpl::list<::domain::Tags::Mesh<VolumeDim>>>>
      box{make_not_null(&databox)};

  std::array<SegmentId, VolumeDim> segment_ids;
  alg::transform(element_refinement_levels, segment_ids.begin(),
                 [](const size_t extent) {
                   return SegmentId{extent, 0_st};
                 });
  ElementId<VolumeDim> element_id{0, segment_ids};
  auto flags = criterion.evaluate(box, empty_cache, element_id);
  CHECK(flags == expected_flags);
}

template <amr::Criteria::Type CriteriaType, size_t VolumeDim>
void test(
    const std::array<size_t, VolumeDim>& target,
    const std::array<amr::Flag, VolumeDim>& flags_at_target,
    const std::vector<
        std::tuple<std::array<size_t, VolumeDim>, std::array<size_t, VolumeDim>,
                   std::array<amr::Flag, VolumeDim>>>& test_cases,
    const std::string& option_string) {
  register_factory_classes_with_charm<Metavariables<VolumeDim>>();

  const amr::Criteria::DriveToTarget<VolumeDim, CriteriaType> criterion{
      target, flags_at_target};
  const auto criterion_from_option_string =
      TestHelpers::test_creation<std::unique_ptr<amr::Criterion>,
                                 Metavariables<VolumeDim>>(option_string);
  for (const auto& [extents, levels, expected_flags] : test_cases) {
    test_criterion(criterion, extents, levels, expected_flags);
    test_criterion(*criterion_from_option_string, extents, levels,
                   expected_flags);
    test_criterion(*serialize_and_deserialize(criterion_from_option_string),
                   extents, levels, expected_flags);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Criteria.DriveToTarget",
                  "[Unit][ParallelAlgorithms]") {
  {
    INFO("1D h");
    const std::array target_levels{3_st};
    const std::array flags_at_target{amr::Flag::Join};
    const auto test_cases =
        std::vector{std::tuple{std::array{4_st}, std::array{3_st},
                               std::array{amr::Flag::Join}},
                    std::tuple{std::array{3_st}, std::array{3_st},
                               std::array{amr::Flag::Join}},
                    std::tuple{std::array{5_st}, std::array{3_st},
                               std::array{amr::Flag::Join}},
                    std::tuple{std::array{4_st}, std::array{2_st},
                               std::array{amr::Flag::Split}},
                    std::tuple{std::array{3_st}, std::array{2_st},
                               std::array{amr::Flag::Split}},
                    std::tuple{std::array{5_st}, std::array{2_st},
                               std::array{amr::Flag::Split}},
                    std::tuple{std::array{4_st}, std::array{4_st},
                               std::array{amr::Flag::Join}},
                    std::tuple{std::array{3_st}, std::array{4_st},
                               std::array{amr::Flag::Join}},
                    std::tuple{std::array{5_st}, std::array{4_st},
                               std::array{amr::Flag::Join}}};
    const std::string option =
        "DriveToTargetRefinementLevels:\n"
        "  Target: [3]\n"
        "  OscillationAtTarget: [Join]\n";
    test<amr::Criteria::Type::h>(target_levels, flags_at_target, test_cases,
                                 option);
    CHECK_THROWS_WITH(
        (test<amr::Criteria::Type::h>(target_levels,
                                      std::array{amr::Flag::IncreaseResolution},
                                      test_cases, option)),
        Catch::Matchers::ContainsSubstring(
            "Cannot use p-refinement flag in OscillationAtTarget"));
  }
  {
    INFO("1D p");
    const std::array target_extents{4_st};
    const std::array flags_at_target{amr::Flag::DecreaseResolution};
    const auto test_cases =
        std::vector{std::tuple{std::array{4_st}, std::array{3_st},
                               std::array{amr::Flag::DecreaseResolution}},
                    std::tuple{std::array{3_st}, std::array{3_st},
                               std::array{amr::Flag::IncreaseResolution}},
                    std::tuple{std::array{5_st}, std::array{3_st},
                               std::array{amr::Flag::DecreaseResolution}},
                    std::tuple{std::array{4_st}, std::array{2_st},
                               std::array{amr::Flag::DecreaseResolution}},
                    std::tuple{std::array{3_st}, std::array{2_st},
                               std::array{amr::Flag::IncreaseResolution}},
                    std::tuple{std::array{5_st}, std::array{2_st},
                               std::array{amr::Flag::DecreaseResolution}},
                    std::tuple{std::array{4_st}, std::array{4_st},
                               std::array{amr::Flag::DecreaseResolution}},
                    std::tuple{std::array{3_st}, std::array{4_st},
                               std::array{amr::Flag::IncreaseResolution}},
                    std::tuple{std::array{5_st}, std::array{4_st},
                               std::array{amr::Flag::DecreaseResolution}}};
    const std::string option =
        "DriveToTargetNumberOfGridPoints:\n"
        "  Target: [4]\n"
        "  OscillationAtTarget: [DecreaseResolution]\n";
    test<amr::Criteria::Type::p>(target_extents, flags_at_target, test_cases,
                                 option);
    CHECK_THROWS_WITH(
        (test<amr::Criteria::Type::p>(
            target_extents, std::array{amr::Flag::Join}, test_cases, option)),
        Catch::Matchers::ContainsSubstring(
            "Cannot use h-refinement flag in OscillationAtTarget"));
  }
  {
    INFO("2D h");
    const std::array target_levels{8_st, 3_st};
    const std::array flags_at_target{amr::Flag::Join, amr::Flag::Split};
    const auto test_cases = std::vector{
        std::tuple{std::array{4_st, 6_st}, std::array{8_st, 3_st},
                   std::array{amr::Flag::Join, amr::Flag::Split}},
        std::tuple{std::array{5_st, 6_st}, std::array{8_st, 3_st},
                   std::array{amr::Flag::Join, amr::Flag::Split}},
        std::tuple{std::array{3_st, 6_st}, std::array{7_st, 4_st},
                   std::array{amr::Flag::Split, amr::Flag::Join}},
        std::tuple{std::array{4_st, 6_st}, std::array{8_st, 2_st},
                   std::array{amr::Flag::DoNothing, amr::Flag::Split}}};
    const std::string option =
        "DriveToTargetRefinementLevels:\n"
        "  Target: [8, 3]\n"
        "  OscillationAtTarget: [Join, Split]\n";
    test<amr::Criteria::Type::h>(target_levels, flags_at_target, test_cases,
                                 option);
    CHECK_THROWS_WITH(
        (test<amr::Criteria::Type::h>(
            target_levels,
            std::array{amr::Flag::Split, amr::Flag::IncreaseResolution},
            test_cases, option)),
        Catch::Matchers::ContainsSubstring(
            "Cannot use p-refinement flag in OscillationAtTarget"));
  }
  {
    INFO("2D p");
    const std::array target_extents{4_st, 6_st};
    const std::array flags_at_target{amr::Flag::IncreaseResolution,
                                     amr::Flag::DecreaseResolution};
    const auto test_cases = std::vector{
        std::tuple{std::array{4_st, 6_st}, std::array{8_st, 3_st},
                   std::array{amr::Flag::IncreaseResolution,
                              amr::Flag::DecreaseResolution}},
        std::tuple{
            std::array{5_st, 6_st}, std::array{8_st, 3_st},
            std::array{amr::Flag::DecreaseResolution, amr::Flag::DoNothing}},
        std::tuple{
            std::array{3_st, 6_st}, std::array{7_st, 4_st},
            std::array{amr::Flag::IncreaseResolution, amr::Flag::DoNothing}},
        std::tuple{
            std::array{4_st, 5_st}, std::array{8_st, 2_st},
            std::array{amr::Flag::DoNothing, amr::Flag::IncreaseResolution}}};
    const std::string option =
        "DriveToTargetNumberOfGridPoints:\n"
        "  Target: [4, 6]\n"
        "  OscillationAtTarget: [IncreaseResolution, DecreaseResolution]\n";
    test<amr::Criteria::Type::p>(target_extents, flags_at_target, test_cases,
                                 option);
    CHECK_THROWS_WITH(
        (test<amr::Criteria::Type::p>(
            target_extents,
            std::array{amr::Flag::Join, amr::Flag::DecreaseResolution},
            test_cases, option)),
        Catch::Matchers::ContainsSubstring(
            "Cannot use h-refinement flag in OscillationAtTarget"));
  }
  {
    INFO("3D h");
    const std::array target_levels{5_st, 2_st, 4_st};
    const std::array flags_at_target{amr::Flag::Split, amr::Flag::Join,
                                     amr::Flag::DoNothing};
    const auto test_cases = std::vector{
        std::tuple{std::array{3_st, 9_st, 5_st}, std::array{5_st, 2_st, 4_st},
                   std::array{amr::Flag::Split, amr::Flag::Join,
                              amr::Flag::DoNothing}},
        std::tuple{std::array{3_st, 9_st, 5_st}, std::array{5_st, 5_st, 4_st},
                   std::array{amr::Flag::DoNothing, amr::Flag::Join,
                              amr::Flag::DoNothing}},
        std::tuple{std::array{3_st, 9_st, 3_st}, std::array{5_st, 2_st, 5_st},
                   std::array{amr::Flag::DoNothing, amr::Flag::DoNothing,
                              amr::Flag::Join}}};
    const std::string option =
        "DriveToTargetRefinementLevels:\n"
        "  Target: [5, 2, 4]\n"
        "  OscillationAtTarget: [Split, Join, DoNothing]\n";
    test<amr::Criteria::Type::h>(target_levels, flags_at_target, test_cases,
                                 option);
    CHECK_THROWS_WITH(
        (test<amr::Criteria::Type::h>(
            target_levels,
            std::array{amr::Flag::Split, amr::Flag::IncreaseResolution,
                       amr::Flag::DecreaseResolution},
            test_cases, option)),
        Catch::Matchers::ContainsSubstring(
            "Cannot use p-refinement flag in OscillationAtTarget"));
  }
  {
    INFO("3D p");
    const std::array target_extents{3_st, 9_st, 5_st};
    const std::array flags_at_target{amr::Flag::IncreaseResolution,
                                     amr::Flag::DecreaseResolution,
                                     amr::Flag::DoNothing};
    const auto test_cases = std::vector{
        std::tuple{
            std::array{3_st, 9_st, 5_st}, std::array{5_st, 2_st, 4_st},
            std::array{amr::Flag::IncreaseResolution,
                       amr::Flag::DecreaseResolution, amr::Flag::DoNothing}},
        std::tuple{std::array{4_st, 8_st, 2_st}, std::array{5_st, 5_st, 4_st},
                   std::array{amr::Flag::DecreaseResolution,
                              amr::Flag::IncreaseResolution,
                              amr::Flag::IncreaseResolution}},
        std::tuple{std::array{3_st, 9_st, 3_st}, std::array{5_st, 2_st, 4_st},
                   std::array{amr::Flag::DoNothing, amr::Flag::DoNothing,
                              amr::Flag::IncreaseResolution}}};
    const std::string option =
        "DriveToTargetNumberOfGridPoints:\n"
        "  Target: [3, 9, 5]\n"
        "  OscillationAtTarget: [IncreaseResolution, DecreaseResolution, "
        "DoNothing]\n";
    test<amr::Criteria::Type::p>(target_extents, flags_at_target, test_cases,
                                 option);
    CHECK_THROWS_WITH(
        (test<amr::Criteria::Type::p>(
            target_extents,
            std::array{amr::Flag::Split, amr::Flag::Join,
                       amr::Flag::DecreaseResolution},
            test_cases, option)),
        Catch::Matchers::ContainsSubstring(
            "Cannot use h-refinement flag in OscillationAtTarget"));
  }
}
