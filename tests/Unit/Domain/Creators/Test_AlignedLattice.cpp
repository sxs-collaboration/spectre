// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/AlignedLattice.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Domain.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace Frame {
struct Inertial;
}  // namespace Frame

namespace domain {
namespace {
template <size_t VolumeDim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition() {
  return std::make_unique<TestHelpers::domain::BoundaryConditions::
                              TestBoundaryCondition<VolumeDim>>(
      Direction<VolumeDim>::upper_xi(), 100);
}

template <size_t VolumeDim>
auto make_domain_creator(const std::string& opt_string,
                         const bool use_boundary_condition) {
  if (use_boundary_condition) {
    return TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<VolumeDim>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithBoundaryConditions<
                VolumeDim, domain::creators::AlignedLattice<VolumeDim>>>(
        opt_string + std::string{"  BoundaryCondition:\n"
                                 "    TestBoundaryCondition:\n"
                                 "      Direction: upper-xi\n"
                                 "      BlockId: 100\n"});
  } else {
    return TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<VolumeDim>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithoutBoundaryConditions<
                VolumeDim, domain::creators::AlignedLattice<VolumeDim>>>(
        opt_string);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.AlignedLattice", "[Domain][Unit]") {
  domain::creators::register_derived_with_charm();
  TestHelpers::domain::BoundaryConditions::register_derived_with_charm();

  for (const bool use_boundary_condition : {true, false}) {
    CAPTURE(use_boundary_condition);
    const auto domain_creator_1d = make_domain_creator<1>(
        "AlignedLattice:\n"
        "  BlockBounds: [[0.1, 2.6, 5.1, 5.2, 7.2]]\n" +
            std::string{use_boundary_condition ? ""
                                               : "  IsPeriodicIn: [false]\n"} +
            "  InitialGridPoints: [3]\n"
            "  InitialLevels: [2]\n"
            "  RefinedLevels: []\n"
            "  RefinedGridPoints: []\n"
            "  BlocksToExclude: []\n",
        use_boundary_condition);
    const auto* aligned_blocks_creator_1d =
        dynamic_cast<const creators::AlignedLattice<1>*>(
            domain_creator_1d.get());
    TestHelpers::domain::creators::test_domain_creator(
        *aligned_blocks_creator_1d, use_boundary_condition);

    const auto domain_creator_2d = make_domain_creator<2>(
        "AlignedLattice:\n"
        "  BlockBounds: [[0.1, 2.6, 5.1], [-0.4, 3.2, 6.2, 8.9]]\n" +
            std::string{use_boundary_condition
                            ? ""
                            : "  IsPeriodicIn: [false, false]\n"} +
            "  InitialGridPoints: [3, 4]\n"
            "  InitialLevels: [2, 1]\n"
            "  RefinedLevels: []\n"
            "  RefinedGridPoints: []\n"
            "  BlocksToExclude: []\n",
        use_boundary_condition);
    const auto* aligned_blocks_creator_2d =
        dynamic_cast<const creators::AlignedLattice<2>*>(
            domain_creator_2d.get());
    TestHelpers::domain::creators::test_domain_creator(
        *aligned_blocks_creator_2d, use_boundary_condition);

    const auto domain_creator_3d = make_domain_creator<3>(
        "AlignedLattice:\n"
        "  BlockBounds: [[0.1, 2.6, 5.1], [-0.4, 3.2, 6.2], [-0.2, 3.2]]\n" +
            std::string{use_boundary_condition
                            ? ""
                            : "  IsPeriodicIn: [false, false, false]\n"} +
            "  InitialGridPoints: [3, 4, 5]\n"
            "  InitialLevels: [2, 1, 0]\n"
            "  RefinedLevels: []\n"
            "  RefinedGridPoints: []\n"
            "  BlocksToExclude: []\n",
        use_boundary_condition);
    const auto* aligned_blocks_creator_3d =
        dynamic_cast<const creators::AlignedLattice<3>*>(
            domain_creator_3d.get());
    TestHelpers::domain::creators::test_domain_creator(
        *aligned_blocks_creator_3d, use_boundary_condition);

    const auto cubical_shell_domain = make_domain_creator<3>(
        "AlignedLattice:\n"
        "  BlockBounds: [[0.1, 2.6, 5.1, 6.0], [-0.4, 3.2, 6.2, 7.0], "
        "[-0.2, 3.2, 4.0, 5.2]]\n" +
            std::string{use_boundary_condition
                            ? ""
                            : "  IsPeriodicIn: [false, false, false]\n"} +
            "  InitialGridPoints: [3, 4, 5]\n"
            "  InitialLevels: [2, 1, 0]\n"
            "  RefinedLevels: []\n"
            "  RefinedGridPoints: []\n"
            "  BlocksToExclude: [[1, 1, 1]]\n",
        use_boundary_condition);
    const auto* cubical_shell_creator_3d =
        dynamic_cast<const creators::AlignedLattice<3>*>(
            cubical_shell_domain.get());
    TestHelpers::domain::creators::test_domain_creator(
        *cubical_shell_creator_3d, use_boundary_condition);

    const auto unit_cubical_shell_domain = make_domain_creator<3>(
        "AlignedLattice:\n"
        "  BlockBounds: [[-1.5, -0.5, 0.5, 1.5], [-1.5, -0.5, 0.5, 1.5], "
        "[-1.5, -0.5, 0.5, 1.5]]\n" +
            std::string{use_boundary_condition
                            ? ""
                            : "  IsPeriodicIn: [false, false, false]\n"} +
            "  InitialGridPoints: [5, 5, 5]\n"
            "  InitialLevels: [1, 1, 1]\n"
            "  RefinedLevels: []\n"
            "  RefinedGridPoints: []\n"
            "  BlocksToExclude: [[1, 1, 1]]\n",
        use_boundary_condition);
    const auto* unit_cubical_shell_creator_3d =
        dynamic_cast<const creators::AlignedLattice<3>*>(
            unit_cubical_shell_domain.get());
    TestHelpers::domain::creators::test_domain_creator(
        *unit_cubical_shell_creator_3d, use_boundary_condition);
  }

  const auto domain_creator_2d_periodic = TestHelpers::test_option_tag<
      domain::OptionTags::DomainCreator<2>,
      TestHelpers::domain::BoundaryConditions::
          MetavariablesWithoutBoundaryConditions<
              2, domain::creators::AlignedLattice<2>>>(
      "AlignedLattice:\n"
      "  BlockBounds: [[0.1, 2.6, 5.1], [-0.4, 3.2, 6.2, 8.9]]\n"
      "  IsPeriodicIn: [false, true]\n"
      "  InitialGridPoints: [3, 4]\n"
      "  InitialLevels: [2, 1]\n"
      "  RefinedLevels: []\n"
      "  RefinedGridPoints: []\n"
      "  BlocksToExclude: []\n");
  const auto* aligned_blocks_creator_2d_periodic =
      dynamic_cast<const creators::AlignedLattice<2>*>(
          domain_creator_2d_periodic.get());
  TestHelpers::domain::creators::test_domain_creator(
      *aligned_blocks_creator_2d_periodic, false, true);

  const auto domain_creator_3d_periodic = TestHelpers::test_option_tag<
      domain::OptionTags::DomainCreator<3>,
      TestHelpers::domain::BoundaryConditions::
          MetavariablesWithoutBoundaryConditions<
              3, domain::creators::AlignedLattice<3>>>(
      "AlignedLattice:\n"
      "  BlockBounds: [[0.1, 2.6, 5.1], [-0.4, 3.2, 6.2], [-0.2, 3.2]]\n"
      "  IsPeriodicIn: [false, true, false]\n"
      "  InitialGridPoints: [3, 4, 5]\n"
      "  InitialLevels: [2, 1, 0]\n"
      "  RefinedLevels: []\n"
      "  RefinedGridPoints: []\n"
      "  BlocksToExclude: []\n");
  const auto* aligned_blocks_creator_3d_periodic =
      dynamic_cast<const creators::AlignedLattice<3>*>(
          domain_creator_3d_periodic.get());
  TestHelpers::domain::creators::test_domain_creator(
      *aligned_blocks_creator_3d_periodic, false, true);

  {
    // Expected domain refinement:
    // 23 23 67
    // 23 45 67
    // 23 XX 45
    const auto refined_domain = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<2>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithoutBoundaryConditions<
                2, domain::creators::AlignedLattice<2>>>(
        "AlignedLattice:\n"
        "  BlockBounds: [[70, 71, 72, 73], [90, 91, 92, 93]]\n"
        "  IsPeriodicIn: [false, false]\n"
        "  InitialGridPoints: [2, 3]\n"
        "  InitialLevels: [0, 0]\n"
        "  BlocksToExclude: [[1, 0]]\n"
        "  RefinedLevels: []\n"
        "  RefinedGridPoints:\n"
        "  - LowerCornerIndex: [1, 0]\n"
        "    UpperCornerIndex: [3, 2]\n"
        "    Refinement: [4, 5]\n"
        "  - LowerCornerIndex: [2, 1]\n"
        "    UpperCornerIndex: [3, 3]\n"
        "    Refinement: [6, 7]");
    std::unordered_set<
        std::pair<std::vector<double>, std::array<size_t, 2>>,
        boost::hash<std::pair<std::vector<double>, std::array<size_t, 2>>>>
        expected_blocks{{{70.0, 90.0}, {{2, 3}}}, {{72.0, 90.0}, {{4, 5}}},
                        {{70.0, 91.0}, {{2, 3}}}, {{71.0, 91.0}, {{4, 5}}},
                        {{72.0, 91.0}, {{6, 7}}}, {{70.0, 92.0}, {{2, 3}}},
                        {{71.0, 92.0}, {{2, 3}}}, {{72.0, 92.0}, {{6, 7}}}};
    const auto domain = TestHelpers::domain::creators::test_domain_creator(
        *refined_domain, false);

    const auto& blocks = domain.blocks();
    const auto extents = refined_domain->initial_extents();
    REQUIRE(blocks.size() == extents.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
      const auto location =
          blocks[i]
              .stationary_map()(
                  tnsr::I<double, 2, Frame::BlockLogical>{{{-1.0, -1.0}}})
              .get_vector_of_data()
              .second;
      INFO("Unexpected block");
      CAPTURE(location);
      CAPTURE(extents[i]);
      CHECK(expected_blocks.erase({location, extents[i]}) == 1);
    }
    CAPTURE(expected_blocks);
    CHECK(expected_blocks.empty());
  }

  {
    // Expected domain refinement:
    // 25 25 46
    // 25 35 46
    // 25 XX 35
    const auto refined_domain = TestHelpers::test_option_tag<
        domain::OptionTags::DomainCreator<2>,
        TestHelpers::domain::BoundaryConditions::
            MetavariablesWithoutBoundaryConditions<
                2, domain::creators::AlignedLattice<2>>>(
        "AlignedLattice:\n"
        "  BlockBounds: [[70, 71, 72, 73], [90, 91, 92, 93]]\n"
        "  IsPeriodicIn: [false, false]\n"
        "  InitialGridPoints: [10, 10]\n"
        "  InitialLevels: [2, 5]\n"
        "  BlocksToExclude: [[1, 0]]\n"
        "  RefinedGridPoints: []\n"
        "  RefinedLevels:\n"
        "  - LowerCornerIndex: [1, 0]\n"
        "    UpperCornerIndex: [3, 2]\n"
        "    Refinement: [3, 5]\n"
        "  - LowerCornerIndex: [2, 1]\n"
        "    UpperCornerIndex: [3, 3]\n"
        "    Refinement: [4, 6]");
    std::unordered_set<
        std::pair<std::vector<double>, std::array<size_t, 2>>,
        boost::hash<std::pair<std::vector<double>, std::array<size_t, 2>>>>
        expected_blocks{{{70.0, 90.0}, {{2, 5}}}, {{72.0, 90.0}, {{3, 5}}},
                        {{70.0, 91.0}, {{2, 5}}}, {{71.0, 91.0}, {{3, 5}}},
                        {{72.0, 91.0}, {{4, 6}}}, {{70.0, 92.0}, {{2, 5}}},
                        {{71.0, 92.0}, {{2, 5}}}, {{72.0, 92.0}, {{4, 6}}}};
    const auto domain = TestHelpers::domain::creators::test_domain_creator(
        *refined_domain, false);
    const auto refinement_levels = refined_domain->initial_refinement_levels();

    const auto& blocks = domain.blocks();
    REQUIRE(blocks.size() == refinement_levels.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
      const auto location =
          blocks[i]
              .stationary_map()(
                  tnsr::I<double, 2, Frame::BlockLogical>{{{-1.0, -1.0}}})
              .get_vector_of_data()
              .second;
      INFO("Unexpected block");
      CAPTURE(location);
      CAPTURE(refinement_levels[i]);
      CHECK(expected_blocks.erase({location, refinement_levels[i]}) == 1);
    }
    CAPTURE(expected_blocks);
    CHECK(expected_blocks.empty());
  }

  CHECK_THROWS_WITH(
      creators::AlignedLattice<3>({{{{-1.5, -0.5, 0.5, 1.5}},
                                    {{1.5, -0.5, 0.5, 1.5}},
                                    {{-1.5, -0.5, 0.5, 1.5}}}},
                                  {{1, 1, 1}}, {{5, 5, 5}}, {}, {},
                                  {{{{1, 1, 1}}}}, {{true, false, false}},
                                  Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Cannot exclude blocks as well as have periodic boundary"));
  CHECK_THROWS_WITH(
      creators::AlignedLattice<3>({{{{-1.5, -0.5, 0.5, 1.5}},
                                    {{1.5, -0.5, 0.5, 1.5}},
                                    {{-1.5, -0.5, 0.5, 1.5}}}},
                                  {{1, 1, 1}}, {{5, 5, 5}}, {}, {},
                                  {{{{1, 1, 1}}}}, {{false, true, false}},
                                  Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Cannot exclude blocks as well as have periodic boundary"));
  CHECK_THROWS_WITH(
      creators::AlignedLattice<3>({{{{-1.5, -0.5, 0.5, 1.5}},
                                    {{1.5, -0.5, 0.5, 1.5}},
                                    {{-1.5, -0.5, 0.5, 1.5}}}},
                                  {{1, 1, 1}}, {{5, 5, 5}}, {}, {},
                                  {{{{1, 1, 1}}}}, {{true, false, true}},
                                  Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Cannot exclude blocks as well as have periodic boundary"));
  CHECK_THROWS_WITH(
      creators::AlignedLattice<3>(
          {{{{-1.5, -0.5, 0.5, 1.5}},
            {{1.5, -0.5, 0.5, 1.5}},
            {{-1.5, -0.5, 0.5, 1.5}}}},
          {{1, 1, 1}}, {{5, 5, 5}}, {}, {}, {{{{1, 1, 1}}}},
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Cannot exclude blocks as well as have periodic boundary"));
  CHECK_THROWS_WITH(
      creators::AlignedLattice<3>(
          {{{{-1.5, -0.5, 0.5, 1.5}},
            {{1.5, -0.5, 0.5, 1.5}},
            {{-1.5, -0.5, 0.5, 1.5}}}},
          {{1, 1, 1}}, {{5, 5, 5}}, {}, {}, {{{{1, 1, 1}}}},
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestNoneBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "None boundary condition is not supported. If you would like an "
          "outflow-type boundary condition, you must use that."));
}
}  // namespace domain
