// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <pup.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Domain/Block.hpp"                       // IWYU pragma: keep
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/CylindricalBinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/OptionTags.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"

namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime

namespace {
using Translation = domain::CoordinateMaps::TimeDependent::Translation;
using Translation3D = domain::CoordinateMaps::TimeDependent::ProductOf3Maps<
    Translation, Translation, Translation>;
using BoundaryCondVector = std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>;

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_inner_boundary_condition() {
  // TestHelpers::domain::BoundaryConditions::TestBoundaryCondition
  // takes a direction and a block_id in its constructor.  In this
  // test, these parameters are not used for anything (other than when
  // comparing different BoundaryCondition objects using operator==)
  // and they have no relationship with any actual boundary directions
  // or block_ids in the Domain.  So we fill them with arbitrarily
  // chosen values, making sure that the parameters are chosen
  // differently for the inner and outer boundary conditions.
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_xi(), 463);
}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_outer_boundary_condition() {
  // TestHelpers::domain::BoundaryConditions::TestBoundaryCondition
  // takes a direction and a block_id in its constructor.  In this
  // test, these parameters are not used for anything (other than when
  // comparing different BoundaryCondition objects using operator==)
  // and they have no relationship with any actual boundary directions
  // or block_ids in the Domain.  So we fill them with arbitrarily
  // chosen values, making sure that the parameters are chosen
  // differently for the inner and outer boundary conditions.
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_eta(), 314);
}

auto create_boundary_conditions(const bool include_sphere_E) {
  size_t total_blocks = 46;
  if (include_sphere_E) {
    total_blocks += 13;
  }

  BoundaryCondVector boundary_conditions_all_blocks{total_blocks};

  // CA Filled Cylinder
  for (size_t block_id = 0; block_id < 5; ++block_id) {
    boundary_conditions_all_blocks[block_id][Direction<3>::upper_zeta()] =
        create_outer_boundary_condition();
  }

  // CA Cylinder
  for (size_t block_id = 5; block_id < 9; ++block_id) {
    boundary_conditions_all_blocks[block_id][Direction<3>::upper_xi()] =
        create_outer_boundary_condition();
  }

  // EA Filled Cylinder
  for (size_t block_id = 9; block_id < 14; ++block_id) {
    boundary_conditions_all_blocks[block_id][Direction<3>::lower_zeta()] =
        create_inner_boundary_condition();
  }

  // EA Cylinder
  for (size_t block_id = 14; block_id < 18; ++block_id) {
    boundary_conditions_all_blocks[block_id][Direction<3>::upper_xi()] =
        create_inner_boundary_condition();
  }

  // EB Filled Cylinder
  for (size_t block_id = 18; block_id < 23; ++block_id) {
    boundary_conditions_all_blocks[block_id][Direction<3>::lower_zeta()] =
        create_inner_boundary_condition();
  }

  // EB Cylinder
  for (size_t block_id = 23; block_id < 27; ++block_id) {
    boundary_conditions_all_blocks[block_id][Direction<3>::upper_xi()] =
        create_inner_boundary_condition();
  }

  // MA Filled Cylinder
  for (size_t block_id = 27; block_id < 32; ++block_id) {
    boundary_conditions_all_blocks[block_id][Direction<3>::upper_zeta()] =
        create_inner_boundary_condition();
  }

  // MB Filled Cylinder
  for (size_t block_id = 32; block_id < 37; ++block_id) {
    boundary_conditions_all_blocks[block_id][Direction<3>::upper_zeta()] =
        create_inner_boundary_condition();
  }

  // CB Filled Cylinder
  for (size_t block_id = 37; block_id < 42; ++block_id) {
    boundary_conditions_all_blocks[block_id][Direction<3>::upper_zeta()] =
        create_outer_boundary_condition();
  }

  // CB Cylinder
  for (size_t block_id = 42; block_id < 46; ++block_id) {
    boundary_conditions_all_blocks[block_id][Direction<3>::upper_xi()] =
        create_outer_boundary_condition();
  }

  return boundary_conditions_all_blocks;
}

template <typename... FuncsOfTime>
void test_binary_compact_object_construction(
    const domain::creators::CylindricalBinaryCompactObject&
        binary_compact_object,
    double time = std::numeric_limits<double>::signaling_NaN(),
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time = {},
    const std::tuple<std::pair<std::string, FuncsOfTime>...>&
        expected_functions_of_time = {},
    const BoundaryCondVector& expected_external_boundary_conditions = {}) {
  const auto domain = binary_compact_object.create_domain();
  test_initial_domain(domain,
                      binary_compact_object.initial_refinement_levels());
  test_det_jac_positive(domain.blocks(), time, functions_of_time);
  test_physical_separation(domain.blocks(), time, functions_of_time);

  for (size_t block_id = 0;
       block_id < expected_external_boundary_conditions.size(); ++block_id) {
    CAPTURE(block_id);
    const auto& block = domain.blocks()[block_id];
    REQUIRE(block.external_boundaries().size() ==
            expected_external_boundary_conditions[block_id].size());
    for (const auto& [direction, expected_bc_ptr] :
         expected_external_boundary_conditions[block_id]) {
      CAPTURE(direction);
      REQUIRE(block.external_boundary_conditions().count(direction) == 1);
      REQUIRE(block.external_boundary_conditions().at(direction) != nullptr);
      const auto& bc =
          dynamic_cast<const TestHelpers::domain::BoundaryConditions::
                           TestBoundaryCondition<3>&>(
              *block.external_boundary_conditions().at(direction));
      const auto& expected_bc =
          dynamic_cast<const TestHelpers::domain::BoundaryConditions::
                           TestBoundaryCondition<3>&>(*expected_bc_ptr);
      CHECK(bc.direction() == expected_bc.direction());
      CHECK(bc.block_id() == expected_bc.block_id());
    }
  }

  TestHelpers::domain::creators::test_functions_of_time(
      binary_compact_object, expected_functions_of_time);
}

void test_connectivity() {
  // ObjectA:
  constexpr double inner_radius_objectA = 1.0;

  // Misc.:
  constexpr double outer_radius = 25.0;
  constexpr size_t refinement = 1;
  constexpr size_t grid_points = 3;

  // When we add sphere_e support we will make the following
  // loop go over {true, false}
  for (const bool with_sphere_e : {false}) {
    CAPTURE(with_sphere_e);
    // ObjectA:
    const std::array<double, 3> center_objectA = {with_sphere_e ? 2.0 : 3.0,
                                                  0.05, 0.0};
    // ObjectB:
    const double inner_radius_objectB = with_sphere_e ? 0.4 : 1.0;
    const std::array<double, 3> center_objectB = {with_sphere_e ? -5.0 : -3.0,
                                                  0.05, 0.0};

    for (const bool with_boundary_conditions : {true, false}) {
      CAPTURE(with_boundary_conditions);
      const domain::creators::CylindricalBinaryCompactObject
          binary_compact_object{
              center_objectA,
              center_objectB,
              inner_radius_objectA,
              inner_radius_objectB,
              outer_radius,
              refinement,
              grid_points,
              nullptr,
              with_boundary_conditions ? create_inner_boundary_condition()
                                       : nullptr,
              with_boundary_conditions ? create_outer_boundary_condition()
                                       : nullptr};
      test_binary_compact_object_construction(
          binary_compact_object, std::numeric_limits<double>::signaling_NaN(),
          {}, {},
          with_boundary_conditions ? create_boundary_conditions(with_sphere_e)
                                   : BoundaryCondVector{});

      if (with_boundary_conditions) {
        CHECK_THROWS_WITH(
            domain::creators::CylindricalBinaryCompactObject(
                center_objectA, center_objectB, inner_radius_objectA,
                inner_radius_objectB, outer_radius, refinement, grid_points,
                nullptr, create_inner_boundary_condition(),
                std::make_unique<TestHelpers::domain::BoundaryConditions::
                                     TestPeriodicBoundaryCondition<3>>(),
                Options::Context{false, {}, 1, 1}),
            Catch::Matchers::Contains("Cannot have periodic boundary "
                                      "conditions with a binary domain"));
        CHECK_THROWS_WITH(
            domain::creators::CylindricalBinaryCompactObject(
                center_objectA, center_objectB, inner_radius_objectA,
                inner_radius_objectB, outer_radius, refinement, grid_points,
                nullptr,
                std::make_unique<TestHelpers::domain::BoundaryConditions::
                                     TestPeriodicBoundaryCondition<3>>(),
                create_outer_boundary_condition(),
                Options::Context{false, {}, 1, 1}),
            Catch::Matchers::Contains("Cannot have periodic boundary "
                                      "conditions with a binary domain"));
        CHECK_THROWS_WITH(
            domain::creators::CylindricalBinaryCompactObject(
                center_objectA, center_objectB, inner_radius_objectA,
                inner_radius_objectB, outer_radius, refinement, grid_points,
                nullptr, nullptr, create_outer_boundary_condition(),
                Options::Context{false, {}, 1, 1}),
            Catch::Matchers::Contains(
                "Must specify either both inner and outer boundary "
                "conditions or neither."));
        CHECK_THROWS_WITH(
            domain::creators::CylindricalBinaryCompactObject(
                center_objectA, center_objectB, inner_radius_objectA,
                inner_radius_objectB, outer_radius, refinement, grid_points,
                nullptr, create_inner_boundary_condition(), nullptr,
                Options::Context{false, {}, 1, 1}),
            Catch::Matchers::Contains(
                "Must specify either both inner and outer boundary "
                "conditions or neither."));
      }
    }
  }
}

std::string create_option_string(const bool add_time_dependence,
                                 const bool add_boundary_condition) {
  const std::string time_dependence{
      add_time_dependence
          ? "  TimeDependence:\n"
            "    UniformTranslation:\n"
            "      InitialTime: 1.0\n"
            "      InitialExpirationDeltaT: 9.0\n"
            "      Velocity: [2.3, -0.3, 0.5]\n"
            "      FunctionOfTimeNames: [TranslationX, TranslationY, "
            "TranslationZ]\n"
          : "  TimeDependence: None\n"};
  const std::string boundary_conditions{
      add_boundary_condition ? std::string{"  BoundaryConditions:\n"
                                           "    InnerBoundary:\n"
                                           "      TestBoundaryCondition:\n"
                                           "        Direction: upper-xi\n"
                                           "        BlockId: 463\n"
                                           "    OuterBoundary:\n"
                                           "      TestBoundaryCondition:\n"
                                           "        Direction: upper-eta\n"
                                           "        BlockId: 314\n"}
                             : ""};
  return "CylindricalBinaryCompactObject:\n"
         "  CenterA: [3.0, 0.05, 0.0]\n"
         "  RadiusA: 1.0\n"
         "  CenterB: [-3.0, 0.05, 0.0]\n"
         "  RadiusB: 1.0\n"
         "  OuterRadius: 25.0\n"
         "  InitialRefinement: 1\n"
         "  InitialGridPoints: 3\n" +
         time_dependence + boundary_conditions;
}

void test_bbh_time_dependent_factory(const bool with_boundary_conditions) {
  const auto binary_compact_object = [&with_boundary_conditions]() {
    if (with_boundary_conditions) {
      return TestHelpers::test_factory_creation<
          DomainCreator<3>, domain::OptionTags::DomainCreator<3>,
          TestHelpers::domain::BoundaryConditions::
              MetavariablesWithBoundaryConditions<3>>(
          create_option_string(true, with_boundary_conditions));
    } else {
      return TestHelpers::test_factory_creation<
          DomainCreator<3>, domain::OptionTags::DomainCreator<3>,
          TestHelpers::domain::BoundaryConditions::
              MetavariablesWithoutBoundaryConditions<3>>(
          create_option_string(true, with_boundary_conditions));
    }
  }();
  const std::array<double, 4> times_to_check{{0.0, 4.4, 7.8}};

  constexpr double initial_time = 0.0;
  constexpr double expiration_time = 10.0;
  constexpr double expected_time = 1.0; // matches InitialTime: 1.0 above
  constexpr double expected_update_delta_t =
      9.0;  // matches InitialExpirationDeltaT: 9.0 above
  std::array<DataVector, 3> function_of_time_coefficients_x{
      {{0.0}, {2.3}, {0.0}}};
  const std::array<DataVector, 3> function_of_time_coefficients_y{
      {{0.0}, {-0.3}, {0.0}}};
  const std::array<DataVector, 3> function_of_time_coefficients_z{
      {{0.0}, {0.5}, {0.0}}};

  const std::tuple<
      std::pair<std::string, domain::FunctionsOfTime::PiecewisePolynomial<2>>,
      std::pair<std::string, domain::FunctionsOfTime::PiecewisePolynomial<2>>,
      std::pair<std::string, domain::FunctionsOfTime::PiecewisePolynomial<2>>>
      expected_functions_of_time = std::make_tuple(
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<2>>{
              "TranslationX"s,
              {expected_time, function_of_time_coefficients_x,
               expected_time + expected_update_delta_t}},
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<2>>{
              "TranslationY"s,
              {expected_time, function_of_time_coefficients_y,
               expected_time + expected_update_delta_t}},
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<2>>{
              "TranslationZ"s,
              {expected_time, function_of_time_coefficients_z,
               expected_time + expected_update_delta_t}});
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time["TranslationX"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, function_of_time_coefficients_x, expiration_time);
  functions_of_time["TranslationY"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, function_of_time_coefficients_y, expiration_time);
  functions_of_time["TranslationZ"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, function_of_time_coefficients_z, expiration_time);

  for (const double time : times_to_check) {
    test_binary_compact_object_construction(
        dynamic_cast<const domain::creators::CylindricalBinaryCompactObject&>(
            *binary_compact_object),
        time, functions_of_time, expected_functions_of_time,
        with_boundary_conditions ? create_boundary_conditions(false)
                                 : BoundaryCondVector{});
  }
}

void test_binary_factory() {
  const auto check_impl = [](const std::string& opt_string,
                             const bool with_boundary_conditions) {
    const auto binary_compact_object = [&opt_string,
                                        &with_boundary_conditions]() {
      if (with_boundary_conditions) {
        return TestHelpers::test_factory_creation<
            DomainCreator<3>, domain::OptionTags::DomainCreator<3>,
            TestHelpers::domain::BoundaryConditions::
                MetavariablesWithBoundaryConditions<3>>(opt_string);
      } else {
        return TestHelpers::test_factory_creation<
            DomainCreator<3>, domain::OptionTags::DomainCreator<3>,
            TestHelpers::domain::BoundaryConditions::
                MetavariablesWithoutBoundaryConditions<3>>(opt_string);
      }
    }();
    test_binary_compact_object_construction(
        dynamic_cast<const domain::creators::CylindricalBinaryCompactObject&>(
            *binary_compact_object));
  };
  for (const bool with_boundary_conds : {true, false}) {
    check_impl(create_option_string(false, with_boundary_conds),
               with_boundary_conds);
  }
}

void test_parse_errors() {
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 1.0, 0.4, 1.0, 1, 3, nullptr,
          create_inner_boundary_condition(), create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("OuterRadius is too small"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{-2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 1.0, 0.4, 25.0, 1, 3, nullptr,
          create_inner_boundary_condition(), create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "The x-coordinate of the input CenterA is expected to be positive"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {5.0, 0.05, 0.0}, 1.0, 0.4, 25.0, 1, 3, nullptr,
          create_inner_boundary_condition(), create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "The x-coordinate of the input CenterB is expected to be negative"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, -1.0, 0.4, 25.0, 1, 3, nullptr,
          create_inner_boundary_condition(), create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("RadiusA and RadiusB are expected "
                                "to be positive"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 1.0, -0.4, 25.0, 1, 3, nullptr,
          create_inner_boundary_condition(), create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("RadiusA and RadiusB are expected "
                                "to be positive"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 0.15, 0.4, 25.0, 1, 3, nullptr,
          create_inner_boundary_condition(), create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "RadiusA should not be smaller than RadiusB"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-1.0, 0.05, 0.0}, 1.0, 0.4, 25.0, 1, 3, nullptr,
          create_inner_boundary_condition(), create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("We expect |x_A| <= |x_B|"));
  // Note: the boundary condition-related parse errors are checked in the
  // test_connectivity function.
}
}  // namespace

// [[Timeout, 15]]
SPECTRE_TEST_CASE(
    "Unit.Domain.Creators.CylindricalBinaryCompactObject",
    "[Domain][Unit]") {
  test_connectivity();
  test_bbh_time_dependent_factory(true);
  test_bbh_time_dependent_factory(false);
  test_binary_factory();
  test_parse_errors();
}
