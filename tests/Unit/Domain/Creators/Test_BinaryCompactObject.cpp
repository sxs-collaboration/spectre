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
#include "Domain/Creators/BinaryCompactObject.hpp"
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
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::lower_zeta(), 50);
}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_outer_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_zeta(), 50);
}

auto create_boundary_conditions(const bool excise_A, const bool excise_B) {
  size_t total_blocks = 54;
  if (not excise_A) {
    total_blocks++;
  }
  if (not excise_B) {
    total_blocks++;
  }
  BoundaryCondVector boundary_conditions_all_blocks{total_blocks};
  if (excise_A) {
    for (size_t block_id = 0; block_id < 6; ++block_id) {
      boundary_conditions_all_blocks[block_id][Direction<3>::lower_zeta()] =
          create_inner_boundary_condition();
    }
  }
  if (excise_B) {
    const size_t block_offset = 12;
    for (size_t block_id = block_offset; block_id < block_offset + 6;
         ++block_id) {
      boundary_conditions_all_blocks[block_id][Direction<3>::lower_zeta()] =
          create_inner_boundary_condition();
    }
  }
  const size_t block_offset = 44;
  for (size_t block_id = block_offset; block_id < block_offset + 10;
       ++block_id) {
    boundary_conditions_all_blocks[block_id][Direction<3>::upper_zeta()] =
        create_outer_boundary_condition();
  }
  return boundary_conditions_all_blocks;
}

template <typename... FuncsOfTime>
void test_binary_compact_object_construction(
    const domain::creators::BinaryCompactObject& binary_compact_object,
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
  test_physical_separation(binary_compact_object.create_domain().blocks(), time,
                           functions_of_time);

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
  constexpr double inner_radius_objectA = 0.5;
  constexpr double outer_radius_objectA = 1.0;
  constexpr double xcoord_objectA = -3.0;

  // ObjectB:
  constexpr double inner_radius_objectB = 0.3;
  constexpr double outer_radius_objectB = 1.0;
  constexpr double xcoord_objectB = 3.0;

  // Enveloping Cube:
  constexpr double radius_enveloping_cube = 25.5;
  constexpr double radius_enveloping_sphere = 32.4;

  // Misc.:
  constexpr size_t refinement = 1;
  constexpr size_t grid_points = 3;
  constexpr bool use_projective_map = true;

  // Options for outer sphere
  constexpr size_t addition_to_outer_layer_radial_refinement_level = 3;

  for (const bool with_boundary_conditions : {true, false}) {
    CAPTURE(with_boundary_conditions);
    for (const bool excise_interiorA : {true, false}) {
      CAPTURE(excise_interiorA);
      for (const bool excise_interiorB : {true, false}) {
        CAPTURE(excise_interiorB);
        for (const bool use_logarithmic_map_outer_spherical_shell :
             {true, false}) {
          CAPTURE(use_logarithmic_map_outer_spherical_shell);
          const domain::creators::BinaryCompactObject binary_compact_object{
              inner_radius_objectA,
              outer_radius_objectA,
              xcoord_objectA,
              excise_interiorA,
              inner_radius_objectB,
              outer_radius_objectB,
              xcoord_objectB,
              excise_interiorB,
              radius_enveloping_cube,
              radius_enveloping_sphere,
              refinement,
              grid_points,
              use_projective_map,
              use_logarithmic_map_outer_spherical_shell,
              addition_to_outer_layer_radial_refinement_level,
              false,
              0,
              false,
              0,
              nullptr,
              with_boundary_conditions
                  ? (excise_interiorA or excise_interiorB
                         ? create_inner_boundary_condition()
                         : std::make_unique<
                               TestHelpers::domain::BoundaryConditions::
                                   TestNoneBoundaryCondition<3>>())
                  : nullptr,
              with_boundary_conditions ? create_outer_boundary_condition()
                                       : nullptr};
          test_binary_compact_object_construction(
              binary_compact_object,
              std::numeric_limits<double>::signaling_NaN(), {}, {},
              with_boundary_conditions ? create_boundary_conditions(
                                             excise_interiorA, excise_interiorB)
                                       : BoundaryCondVector{});

          // Also check whether the radius of the inner boundary of Layer 5 is
          // chosen correctly.
          // Compute the radius of a point in the grid frame on this boundary.
          // Block 44 is one block whose -zeta face is on this boundary.
          const auto map{binary_compact_object.create_domain()
                             .blocks()[44]
                             .stationary_map()
                             .get_clone()};
          tnsr::I<double, 3, Frame::Logical> logical_point(
              std::array<double, 3>{{0.0, 0.0, -1.0}});
          const double layer_5_inner_radius =
              get(magnitude(std::move(map)->operator()(logical_point)));
          // The number of radial divisions in layers 4 and 5, excluding those
          // resulting from InitialRefinement > 0.
          const auto radial_divisions_in_outer_layers = static_cast<double>(
              pow(2, addition_to_outer_layer_radial_refinement_level) + 1);
          if (use_logarithmic_map_outer_spherical_shell) {
            CHECK(layer_5_inner_radius / radius_enveloping_cube ==
                  approx(pow(radius_enveloping_sphere / radius_enveloping_cube,
                             1.0 / radial_divisions_in_outer_layers)));
          } else {
            CHECK(layer_5_inner_radius - radius_enveloping_cube ==
                  approx((radius_enveloping_sphere - radius_enveloping_cube) /
                         radial_divisions_in_outer_layers));
          }
          if (with_boundary_conditions) {
            CHECK_THROWS_WITH(
                domain::creators::BinaryCompactObject(
                    inner_radius_objectA, outer_radius_objectA, xcoord_objectA,
                    excise_interiorA, inner_radius_objectB,
                    outer_radius_objectB, xcoord_objectB, excise_interiorB,
                    radius_enveloping_cube, radius_enveloping_sphere,
                    refinement, grid_points, use_projective_map,
                    use_logarithmic_map_outer_spherical_shell,
                    addition_to_outer_layer_radial_refinement_level, false, 0,
                    false, 0, nullptr,
                    excise_interiorA or excise_interiorB
                        ? create_inner_boundary_condition()
                        : std::make_unique<
                              TestHelpers::domain::BoundaryConditions::
                                  TestNoneBoundaryCondition<3>>(),
                    std::make_unique<TestHelpers::domain::BoundaryConditions::
                                         TestPeriodicBoundaryCondition<3>>(),
                    Options::Context{false, {}, 1, 1}),
                Catch::Matchers::Contains("Cannot have periodic boundary "
                                          "conditions with a binary domain"));
            if (excise_interiorA or excise_interiorB) {
              CHECK_THROWS_WITH(
                  domain::creators::BinaryCompactObject(
                      inner_radius_objectA, outer_radius_objectA,
                      xcoord_objectA, excise_interiorA, inner_radius_objectB,
                      outer_radius_objectB, xcoord_objectB, excise_interiorB,
                      radius_enveloping_cube, radius_enveloping_sphere,
                      refinement, grid_points, use_projective_map,
                      use_logarithmic_map_outer_spherical_shell,
                      addition_to_outer_layer_radial_refinement_level, false, 0,
                      false, 0, nullptr,
                      std::make_unique<TestHelpers::domain::BoundaryConditions::
                                           TestPeriodicBoundaryCondition<3>>(),
                      create_outer_boundary_condition(),
                      Options::Context{false, {}, 1, 1}),
                  Catch::Matchers::Contains("Cannot have periodic boundary "
                                            "conditions with a binary domain"));
            }
            CHECK_THROWS_WITH(
                domain::creators::BinaryCompactObject(
                    inner_radius_objectA, outer_radius_objectA, xcoord_objectA,
                    excise_interiorA, inner_radius_objectB,
                    outer_radius_objectB, xcoord_objectB, excise_interiorB,
                    radius_enveloping_cube, radius_enveloping_sphere,
                    refinement, grid_points, use_projective_map,
                    use_logarithmic_map_outer_spherical_shell,
                    addition_to_outer_layer_radial_refinement_level, false, 0,
                    false, 0, nullptr,
                    excise_interiorA or excise_interiorB
                        ? create_inner_boundary_condition()
                        : std::make_unique<
                              TestHelpers::domain::BoundaryConditions::
                                  TestNoneBoundaryCondition<3>>(),
                    nullptr, Options::Context{false, {}, 1, 1}),
                Catch::Matchers::Contains(
                    "Must specify either both inner and outer boundary "
                    "conditions or neither."));
            CHECK_THROWS_WITH(
                domain::creators::BinaryCompactObject(
                    inner_radius_objectA, outer_radius_objectA, xcoord_objectA,
                    excise_interiorA, inner_radius_objectB,
                    outer_radius_objectB, xcoord_objectB, excise_interiorB,
                    radius_enveloping_cube, radius_enveloping_sphere,
                    refinement, grid_points, use_projective_map,
                    use_logarithmic_map_outer_spherical_shell,
                    addition_to_outer_layer_radial_refinement_level, false, 0,
                    false, 0, nullptr, nullptr,
                    create_outer_boundary_condition(),
                    Options::Context{false, {}, 1, 1}),
                Catch::Matchers::Contains(
                    excise_interiorA or excise_interiorB
                        ? std::string{"Must specify either both inner and "
                                      "outer boundary "
                                      "conditions or neither."}
                        : std::string{"Inner boundary condition must be None "
                                      "if ExciseInteriorA and ExciseInteriorB "
                                      "are both false"}));
          }
        }
      }
    }
  }
}

std::string stringize(const bool t) { return t ? "true" : "false"; }

std::string create_option_string(const bool excise_A, const bool excise_B,
                                 const bool add_time_dependence,
                                 const bool use_logarithmic_map_AB,
                                 const size_t additional_refinement_outer,
                                 const size_t additional_refinement_A,
                                 const size_t additional_refinement_B,
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
      add_boundary_condition
          ? std::string{"  BoundaryConditions:\n"
                        "    InnerBoundary:\n" +
                        std::string{excise_A or excise_B
                                        ? "      TestBoundaryCondition:\n"
                                          "        Direction: lower-zeta\n"
                                          "        BlockId: 50\n"
                                        : "      None:\n"} +
                        "    OuterBoundary:\n"
                        "      TestBoundaryCondition:\n"
                        "        Direction: upper-zeta\n"
                        "        BlockId: 50\n"}
          : ""};
  return "BinaryCompactObject:\n"
         "  InnerRadiusObjectA: 0.2\n"
         "  OuterRadiusObjectA: 1.0\n"
         "  XCoordObjectA: -2.0\n"
         "  ExciseInteriorA: " +
         stringize(excise_A) +
         "\n"
         "  InnerRadiusObjectB: 1.0\n"
         "  OuterRadiusObjectB: 2.0\n"
         "  XCoordObjectB: 3.0\n"
         "  ExciseInteriorB: " +
         stringize(excise_B) +
         "\n"
         "  RadiusOuterCube: 22.0\n"
         "  RadiusOuterSphere: 25.0\n"
         "  InitialRefinement: 1\n"
         "  InitialGridPoints: 3\n"
         "  UseProjectiveMap: true\n"
         "  UseLogarithmicMapOuterSphericalShell: false\n"
         "  AdditionToOuterLayerRadialRefinementLevel: " +
         std::to_string(additional_refinement_outer) +
         "\n"
         "  UseLogarithmicMapObjectA: " +
         stringize(use_logarithmic_map_AB) +
         "\n"
         "  AdditionToObjectARadialRefinementLevel: " +
         std::to_string(additional_refinement_A) +
         "\n"
         "  UseLogarithmicMapObjectB: " +
         stringize(use_logarithmic_map_AB) +
         "\n"
         "  AdditionToObjectBRadialRefinementLevel: " +
         std::to_string(additional_refinement_B) + "\n" + time_dependence +
         boundary_conditions;
}

void test_bbh_time_dependent_factory(const bool with_boundary_conditions) {
  const auto binary_compact_object = [&with_boundary_conditions]() {
    if (with_boundary_conditions) {
      return TestHelpers::test_factory_creation<
          DomainCreator<3>, domain::OptionTags::DomainCreator<3>,
          TestHelpers::domain::BoundaryConditions::
              MetavariablesWithBoundaryConditions<3>>(create_option_string(
          true, true, true, false, 0, 0, 0, with_boundary_conditions));
    } else {
      return TestHelpers::test_factory_creation<
          DomainCreator<3>, domain::OptionTags::DomainCreator<3>,
          TestHelpers::domain::BoundaryConditions::
              MetavariablesWithoutBoundaryConditions<3>>(create_option_string(
          true, true, true, false, 0, 0, 0, with_boundary_conditions));
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
        dynamic_cast<const domain::creators::BinaryCompactObject&>(
            *binary_compact_object),
        time, functions_of_time, expected_functions_of_time,
        with_boundary_conditions ? create_boundary_conditions(true, true)
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
        dynamic_cast<const domain::creators::BinaryCompactObject&>(
            *binary_compact_object));
  };
  for (const bool with_boundary_conds : {true, false}) {
    check_impl(create_option_string(true, true, false, false, 2, 0, 2,
                                    with_boundary_conds),
               with_boundary_conds);
    check_impl(create_option_string(true, true, false, true, 3, 3, 0,
                                    with_boundary_conds),
               with_boundary_conds);
    check_impl(create_option_string(true, true, false, false, 0, 0, 0,
                                    with_boundary_conds),
               with_boundary_conds);
    check_impl(create_option_string(false, false, false, false, 0, 0, 0,
                                    with_boundary_conds),
               with_boundary_conds);
    check_impl(create_option_string(true, false, false, false, 0, 0, 0,
                                    with_boundary_conds),
               with_boundary_conds);
    check_impl(create_option_string(false, true, false, false, 0, 0, 0,
                                    with_boundary_conds),
               with_boundary_conds);
  }
}
}  // namespace

// [[Timeout, 15]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObject.FactoryTests",
                  "[Domain][Unit]") {
  test_connectivity();
  test_bbh_time_dependent_factory(true);
  test_bbh_time_dependent_factory(false);
  test_binary_factory();
}

// [[OutputRegex, The radius for the enveloping cube is too small! The Frustums
// will be malformed.]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObject.Options1",
                  "[Domain][Unit]") {
  ERROR_TEST();
  // ObjectA:
  const double inner_radius_objectA = 0.5;
  const double outer_radius_objectA = 1.0;
  const double xcoord_objectA = -7.0;
  const bool excise_interiorA = true;

  // ObjectB:
  const double inner_radius_objectB = 0.3;
  const double outer_radius_objectB = 1.0;
  const double xcoord_objectB = 8.0;
  const bool excise_interiorB = true;

  // Enveloping Cube:
  const double radius_enveloping_cube = 25.5;
  const double radius_enveloping_sphere = 32.4;

  // Misc.:
  const size_t refinement = 2;
  const size_t grid_points = 6;

  domain::creators::BinaryCompactObject binary_compact_object{
      inner_radius_objectA,     outer_radius_objectA, xcoord_objectA,
      excise_interiorA,         inner_radius_objectB, outer_radius_objectB,
      xcoord_objectB,           excise_interiorB,     radius_enveloping_cube,
      radius_enveloping_sphere, refinement,           grid_points};
}
// [[OutputRegex, ObjectA's inner radius must be less than its outer radius.]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObject.Options2",
                  "[Domain][Unit]") {
  ERROR_TEST();
  // ObjectA:
  const double inner_radius_objectA = 1.5;
  const double outer_radius_objectA = 1.0;
  const double xcoord_objectA = -1.0;
  const bool excise_interiorA = true;

  // ObjectB:
  const double inner_radius_objectB = 0.3;
  const double outer_radius_objectB = 1.0;
  const double xcoord_objectB = 1.0;
  const bool excise_interiorB = true;

  // Enveloping Cube:
  const double radius_enveloping_cube = 25.5;
  const double radius_enveloping_sphere = 32.4;

  // Misc.:
  const size_t refinement = 2;
  const size_t grid_points = 6;

  domain::creators::BinaryCompactObject binary_compact_object{
      inner_radius_objectA,     outer_radius_objectA, xcoord_objectA,
      excise_interiorA,         inner_radius_objectB, outer_radius_objectB,
      xcoord_objectB,           excise_interiorB,     radius_enveloping_cube,
      radius_enveloping_sphere, refinement,           grid_points};
}
// [[OutputRegex, ObjectB's inner radius must be less than its outer radius.]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObject.Options3",
                  "[Domain][Unit]") {
  ERROR_TEST();
  // ObjectA:
  const double inner_radius_objectA = 0.5;
  const double outer_radius_objectA = 1.0;
  const double xcoord_objectA = -1.0;
  const bool excise_interiorA = true;

  // ObjectB:
  const double inner_radius_objectB = 3.3;
  const double outer_radius_objectB = 1.0;
  const double xcoord_objectB = 1.0;
  const bool excise_interiorB = true;

  // Enveloping Cube:
  const double radius_enveloping_cube = 25.5;
  const double radius_enveloping_sphere = 32.4;

  // Misc.:
  const size_t refinement = 2;
  const size_t grid_points = 6;

  domain::creators::BinaryCompactObject binary_compact_object{
      inner_radius_objectA,     outer_radius_objectA, xcoord_objectA,
      excise_interiorA,         inner_radius_objectB, outer_radius_objectB,
      xcoord_objectB,           excise_interiorB,     radius_enveloping_cube,
      radius_enveloping_sphere, refinement,           grid_points};
}
