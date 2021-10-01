// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
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
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/OptionTags.hpp"
#include "Domain/Protocols/Metavariables.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime

namespace {
using BoundaryCondVector = std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>;
using Object = domain::creators::BinaryCompactObject::Object;
using Excision = domain::creators::BinaryCompactObject::Excision;

template <size_t Dim, bool EnableTimeDependentMaps, bool WithBoundaryConditions>
struct Metavariables {
  struct domain : tt::ConformsTo<::domain::protocols::Metavariables> {
    static constexpr bool enable_time_dependent_maps = EnableTimeDependentMaps;
  };
  using system = tmpl::conditional_t<WithBoundaryConditions,
                                     TestHelpers::domain::BoundaryConditions::
                                         SystemWithBoundaryConditions<Dim>,
                                     TestHelpers::domain::BoundaryConditions::
                                         SystemWithoutBoundaryConditions<Dim>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        DomainCreator<3>, tmpl::list<::domain::creators::BinaryCompactObject>>>;
  };
};

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
  constexpr size_t grid_points = 3;
  constexpr bool use_projective_map = true;

  for (const bool with_boundary_conditions : {true, false}) {
    CAPTURE(with_boundary_conditions);
    for (const bool excise_interiorA : {true, false}) {
      CAPTURE(excise_interiorA);
      for (const bool excise_interiorB : {true, false}) {
        CAPTURE(excise_interiorB);
        std::unordered_map<std::string, std::array<size_t, 3>> refinement{
            {"ObjectAShell", {{1, 1, 1}}},
            {"ObjectACube", {{1, 1, 1}}},
            {"ObjectBShell", {{1, 1, 1}}},
            {"ObjectBCube", {{1, 1, 1}}},
            {"EnvelopingCube", {{1, 1, 1}}},
            {"CubedShell", {{1, 1, 1}}},
            // Add some radial refinement in outer shell
            {"OuterShell", {{1, 1, 4}}}};
        if (not excise_interiorA) {
          refinement["ObjectAInterior"] = std::array<size_t, 3>{{1, 1, 1}};
        }
        if (not excise_interiorB) {
          refinement["ObjectBInterior"] = std::array<size_t, 3>{{1, 1, 1}};
        }
        for (const bool use_logarithmic_map_outer_spherical_shell :
             {true, false}) {
          CAPTURE(use_logarithmic_map_outer_spherical_shell);
          const domain::creators::BinaryCompactObject binary_compact_object{
              Object{inner_radius_objectA, outer_radius_objectA, xcoord_objectA,
                     excise_interiorA
                         ? std::make_optional(
                               Excision{with_boundary_conditions
                                            ? create_inner_boundary_condition()
                                            : nullptr})
                         : std::nullopt,
                     false},
              Object{inner_radius_objectB, outer_radius_objectB, xcoord_objectB,
                     excise_interiorB
                         ? std::make_optional(
                               Excision{with_boundary_conditions
                                            ? create_inner_boundary_condition()
                                            : nullptr})
                         : std::nullopt,
                     false},
              radius_enveloping_cube,
              radius_enveloping_sphere,
              refinement,
              grid_points,
              use_projective_map,
              use_logarithmic_map_outer_spherical_shell,
              with_boundary_conditions ? create_outer_boundary_condition()
                                       : nullptr};

          std::vector<std::string> expected_block_names{
              "ObjectAShellUpperZ",       "ObjectAShellLowerZ",
              "ObjectAShellUpperY",       "ObjectAShellLowerY",
              "ObjectAShellUpperX",       "ObjectAShellLowerX",
              "ObjectACubeUpperZ",        "ObjectACubeLowerZ",
              "ObjectACubeUpperY",        "ObjectACubeLowerY",
              "ObjectACubeUpperX",        "ObjectACubeLowerX",
              "ObjectBShellUpperZ",       "ObjectBShellLowerZ",
              "ObjectBShellUpperY",       "ObjectBShellLowerY",
              "ObjectBShellUpperX",       "ObjectBShellLowerX",
              "ObjectBCubeUpperZ",        "ObjectBCubeLowerZ",
              "ObjectBCubeUpperY",        "ObjectBCubeLowerY",
              "ObjectBCubeUpperX",        "ObjectBCubeLowerX",
              "EnvelopingCubeUpperZLeft", "EnvelopingCubeUpperZRight",
              "EnvelopingCubeLowerZLeft", "EnvelopingCubeLowerZRight",
              "EnvelopingCubeUpperYLeft", "EnvelopingCubeUpperYRight",
              "EnvelopingCubeLowerYLeft", "EnvelopingCubeLowerYRight",
              "EnvelopingCubeUpperX",     "EnvelopingCubeLowerX",
              "CubedShellUpperZLeft",     "CubedShellUpperZRight",
              "CubedShellLowerZLeft",     "CubedShellLowerZRight",
              "CubedShellUpperYLeft",     "CubedShellUpperYRight",
              "CubedShellLowerYLeft",     "CubedShellLowerYRight",
              "CubedShellUpperX",         "CubedShellLowerX",
              "OuterShellUpperZLeft",     "OuterShellUpperZRight",
              "OuterShellLowerZLeft",     "OuterShellLowerZRight",
              "OuterShellUpperYLeft",     "OuterShellUpperYRight",
              "OuterShellLowerYLeft",     "OuterShellLowerYRight",
              "OuterShellUpperX",         "OuterShellLowerX"};
          std::unordered_map<std::string, std::unordered_set<std::string>>
              expected_block_groups{
                  {"ObjectAShell",
                   {"ObjectAShellLowerZ", "ObjectAShellUpperX",
                    "ObjectAShellLowerX", "ObjectAShellUpperY",
                    "ObjectAShellUpperZ", "ObjectAShellLowerY"}},
                  {"ObjectBShell",
                   {"ObjectBShellLowerZ", "ObjectBShellUpperX",
                    "ObjectBShellLowerX", "ObjectBShellUpperY",
                    "ObjectBShellUpperZ", "ObjectBShellLowerY"}},
                  {"ObjectACube",
                   {"ObjectACubeLowerY", "ObjectACubeLowerZ",
                    "ObjectACubeUpperY", "ObjectACubeUpperX",
                    "ObjectACubeUpperZ", "ObjectACubeLowerX"}},
                  {"ObjectBCube",
                   {"ObjectBCubeLowerY", "ObjectBCubeLowerZ",
                    "ObjectBCubeUpperY", "ObjectBCubeUpperX",
                    "ObjectBCubeUpperZ", "ObjectBCubeLowerX"}},
                  {"EnvelopingCube",
                   {"EnvelopingCubeUpperZRight", "EnvelopingCubeUpperX",
                    "EnvelopingCubeLowerZLeft", "EnvelopingCubeLowerZRight",
                    "EnvelopingCubeUpperYRight", "EnvelopingCubeLowerYRight",
                    "EnvelopingCubeUpperYLeft", "EnvelopingCubeUpperZLeft",
                    "EnvelopingCubeLowerYLeft", "EnvelopingCubeLowerX"}},
                  {"CubedShell",
                   {"CubedShellUpperZLeft", "CubedShellUpperZRight",
                    "CubedShellLowerZLeft", "CubedShellLowerZRight",
                    "CubedShellUpperYLeft", "CubedShellUpperYRight",
                    "CubedShellLowerYLeft", "CubedShellLowerYRight",
                    "CubedShellUpperX", "CubedShellLowerX"}},
                  {"OuterShell",
                   {"OuterShellUpperZLeft", "OuterShellUpperZRight",
                    "OuterShellLowerZLeft", "OuterShellLowerZRight",
                    "OuterShellUpperYLeft", "OuterShellUpperYRight",
                    "OuterShellLowerYLeft", "OuterShellLowerYRight",
                    "OuterShellUpperX", "OuterShellLowerX"}}};
          if (not excise_interiorA) {
            expected_block_names.emplace_back("ObjectAInterior");
          }
          if (not excise_interiorB) {
            expected_block_names.emplace_back("ObjectBInterior");
          }
          CHECK(binary_compact_object.block_names() == expected_block_names);
          CHECK(binary_compact_object.block_groups() == expected_block_groups);

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
          tnsr::I<double, 3, Frame::BlockLogical> logical_point(
              std::array<double, 3>{{0.0, 0.0, -1.0}});
          const double layer_5_inner_radius =
              get(magnitude(std::move(map)->operator()(logical_point)));
          // The number of radial divisions in layers 4 and 5, excluding those
          // resulting from InitialRefinement > 0.
          const double radial_divisions_in_outer_layers = 9;
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
            using PeriodicBc = TestHelpers::domain::BoundaryConditions::
                TestPeriodicBoundaryCondition<3>;
            CHECK_THROWS_WITH(
                domain::creators::BinaryCompactObject(
                    Object{inner_radius_objectA, outer_radius_objectA,
                           xcoord_objectA,
                           excise_interiorA
                               ? std::make_optional(Excision{
                                     create_inner_boundary_condition()})
                               : std::nullopt,
                           false},
                    domain::creators::BinaryCompactObject::Object{
                        inner_radius_objectB, outer_radius_objectB,
                        xcoord_objectB,
                        excise_interiorB
                            ? std::make_optional(
                                  Excision{create_inner_boundary_condition()})
                            : std::nullopt,
                        false},
                    radius_enveloping_cube, radius_enveloping_sphere,
                    refinement, grid_points, use_projective_map,
                    use_logarithmic_map_outer_spherical_shell,
                    std::make_unique<PeriodicBc>(),
                    Options::Context{false, {}, 1, 1}),
                Catch::Matchers::Contains("Cannot have periodic boundary "
                                          "conditions with a binary domain"));
            if (excise_interiorA or excise_interiorB) {
              CHECK_THROWS_WITH(
                  domain::creators::BinaryCompactObject(
                      Object{inner_radius_objectA, outer_radius_objectA,
                             xcoord_objectA,
                             excise_interiorA
                                 ? std::make_optional(
                                       Excision{std::make_unique<PeriodicBc>()})
                                 : std::nullopt,
                             false},
                      Object{inner_radius_objectB, outer_radius_objectB,
                             xcoord_objectB,
                             excise_interiorB
                                 ? std::make_optional(
                                       Excision{std::make_unique<PeriodicBc>()})
                                 : std::nullopt,
                             false},
                      radius_enveloping_cube, radius_enveloping_sphere,
                      refinement, grid_points, use_projective_map,
                      use_logarithmic_map_outer_spherical_shell,
                      create_outer_boundary_condition(),
                      Options::Context{false, {}, 1, 1}),
                  Catch::Matchers::Contains("Cannot have periodic boundary "
                                            "conditions with a binary domain"));
              CHECK_THROWS_WITH(
                  domain::creators::BinaryCompactObject(
                      Object{inner_radius_objectA, outer_radius_objectA,
                             xcoord_objectA,
                             excise_interiorA
                                 ? std::make_optional(Excision{
                                       create_inner_boundary_condition()})
                                 : std::nullopt,
                             false},
                      Object{inner_radius_objectB, outer_radius_objectB,
                             xcoord_objectB,
                             excise_interiorB
                                 ? std::make_optional(Excision{
                                       create_inner_boundary_condition()})
                                 : std::nullopt,
                             false},
                      radius_enveloping_cube, radius_enveloping_sphere,
                      refinement, grid_points, use_projective_map,
                      use_logarithmic_map_outer_spherical_shell, nullptr,
                      Options::Context{false, {}, 1, 1}),
                  Catch::Matchers::Contains(
                      "Must specify either both inner and outer boundary "
                      "conditions or neither."));
              CHECK_THROWS_WITH(
                  domain::creators::BinaryCompactObject(
                      Object{inner_radius_objectA, outer_radius_objectA,
                             xcoord_objectA,
                             excise_interiorA
                                 ? std::make_optional(Excision{nullptr})
                                 : std::nullopt,
                             false},
                      Object{inner_radius_objectB, outer_radius_objectB,
                             xcoord_objectB,
                             excise_interiorB
                                 ? std::make_optional(Excision{nullptr})
                                 : std::nullopt,
                             false},
                      radius_enveloping_cube, radius_enveloping_sphere,
                      refinement, grid_points, use_projective_map,
                      use_logarithmic_map_outer_spherical_shell,
                      create_outer_boundary_condition(),
                      Options::Context{false, {}, 1, 1}),
                  Catch::Matchers::Contains(
                      "Must specify either both inner and outer boundary "
                      "conditions or neither."));
            }
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
      add_time_dependence ? "  TimeDependentMaps:\n"
                            "    InitialTime: 1.0\n"
                            "    InitialExpirationDeltaT: 9.0\n"
                            "    ExpansionMap: \n"
                            "      OuterBoundary: 25.0\n"
                            "      InitialExpansion: 1.0\n"
                            "      InitialExpansionVelocity: -0.1\n"
                            "      FunctionOfTimeName: 'ExpansionFactor'\n"
                            "      AsymptoticVelocityOuterBoundary: -0.1\n"
                            "      DecayTimescaleOuterBoundaryVelocity: 5.0\n"
                            "    RotationAboutZAxisMap:\n"
                            "      InitialRotationAngle: 2.0\n"
                            "      InitialAngularVelocity: -0.2\n"
                            "      FunctionOfTimeName: RotationAngle\n"
                            "    SizeMap:\n"
                            "      InitialValues: [0.0, 0.0]\n"
                            "      InitialVelocities: [-0.1, -0.2]\n"
                            "      InitialAccelerations: [0.01, 0.02]\n"
                            "      FunctionOfTimeNames: ['LambdaFactorA0', "
                            " 'LambdaFactorB0']"
                          : ""};
  const std::string interior_A{
      add_boundary_condition
          ? std::string{"    Interior:\n" +
                        std::string{excise_A
                                        ? "      ExciseWithBoundaryCondition:\n"
                                          "        TestBoundaryCondition:\n"
                                          "          Direction: lower-zeta\n"
                                          "          BlockId: 50\n"
                                        : "      Auto\n"}}
          : "    ExciseInterior: " + stringize(excise_A) + "\n"};
  const std::string interior_B{
      add_boundary_condition
          ? std::string{"    Interior:\n" +
                        std::string{excise_B
                                        ? "      ExciseWithBoundaryCondition:\n"
                                          "        TestBoundaryCondition:\n"
                                          "          Direction: lower-zeta\n"
                                          "          BlockId: 50\n"
                                        : "      Auto\n"}}
          : "    ExciseInterior: " + stringize(excise_B) + "\n"};
  const std::string outer_boundary_condition{
      add_boundary_condition ? std::string{"    BoundaryCondition:\n"
                                           "      TestBoundaryCondition:\n"
                                           "        Direction: upper-zeta\n"
                                           "        BlockId: 50\n"}
                             : ""};
  return "BinaryCompactObject:\n"
         "  ObjectA:\n"
         "    InnerRadius: 0.2\n"
         "    OuterRadius: 1.0\n"
         "    XCoord: -2.0\n" +
         interior_A +
         "    UseLogarithmicMap: " + stringize(use_logarithmic_map_AB) +
         "\n"
         "  ObjectB:\n"
         "    InnerRadius: 1.0\n"
         "    OuterRadius: 2.0\n"
         "    XCoord: 3.0\n" +
         interior_B +
         "    UseLogarithmicMap: " + stringize(use_logarithmic_map_AB) +
         "\n"
         "  EnvelopingCube:\n"
         "    Radius: 22.0\n"
         "    UseProjectiveMap: true\n"
         "  OuterSphere:\n"
         "    Radius: 25.0\n"
         "    UseLogarithmicMap: false\n" +
         outer_boundary_condition + "  InitialRefinement:\n" +
         (excise_A ? "" : "    ObjectAInterior: [1, 1, 1]\n") +
         (excise_B ? "" : "    ObjectBInterior: [1, 1, 1]\n") +
         "    ObjectAShell: [1, 1, " +
         std::to_string(1 + additional_refinement_A) +
         "]\n"
         "    ObjectBShell: [1, 1, " +
         std::to_string(1 + additional_refinement_B) +
         "]\n"
         "    ObjectACube: [1, 1, 1]\n"
         "    ObjectBCube: [1, 1, 1]\n"
         "    EnvelopingCube: [1, 1, 1]\n"
         "    CubedShell: [1, 1, 1]\n"
         "    OuterShell: [1, 1, " +
         std::to_string(1 + additional_refinement_outer) +
         "]\n"
         "  InitialGridPoints: 3\n" +
         time_dependence;
}

void test_bbh_time_dependent_factory(const bool with_boundary_conditions) {
  const auto binary_compact_object = [&with_boundary_conditions]() {
    if (with_boundary_conditions) {
      return TestHelpers::test_option_tag<domain::OptionTags::DomainCreator<3>,
                                          Metavariables<3, true, true>>(
          create_option_string(true, true, true, false, 0, 0, 0,
                               with_boundary_conditions));
    } else {
      return TestHelpers::test_option_tag<domain::OptionTags::DomainCreator<3>,
                                          Metavariables<3, true, false>>(
          create_option_string(true, true, true, false, 0, 0, 0,
                               with_boundary_conditions));
    }
  }();
  const std::array<double, 4> times_to_check{{0.0, 4.4, 7.8}};

  constexpr double initial_time = 0.0;
  constexpr double expiration_time = 10.0;
  constexpr double expected_time = 1.0; // matches InitialTime: 1.0 above
  constexpr double expected_update_delta_t =
      9.0;  // matches InitialExpirationDeltaT: 9.0 above
  constexpr double expected_initial_function_value =
      1.0;  // hard-coded in BinaryCompactObject.cpp
  constexpr double expected_asymptotic_velocity_outer_boundary =
      -0.1;  // matches AsymptoticVelocityOuterBoundary: -0.1 above
  constexpr double expected_decay_timescale_outer_boundary_velocity =
      5.0;  // matches DecayTimescaleOuterBoundaryVelocity: 5.0 above
  std::array<DataVector, 3> expansion_factor_coefs{{{1.0}, {-0.1}, {0.0}}};
  std::array<DataVector, 4> rotation_angle_coefs{{{2.0}, {-0.2}, {0.0}, {0.0}}};
  std::array<DataVector, 4> lambda_factor_a0_coefs{
      {{0.0}, {-0.1}, {0.01}, {0.0}}};
  std::array<DataVector, 4> lambda_factor_b0_coefs{
      {{0.0}, {-0.2}, {0.02}, {0.0}}};

  const std::tuple<
      std::pair<std::string, domain::FunctionsOfTime::PiecewisePolynomial<3>>,
      std::pair<std::string, domain::FunctionsOfTime::PiecewisePolynomial<3>>,
      std::pair<std::string, domain::FunctionsOfTime::PiecewisePolynomial<2>>,
      std::pair<std::string, domain::FunctionsOfTime::FixedSpeedCubic>,
      std::pair<std::string, domain::FunctionsOfTime::PiecewisePolynomial<3>>>
      expected_functions_of_time = std::make_tuple(
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<3>>{
              "LambdaFactorA0"s,
              {expected_time, lambda_factor_a0_coefs,
               expected_time + expected_update_delta_t}},
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<3>>{
              "LambdaFactorB0"s,
              {expected_time, lambda_factor_b0_coefs,
               expected_time + expected_update_delta_t}},
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<2>>{
              "ExpansionFactor"s,
              {expected_time, expansion_factor_coefs,
               expected_time + expected_update_delta_t}},
          std::pair<std::string, domain::FunctionsOfTime::FixedSpeedCubic>{
              "ExpansionFactorOuterBoundary"s,
              {expected_initial_function_value, expected_time,
               expected_asymptotic_velocity_outer_boundary,
               expected_decay_timescale_outer_boundary_velocity}},
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<3>>{
              "RotationAngle"s,
              {expected_time, rotation_angle_coefs,
               expected_time + expected_update_delta_t}});
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time["ExpansionFactor"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, expansion_factor_coefs, expiration_time);
  functions_of_time["ExpansionFactorOuterBoundary"] =
      std::make_unique<domain::FunctionsOfTime::FixedSpeedCubic>(
          expected_initial_function_value, initial_time,
          expected_asymptotic_velocity_outer_boundary,
          expected_decay_timescale_outer_boundary_velocity);
  functions_of_time["RotationAngle"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time, rotation_angle_coefs, expiration_time);
  functions_of_time["LambdaFactorA0"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time, lambda_factor_a0_coefs, expiration_time);
  functions_of_time["LambdaFactorB0"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time, lambda_factor_b0_coefs, expiration_time);

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
        return TestHelpers::test_option_tag<
            domain::OptionTags::DomainCreator<3>,
            Metavariables<3, false, true>>(opt_string);
      } else {
        return TestHelpers::test_option_tag<
            domain::OptionTags::DomainCreator<3>,
            Metavariables<3, false, false>>(opt_string);
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

void test_parse_errors() {
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.5, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          {0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, 2_st, 6_st, true, false, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "The x-coordinate of ObjectA's center is expected to be negative."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          {0.3, 1.0, -1.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, 2_st, 6_st, true, false, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "The x-coordinate of ObjectB's center is expected to be positive."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.5, 1.0, -7.0, {{create_inner_boundary_condition()}}, false},
          {0.3, 1.0, 8.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, 2_st, 6_st, true, false, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("The radius for the enveloping cube is too "
                                "small! The Frustums will be malformed."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {1.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          {0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, 2_st, 6_st, true, false, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "ObjectA's inner radius must be less than its outer radius."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          {3.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, 2_st, 6_st, true, false, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "ObjectB's inner radius must be less than its outer radius."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.5, 1.0, -1.0, std::nullopt, true},
          {0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, 2_st, 6_st, true, false, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Using a logarithmically spaced radial grid in the "
          "part of Layer 1 enveloping Object A requires excising the interior "
          "of Object A"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          {0.3, 1.0, 1.0, std::nullopt, true}, 25.5, 32.4, 2_st, 6_st, true,
          false, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Using a logarithmically spaced radial grid in the "
          "part of Layer 1 enveloping Object B requires excising the interior "
          "of Object B"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          {0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, std::vector<std::array<size_t, 3>>{}, 6_st, true, false,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("Invalid 'InitialRefinement'"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          {0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, 2_st, std::vector<std::array<size_t, 3>>{}, true, false,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("Invalid 'InitialGridPoints'"));
  // Note: the boundary condition-related parse errors are checked in the
  // test_connectivity function.
}
}  // namespace

// [[Timeout, 15]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObject.FactoryTests",
                  "[Domain][Unit]") {
  test_connectivity();
  test_bbh_time_dependent_factory(true);
  test_bbh_time_dependent_factory(false);
  test_binary_factory();
  test_parse_errors();
}
