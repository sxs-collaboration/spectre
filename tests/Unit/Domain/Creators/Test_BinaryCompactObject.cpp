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
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
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
using ExpirationTimeMap = std::unordered_map<std::string, double>;
using Object = domain::creators::BinaryCompactObject::Object;
using Excision = domain::creators::BinaryCompactObject::Excision;
using Distribution = domain::CoordinateMaps::Distribution;

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

void test_connectivity() {
  // ObjectA:
  constexpr double inner_radius_objectA = 0.3;
  constexpr double outer_radius_objectA = 1.0;
  constexpr double xcoord_objectA = 3.0;

  // ObjectB:
  constexpr double inner_radius_objectB = 0.5;
  constexpr double outer_radius_objectB = 1.0;
  constexpr double xcoord_objectB = -3.0;

  // Enveloping Cube:
  constexpr double radius_enveloping_cube = 25.5;
  constexpr bool use_projective_map = true;
  constexpr double sphericity_enveloping_cube = 0.5;

  // Enveloping sphere:
  const std::optional<double> radius_enveloping_sphere = std::nullopt;

  // Outer shell:
  constexpr double outer_radius = 32.4;

  // Misc.:
  constexpr size_t grid_points = 3;

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
        for (const auto radial_distribution_outer_shell :
             {Distribution::Linear, Distribution::Logarithmic,
              Distribution::Inverse}) {
          CAPTURE(radial_distribution_outer_shell);
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
              outer_radius,
              refinement,
              grid_points,
              use_projective_map,
              sphericity_enveloping_cube,
              radius_enveloping_sphere,
              radial_distribution_outer_shell,
              with_boundary_conditions ? create_outer_boundary_condition()
                                       : nullptr};

          const auto domain =
              TestHelpers::domain::creators::test_domain_creator(
                  binary_compact_object, with_boundary_conditions);

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
          std::unordered_map<std::string, ExcisionSphere<3>>
              expected_excision_spheres{};
          if (excise_interiorA) {
            expected_excision_spheres.emplace(
                "ObjectAExcisionSphere",
                ExcisionSphere<3>{inner_radius_objectA,
                                  {{xcoord_objectA, 0.0, 0.0}},
                                  {{0, Direction<3>::lower_zeta()},
                                   {1, Direction<3>::lower_zeta()},
                                   {2, Direction<3>::lower_zeta()},
                                   {3, Direction<3>::lower_zeta()},
                                   {4, Direction<3>::lower_zeta()},
                                   {5, Direction<3>::lower_zeta()}}});
          }
          if (excise_interiorB) {
            expected_excision_spheres.emplace(
                "ObjectBExcisionSphere",
                ExcisionSphere<3>{inner_radius_objectB,
                                  {{xcoord_objectB, 0.0, 0.0}},
                                  {{12, Direction<3>::lower_zeta()},
                                   {13, Direction<3>::lower_zeta()},
                                   {14, Direction<3>::lower_zeta()},
                                   {15, Direction<3>::lower_zeta()},
                                   {16, Direction<3>::lower_zeta()},
                                   {17, Direction<3>::lower_zeta()}}});
          }
          CHECK(domain.excision_spheres() == expected_excision_spheres);

          // Also check whether the radius of the inner boundary of Layer 5 is
          // chosen correctly.
          // Compute the radius of a point in the grid frame on this boundary.
          // Block 44 is one block whose -zeta face is on this boundary.
          const auto map{domain.blocks()[44].stationary_map().get_clone()};
          tnsr::I<double, 3, Frame::BlockLogical> logical_point(
              std::array<double, 3>{{0.0, 0.0, -1.0}});
          const double layer_5_inner_radius =
              get(magnitude(std::move(map)->operator()(logical_point)));
          // The number of radial divisions in layers 4 and 5, excluding those
          // resulting from InitialRefinement > 0.
          const double radial_divisions_in_outer_layers = 9;
          if (radial_distribution_outer_shell == Distribution::Logarithmic) {
            CHECK(layer_5_inner_radius / radius_enveloping_cube ==
                  approx(pow(outer_radius / radius_enveloping_cube,
                             1.0 / radial_divisions_in_outer_layers)));
          } else if (radial_distribution_outer_shell == Distribution::Inverse) {
            CHECK(
                layer_5_inner_radius / radius_enveloping_cube ==
                approx(outer_radius /
                       (outer_radius - (outer_radius - radius_enveloping_cube) /
                                           radial_divisions_in_outer_layers)));
          } else {
            CHECK(layer_5_inner_radius - radius_enveloping_cube ==
                  approx((outer_radius - radius_enveloping_cube) /
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
                    radius_enveloping_cube, outer_radius, refinement,
                    grid_points, use_projective_map, sphericity_enveloping_cube,
                    radius_enveloping_sphere, radial_distribution_outer_shell,
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
                      radius_enveloping_cube, outer_radius, refinement,
                      grid_points, use_projective_map,
                      sphericity_enveloping_cube, radius_enveloping_sphere,
                      radial_distribution_outer_shell,
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
                      radius_enveloping_cube, outer_radius, refinement,
                      grid_points, use_projective_map,
                      sphericity_enveloping_cube, radius_enveloping_sphere,
                      radial_distribution_outer_shell, nullptr,
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
                      radius_enveloping_cube, outer_radius, refinement,
                      grid_points, use_projective_map,
                      sphericity_enveloping_cube, radius_enveloping_sphere,
                      radial_distribution_outer_shell,
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
                            "    ExpansionMap: \n"
                            "      OuterBoundary: 25.0\n"
                            "      InitialExpansion: 1.0\n"
                            "      InitialExpansionVelocity: -0.1\n"
                            "      AsymptoticVelocityOuterBoundary: -0.1\n"
                            "      DecayTimescaleOuterBoundaryVelocity: 5.0\n"
                            "    RotationMap:\n"
                            "      InitialAngularVelocity: [0.0, 0.0, -0.2]\n"
                            "    SizeMap:\n"
                            "      InitialValues: [0.0, 0.0]\n"
                            "      InitialVelocities: [-0.1, -0.2]\n"
                            "      InitialAccelerations: [0.01, 0.02]"
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
         "    InnerRadius: 1.0\n"
         "    OuterRadius: 2.0\n"
         "    XCoord: 3.0\n" +
         interior_A +
         "    UseLogarithmicMap: " + stringize(use_logarithmic_map_AB) +
         "\n"
         "  ObjectB:\n"
         "    InnerRadius: 0.2\n"
         "    OuterRadius: 1.0\n"
         "    XCoord: -2.0\n" +
         interior_B +
         "    UseLogarithmicMap: " + stringize(use_logarithmic_map_AB) +
         "\n"
         "  EnvelopingCube:\n"
         "    Radius: 22.0\n"
         "    UseProjectiveMap: true\n"
         "    Sphericity: 1.0\n"
         "  OuterShell:\n"
         "    InnerRadius: Auto\n"
         "    OuterRadius: 25.0\n"
         "    RadialDistribution: Linear\n" +
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
         "    OuterShell: [1, 1, " +
         std::to_string(1 + additional_refinement_outer) +
         "]\n"
         "  InitialGridPoints: 3\n" +
         time_dependence;
}

void test_bbh_time_dependent_factory(const bool with_boundary_conditions,
                                     const bool with_control_systems) {
  INFO("BBH time dependent factory");
  CAPTURE(with_boundary_conditions);
  CAPTURE(with_control_systems);
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
  const std::array<double, 3> times_to_check{{0.0, 0.7, 1.6}};

  constexpr double initial_time = 0.0;
  constexpr double expected_time = 1.0;  // matches InitialTime: 1.0 above
  constexpr double expected_initial_function_value =
      1.0;  // hard-coded in BinaryCompactObject.cpp
  constexpr double expected_asymptotic_velocity_outer_boundary =
      -0.1;  // matches AsymptoticVelocityOuterBoundary: -0.1 above
  constexpr double expected_decay_timescale_outer_boundary_velocity =
      5.0;  // matches DecayTimescaleOuterBoundaryVelocity: 5.0 above
  std::array<DataVector, 3> expansion_factor_coefs{{{1.0}, {-0.1}, {0.0}}};
  const DataVector init_angular_vel{{0.0, 0.0, -0.2}};
  std::array<DataVector, 1> quaternion_coefs{{{1.0, 0.0, 0.0, 0.0}}};
  // Set initial angle for each axis to 0 because it doesn't matter. We don't
  // use it or care about it.
  std::array<DataVector, 4> rotation_angle_coefs{
      {{3, 0.0}, init_angular_vel, {3, 0.0}, {3, 0.0}}};
  std::array<DataVector, 4> lambda_factor_a0_coefs{
      {{0.0}, {-0.1}, {0.01}, {0.0}}};
  std::array<DataVector, 4> lambda_factor_b0_coefs{
      {{0.0}, {-0.2}, {0.02}, {0.0}}};

  // These names must match the hard coded ones in BinaryCompactObject
  const std::string size_a_name = "SizeA";
  const std::string size_b_name = "SizeB";
  const std::string expansion_name = "Expansion";
  const std::string rotation_name = "Rotation";

  ExpirationTimeMap initial_expiration_times{};
  initial_expiration_times[size_a_name] =
      with_control_systems ? 10.0 : std::numeric_limits<double>::infinity();
  initial_expiration_times[size_b_name] =
      with_control_systems ? 10.0 : std::numeric_limits<double>::infinity();
  initial_expiration_times[expansion_name] =
      with_control_systems ? 10.0 : std::numeric_limits<double>::infinity();
  initial_expiration_times[rotation_name] =
      with_control_systems ? 10.0 : std::numeric_limits<double>::infinity();

  const std::tuple<
      std::pair<std::string, domain::FunctionsOfTime::PiecewisePolynomial<3>>,
      std::pair<std::string, domain::FunctionsOfTime::PiecewisePolynomial<3>>,
      std::pair<std::string, domain::FunctionsOfTime::PiecewisePolynomial<2>>,
      std::pair<std::string, domain::FunctionsOfTime::FixedSpeedCubic>,
      std::pair<std::string,
                domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>>
      expected_functions_of_time = std::make_tuple(
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<3>>{
              size_a_name,
              {expected_time, lambda_factor_a0_coefs,
               initial_expiration_times[size_a_name]}},
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<3>>{
              size_b_name,
              {expected_time, lambda_factor_b0_coefs,
               initial_expiration_times[size_b_name]}},
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<2>>{
              expansion_name,
              {expected_time, expansion_factor_coefs,
               initial_expiration_times[expansion_name]}},
          std::pair<std::string, domain::FunctionsOfTime::FixedSpeedCubic>{
              "ExpansionOuterBoundary"s,
              {expected_initial_function_value, expected_time,
               expected_asymptotic_velocity_outer_boundary,
               expected_decay_timescale_outer_boundary_velocity}},
          std::pair<std::string,
                    domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>{
              rotation_name,
              {expected_time, quaternion_coefs, rotation_angle_coefs,
               initial_expiration_times[rotation_name]}});
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time[expansion_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, expansion_factor_coefs,
          initial_expiration_times[expansion_name]);
  functions_of_time["ExpansionOuterBoundary"] =
      std::make_unique<domain::FunctionsOfTime::FixedSpeedCubic>(
          expected_initial_function_value, initial_time,
          expected_asymptotic_velocity_outer_boundary,
          expected_decay_timescale_outer_boundary_velocity);
  functions_of_time[rotation_name] =
      std::make_unique<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>(
          initial_time, quaternion_coefs, rotation_angle_coefs,
          initial_expiration_times[rotation_name]);
  functions_of_time[size_a_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time, lambda_factor_a0_coefs,
          initial_expiration_times[size_a_name]);
  functions_of_time[size_b_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time, lambda_factor_b0_coefs,
          initial_expiration_times[size_b_name]);

  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      *binary_compact_object, with_boundary_conditions);
  for (const double time : times_to_check) {
    CAPTURE(time);
    test_det_jac_positive(domain.blocks(), time, functions_of_time);
    test_physical_separation(domain.blocks(), time, functions_of_time);
    TestHelpers::domain::creators::test_functions_of_time(
        *binary_compact_object, expected_functions_of_time,
        with_control_systems ? initial_expiration_times : ExpirationTimeMap{});
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
    TestHelpers::domain::creators::test_domain_creator(
        *binary_compact_object, with_boundary_conditions);
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
          {0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          {0.3, 1.0, -1.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, 2_st, 6_st, true, 0.0, std::nullopt, Distribution::Linear,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "The x-coordinate of ObjectA's center is expected to be positive."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.5, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          {0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, 2_st, 6_st, true, 0.0, std::nullopt, Distribution::Linear,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "The x-coordinate of ObjectB's center is expected to be negative."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.3, 1.0, 8.0, {{create_inner_boundary_condition()}}, false},
          {0.5, 1.0, -7.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, 2_st, 6_st, true, 0.0, std::nullopt, Distribution::Linear,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("The radius for the enveloping cube is too "
                                "small! The Frustums will be malformed."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          {1.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, 2_st, 6_st, true, 0.0, std::nullopt, Distribution::Linear,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "ObjectB's inner radius must be less than its outer radius."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {3.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          {0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, 2_st, 6_st, true, 0.0, std::nullopt, Distribution::Linear,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "ObjectA's inner radius must be less than its outer radius."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          {0.5, 1.0, -1.0, std::nullopt, true}, 25.5, 32.4, 2_st, 6_st, true,
          0.0, std::nullopt, Distribution::Linear,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Using a logarithmically spaced radial grid in the "
          "part of Layer 1 enveloping Object B requires excising the interior "
          "of Object B"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.3, 1.0, 1.0, std::nullopt, true},
          {0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, 2_st, 6_st, true, 0.0, std::nullopt, Distribution::Linear,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Using a logarithmically spaced radial grid in the "
          "part of Layer 1 enveloping Object A requires excising the interior "
          "of Object A"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          {0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, std::vector<std::array<size_t, 3>>{}, 6_st, true, 0.0,
          std::nullopt, Distribution::Linear, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("Invalid 'InitialRefinement'"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          {0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, 2_st, std::vector<std::array<size_t, 3>>{}, true, 0.0,
          std::nullopt, Distribution::Linear, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("Invalid 'InitialGridPoints'"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          {0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          {0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false}, 25.5,
          32.4, 2_st, 6_st, true, 0.0, 35., Distribution::Linear,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("The 'OuterShell.InnerRadius' must be within"));
  // Note: the boundary condition-related parse errors are checked in the
  // test_connectivity function.
}
}  // namespace

// [[TimeOut, 45]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObject",
                  "[Domain][Unit]") {
  test_connectivity();
  test_bbh_time_dependent_factory(true, true);
  test_bbh_time_dependent_factory(true, false);
  test_bbh_time_dependent_factory(false, true);
  test_bbh_time_dependent_factory(false, false);
  test_binary_factory();
  test_parse_errors();
}
