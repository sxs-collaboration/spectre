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
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime

namespace {
using ExpirationTimeMap = std::unordered_map<std::string, double>;
using Object = domain::creators::BinaryCompactObject<false>::Object;
using CartesianCubeAtXCoord =
    domain::creators::BinaryCompactObject<false>::CartesianCubeAtXCoord;
using Excision = domain::creators::BinaryCompactObject<false>::Excision;
using Distribution = domain::CoordinateMaps::Distribution;

template <size_t Dim, bool WithBoundaryConditions>
struct Metavariables {
  using system = tmpl::conditional_t<WithBoundaryConditions,
                                     TestHelpers::domain::BoundaryConditions::
                                         SystemWithBoundaryConditions<Dim>,
                                     TestHelpers::domain::BoundaryConditions::
                                         SystemWithoutBoundaryConditions<Dim>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<DomainCreator<3>,
                   tmpl::list<::domain::creators::BinaryCompactObject<false>>>>;
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
  MAKE_GENERATOR(gen);

  // ObjectA:
  constexpr double inner_radius_objectA = 0.3;
  constexpr double outer_radius_objectA = 1.0;
  constexpr double xcoord_objectA = 3.0;

  // ObjectB:
  constexpr double inner_radius_objectB = 0.5;
  constexpr double outer_radius_objectB = 1.0;
  constexpr double xcoord_objectB = -3.0;

  // Center of mass offset:
  constexpr std::array<double, 2> center_of_mass_offset{{0.1, 0.2}};

  // Envelope:
  constexpr double envelope_radius = 25.5;

  // Outer shell:
  constexpr double outer_radius = 32.4;

  // Cube length array with and without offsets:
  const std::array<double, 2> cube_scales = {{1.0, 1.5}};

  // Misc.:
  constexpr size_t grid_points = 3;
  constexpr bool use_equiangular_map = true;

  for (const auto& cube_scale : cube_scales) {
    for (const auto& [with_boundary_conditions, excise_interiorA,
                      excise_interiorB, opening_angle,
                      radial_distribution_envelope,
                      radial_distribution_outer_shell] :
         random_sample<5>(
             cartesian_product(
                 make_array(true, false), make_array(true, false),
                 make_array(true, false), make_array(60.0, 90.0, 120.0),
                 make_array(Distribution::Linear, Distribution::Projective,
                            Distribution::Logarithmic),
                 make_array(Distribution::Linear, Distribution::Logarithmic,
                            Distribution::Inverse)),
             make_not_null(&gen))) {
      CAPTURE(with_boundary_conditions);
      CAPTURE(excise_interiorA);
      CAPTURE(excise_interiorB);
      CAPTURE(opening_angle);
      CAPTURE(cube_scale);
      CAPTURE(inner_radius_objectA);
      CAPTURE(outer_radius_objectA);
      CAPTURE(inner_radius_objectB);
      CAPTURE(outer_radius_objectB);
      std::unordered_map<std::string, std::array<size_t, 3>> refinement{
          {"ObjectAShell", {{1, 1, 1}}},
          {"ObjectACube", {{1, 1, 1}}},
          {"ObjectBShell", {{1, 1, 1}}},
          {"ObjectBCube", {{1, 1, 1}}},
          {"Envelope", {{1, 1, 1}}},
          // Add some radial refinement in outer shell
          {"OuterShell", {{1, 1, 4}}}};
      if (not excise_interiorA) {
        refinement["ObjectAInterior"] = std::array<size_t, 3>{{1, 1, 1}};
      }
      if (not excise_interiorB) {
        refinement["ObjectBInterior"] = std::array<size_t, 3>{{1, 1, 1}};
      }
      CAPTURE(radial_distribution_outer_shell);
      CAPTURE(radial_distribution_envelope);
      const domain::creators::BinaryCompactObject binary_compact_object{
          Object{inner_radius_objectA, outer_radius_objectA, xcoord_objectA,
                 excise_interiorA ? std::make_optional(Excision{
                                        with_boundary_conditions
                                            ? create_inner_boundary_condition()
                                            : nullptr})
                                  : std::nullopt,
                 false},
          Object{inner_radius_objectB, outer_radius_objectB, xcoord_objectB,
                 excise_interiorB ? std::make_optional(Excision{
                                        with_boundary_conditions
                                            ? create_inner_boundary_condition()
                                            : nullptr})
                                  : std::nullopt,
                 false},
          center_of_mass_offset,
          envelope_radius,
          outer_radius,
          (excise_interiorA and excise_interiorB and opening_angle == 90)
              ? cube_scale
              : cube_scales[0],
          refinement,
          grid_points,
          use_equiangular_map,
          radial_distribution_envelope,
          radial_distribution_outer_shell,
          opening_angle,
          std::nullopt,
          with_boundary_conditions ? create_outer_boundary_condition()
                                   : nullptr};

      const auto domain = TestHelpers::domain::creators::test_domain_creator(
          binary_compact_object, with_boundary_conditions);

      std::vector<std::string> expected_block_names{
          "ObjectAShellUpperZ",   "ObjectAShellLowerZ",
          "ObjectAShellUpperY",   "ObjectAShellLowerY",
          "ObjectAShellUpperX",   "ObjectAShellLowerX",
          "ObjectACubeUpperZ",    "ObjectACubeLowerZ",
          "ObjectACubeUpperY",    "ObjectACubeLowerY",
          "ObjectACubeUpperX",    "ObjectACubeLowerX",
          "ObjectBShellUpperZ",   "ObjectBShellLowerZ",
          "ObjectBShellUpperY",   "ObjectBShellLowerY",
          "ObjectBShellUpperX",   "ObjectBShellLowerX",
          "ObjectBCubeUpperZ",    "ObjectBCubeLowerZ",
          "ObjectBCubeUpperY",    "ObjectBCubeLowerY",
          "ObjectBCubeUpperX",    "ObjectBCubeLowerX",
          "EnvelopeUpperZLeft",   "EnvelopeUpperZRight",
          "EnvelopeLowerZLeft",   "EnvelopeLowerZRight",
          "EnvelopeUpperYLeft",   "EnvelopeUpperYRight",
          "EnvelopeLowerYLeft",   "EnvelopeLowerYRight",
          "EnvelopeUpperX",       "EnvelopeLowerX",
          "OuterShellUpperZLeft", "OuterShellUpperZRight",
          "OuterShellLowerZLeft", "OuterShellLowerZRight",
          "OuterShellUpperYLeft", "OuterShellUpperYRight",
          "OuterShellLowerYLeft", "OuterShellLowerYRight",
          "OuterShellUpperX",     "OuterShellLowerX"};
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
               {"ObjectACubeLowerY", "ObjectACubeLowerZ", "ObjectACubeUpperY",
                "ObjectACubeUpperX", "ObjectACubeUpperZ", "ObjectACubeLowerX"}},
              {"ObjectBCube",
               {"ObjectBCubeLowerY", "ObjectBCubeLowerZ", "ObjectBCubeUpperY",
                "ObjectBCubeUpperX", "ObjectBCubeUpperZ", "ObjectBCubeLowerX"}},
              {"Envelope",
               {"EnvelopeUpperZRight", "EnvelopeUpperX", "EnvelopeLowerZLeft",
                "EnvelopeLowerZRight", "EnvelopeUpperYRight",
                "EnvelopeLowerYRight", "EnvelopeUpperYLeft",
                "EnvelopeUpperZLeft", "EnvelopeLowerYLeft", "EnvelopeLowerX"}},
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
            "ExcisionSphereA",
            ExcisionSphere<3>{inner_radius_objectA,
                              tnsr::I<double, 3, Frame::Grid>{
                                  {xcoord_objectA, center_of_mass_offset[0],
                                   center_of_mass_offset[1]}},
                              {{0, Direction<3>::lower_zeta()},
                               {1, Direction<3>::lower_zeta()},
                               {2, Direction<3>::lower_zeta()},
                               {3, Direction<3>::lower_zeta()},
                               {4, Direction<3>::lower_zeta()},
                               {5, Direction<3>::lower_zeta()}}});
      }
      if (excise_interiorB) {
        expected_excision_spheres.emplace(
            "ExcisionSphereB",
            ExcisionSphere<3>{inner_radius_objectB,
                              tnsr::I<double, 3, Frame::Grid>{
                                  {xcoord_objectB, center_of_mass_offset[0],
                                   center_of_mass_offset[1]}},
                              {{12, Direction<3>::lower_zeta()},
                               {13, Direction<3>::lower_zeta()},
                               {14, Direction<3>::lower_zeta()},
                               {15, Direction<3>::lower_zeta()},
                               {16, Direction<3>::lower_zeta()},
                               {17, Direction<3>::lower_zeta()}}});
      }
      CHECK(domain.excision_spheres() == expected_excision_spheres);

      if (with_boundary_conditions) {
        using PeriodicBc = TestHelpers::domain::BoundaryConditions::
            TestPeriodicBoundaryCondition<3>;
        CHECK_THROWS_WITH(
            domain::creators::BinaryCompactObject(
                Object{
                    inner_radius_objectA, outer_radius_objectA, xcoord_objectA,
                    excise_interiorA ? std::make_optional(Excision{
                                           create_inner_boundary_condition()})
                                     : std::nullopt,
                    false},
                domain::creators::BinaryCompactObject<false>::Object{
                    inner_radius_objectB, outer_radius_objectB, xcoord_objectB,
                    excise_interiorB ? std::make_optional(Excision{
                                           create_inner_boundary_condition()})
                                     : std::nullopt,
                    false},
                center_of_mass_offset, envelope_radius, outer_radius,
                (excise_interiorA and excise_interiorB) ? cube_scale
                                                        : cube_scales[0],
                refinement, grid_points, use_equiangular_map,
                radial_distribution_envelope, radial_distribution_outer_shell,
                opening_angle, std::nullopt, std::make_unique<PeriodicBc>(),
                Options::Context{false, {}, 1, 1}),
            Catch::Matchers::ContainsSubstring(
                "Cannot have periodic boundary "
                "conditions with a binary domain"));
        if (excise_interiorA or excise_interiorB) {
          CHECK_THROWS_WITH(
              domain::creators::BinaryCompactObject(
                  Object{inner_radius_objectA, outer_radius_objectA,
                         xcoord_objectA,
                         excise_interiorA ? std::make_optional(Excision{
                                                std::make_unique<PeriodicBc>()})
                                          : std::nullopt,
                         false},
                  Object{inner_radius_objectB, outer_radius_objectB,
                         xcoord_objectB,
                         excise_interiorB ? std::make_optional(Excision{
                                                std::make_unique<PeriodicBc>()})
                                          : std::nullopt,
                         false},
                  center_of_mass_offset, envelope_radius, outer_radius,
                  (excise_interiorA and excise_interiorB) ? cube_scale
                                                          : cube_scales[0],
                  refinement, grid_points, use_equiangular_map,
                  radial_distribution_envelope, radial_distribution_outer_shell,
                  opening_angle, std::nullopt,
                  create_outer_boundary_condition(),
                  Options::Context{false, {}, 1, 1}),
              Catch::Matchers::ContainsSubstring(
                  "Cannot have periodic boundary "
                  "conditions with a binary domain"));
          CHECK_THROWS_WITH(
              domain::creators::BinaryCompactObject(
                  Object{inner_radius_objectA, outer_radius_objectA,
                         xcoord_objectA,
                         excise_interiorA
                             ? std::make_optional(
                                   Excision{create_inner_boundary_condition()})
                             : std::nullopt,
                         false},
                  Object{inner_radius_objectB, outer_radius_objectB,
                         xcoord_objectB,
                         excise_interiorB
                             ? std::make_optional(
                                   Excision{create_inner_boundary_condition()})
                             : std::nullopt,
                         false},
                  center_of_mass_offset, envelope_radius, outer_radius,
                  (excise_interiorA and excise_interiorB) ? cube_scale
                                                          : cube_scales[0],
                  refinement, grid_points, use_equiangular_map,
                  radial_distribution_envelope, radial_distribution_outer_shell,
                  opening_angle, std::nullopt, nullptr,
                  Options::Context{false, {}, 1, 1}),
              Catch::Matchers::ContainsSubstring(
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
                  center_of_mass_offset, envelope_radius, outer_radius,
                  (excise_interiorA and excise_interiorB) ? cube_scale
                                                          : cube_scales[0],
                  refinement, grid_points, use_equiangular_map,
                  radial_distribution_envelope, radial_distribution_outer_shell,
                  opening_angle, std::nullopt,
                  create_outer_boundary_condition(),
                  Options::Context{false, {}, 1, 1}),
              Catch::Matchers::ContainsSubstring(
                  "Must specify either both inner and outer boundary "
                  "conditions or neither."));
        }
      }
    }
  }
}

std::string stringize(const bool t) { return t ? "true" : "false"; }

std::string create_option_string(
    const bool excise_A, const bool excise_B, const bool add_time_dependence,
    const bool use_logarithmic_map_AB, const bool use_equiangular_map,
    const size_t additional_refinement_outer,
    const size_t additional_refinement_A, const size_t additional_refinement_B,
    const double opening_angle, const bool add_boundary_condition) {
  const std::string cube_length =
      (excise_A and excise_B and opening_angle == 90) ? "1.5" : "1.0";
  const std::string time_dependence{
      add_time_dependence
          ? "  TimeDependentMaps:\n"
            "    InitialTime: 1.0\n"
            "    ExpansionMap: \n"
            "      InitialValues: [1.0, -0.1]\n"
            "      AsymptoticVelocityOuterBoundary: -0.1\n"
            "      DecayTimescaleOuterBoundaryVelocity: 5.0\n"
            "    RotationMap:\n"
            "      InitialAngularVelocity: [0.0, 0.0, -0.2]\n"
            "    TranslationMap:\n"
            "      InitialValues: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], "
            "      [0.0, 0.0, 0.0]]\n"s +
                (excise_A ? "    ShapeMapA:\n"
                            "      LMax: 8\n"
                            "      InitialValues: Spherical\n"
                            "      SizeInitialValues: [0.0, -0.1, 0.01]\n"
                            "      TransitionEndsAtCube: false\n"s
                          : "    ShapeMapA: None\n"s) +
                (excise_B ? "    ShapeMapB:\n"
                            "      LMax: 8\n"
                            "      InitialValues: Spherical\n"
                            "      SizeInitialValues: [0.0, -0.2, 0.02]\n"
                            "      TransitionEndsAtCube: true"s
                          : "    ShapeMapB: None"s)
          : "  TimeDependentMaps: None"};
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
         "  CenterOfMassOffset: [0.1, 0.2]\n"
         "  Envelope:\n"
         "    Radius: 22.0\n"
         "    RadialDistribution: Projective\n"
         "  OuterShell:\n"
         "    Radius: 25.0\n"
         "    RadialDistribution: Linear\n" +
         "    OpeningAngle: " + std::to_string(opening_angle) + "\n" +
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
         "    Envelope: [1, 1, 1]\n"
         "    OuterShell: [1, 1, " +
         std::to_string(1 + additional_refinement_outer) +
         "]\n"
         "  InitialGridPoints: 3\n" +
         "  CubeScale: " + cube_length +
         "\n"
         "  UseEquiangularMap: " +
         stringize(use_equiangular_map) + "\n" + time_dependence;
}

void test_bns_domain_with_cubes() {
  INFO("BNS domain with cubes");

  MAKE_GENERATOR(gen);

  // ObjectA:
  constexpr double xcoord_objectA = 3.0;

  // ObjectB:
  constexpr double xcoord_objectB = -3.0;

  // CoM offset:
  constexpr std::array<double, 2> center_of_mass_offset{{0.1, 0.2}};

  // Envelope:
  constexpr double envelope_radius = 25.5;
  constexpr auto radial_distribution_envelope = Distribution::Projective;

  // Outer shell:
  constexpr double outer_radius = 32.4;
  constexpr double opening_angle = 90.0;

  // Cube length array with and without offsets:
  const std::array<double, 2> cube_scales = {{1.0, 1.5}};

  // Misc.:
  constexpr size_t grid_points = 3;
  constexpr bool use_equiangular_map = false;
  const auto radial_distribution_outer_shell = Distribution::Inverse;
  constexpr bool with_boundary_conditions = true;

  std::unordered_map<std::string, std::array<size_t, 3>> refinement{
      {"ObjectA", {{1, 1, 1}}},
      {"ObjectB", {{1, 1, 1}}},
      {"Envelope", {{1, 1, 1}}},
      {"OuterShell", {{1, 1, 4}}}};

  for (const auto cube_scale : cube_scales) {
    const domain::creators::BinaryCompactObject binary_compact_object{
        CartesianCubeAtXCoord{xcoord_objectA},
        CartesianCubeAtXCoord{xcoord_objectB},
        center_of_mass_offset,
        envelope_radius,
        outer_radius,
        cube_scale,
        refinement,
        grid_points,
        use_equiangular_map,
        radial_distribution_envelope,
        radial_distribution_outer_shell,
        opening_angle,
        std::nullopt,
        create_outer_boundary_condition()};

    const auto domain = TestHelpers::domain::creators::test_domain_creator(
        binary_compact_object, with_boundary_conditions);

    std::vector<std::string> expected_block_names{"ObjectA",
                                                  "ObjectB",
                                                  "EnvelopeUpperZLeft",
                                                  "EnvelopeUpperZRight",
                                                  "EnvelopeLowerZLeft",
                                                  "EnvelopeLowerZRight",
                                                  "EnvelopeUpperYLeft",
                                                  "EnvelopeUpperYRight",
                                                  "EnvelopeLowerYLeft",
                                                  "EnvelopeLowerYRight",
                                                  "EnvelopeUpperX",
                                                  "EnvelopeLowerX",
                                                  "OuterShellUpperZLeft",
                                                  "OuterShellUpperZRight",
                                                  "OuterShellLowerZLeft",
                                                  "OuterShellLowerZRight",
                                                  "OuterShellUpperYLeft",
                                                  "OuterShellUpperYRight",
                                                  "OuterShellLowerYLeft",
                                                  "OuterShellLowerYRight",
                                                  "OuterShellUpperX",
                                                  "OuterShellLowerX"};
    std::unordered_map<std::string, std::unordered_set<std::string>>
        expected_block_groups{
            {"Envelope",
             {"EnvelopeUpperZRight", "EnvelopeUpperX", "EnvelopeLowerZLeft",
              "EnvelopeLowerZRight", "EnvelopeUpperYRight",
              "EnvelopeLowerYRight", "EnvelopeUpperYLeft", "EnvelopeUpperZLeft",
              "EnvelopeLowerYLeft", "EnvelopeLowerX"}},
            {"OuterShell",
             {"OuterShellUpperZLeft", "OuterShellUpperZRight",
              "OuterShellLowerZLeft", "OuterShellLowerZRight",
              "OuterShellUpperYLeft", "OuterShellUpperYRight",
              "OuterShellLowerYLeft", "OuterShellLowerYRight",
              "OuterShellUpperX", "OuterShellLowerX"}}};
    CHECK(binary_compact_object.block_names() == expected_block_names);
    CHECK(binary_compact_object.block_groups() == expected_block_groups);
  }
}

void test_bbh_time_dependent_factory(const bool with_boundary_conditions,
                                     const bool with_time_dependence,
                                     const bool excise_B) {
  INFO("BBH time dependent factory");
  CAPTURE(with_boundary_conditions);
  CAPTURE(with_time_dependence);
  CAPTURE(excise_B);
  const auto binary_compact_object = [&with_boundary_conditions,
                                      &with_time_dependence, &excise_B]() {
    if (with_boundary_conditions) {
      return TestHelpers::test_option_tag<domain::OptionTags::DomainCreator<3>,
                                          Metavariables<3, true>>(
          create_option_string(true, excise_B, with_time_dependence, false,
                               true, 0, 0, 0, 120.0, with_boundary_conditions));
    } else {
      return TestHelpers::test_option_tag<domain::OptionTags::DomainCreator<3>,
                                          Metavariables<3, false>>(
          create_option_string(true, excise_B, with_time_dependence, false,
                               true, 0, 0, 0, 120.0, with_boundary_conditions));
    }
  }();

  const std::vector<double> times_to_check{{1., 2.3}};
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      *binary_compact_object, with_boundary_conditions, false, times_to_check);

  const auto& blocks = domain.blocks();
  const auto& final_envelope_block = excise_B ? blocks[33] : blocks[21];

  std::unordered_map<std::string, ExcisionSphere<3>>
      expected_excision_spheres{};
  expected_excision_spheres.emplace(
      "ExcisionSphereA",
      ExcisionSphere<3>{1.0,
                        tnsr::I<double, 3, Frame::Grid>{{3.0, 0.1, 0.2}},
                        {{0, Direction<3>::lower_zeta()},
                         {1, Direction<3>::lower_zeta()},
                         {2, Direction<3>::lower_zeta()},
                         {3, Direction<3>::lower_zeta()},
                         {4, Direction<3>::lower_zeta()},
                         {5, Direction<3>::lower_zeta()}}});
  if (with_time_dependence) {
    expected_excision_spheres.at("ExcisionSphereA")
        .inject_time_dependent_maps(
            final_envelope_block.moving_mesh_grid_to_inertial_map()
                .get_clone());
  }
  if (excise_B) {
    expected_excision_spheres.emplace(
        "ExcisionSphereB",
        ExcisionSphere<3>{0.2,
                          tnsr::I<double, 3, Frame::Grid>{{-2.0, 0.1, 0.2}},
                          {{12, Direction<3>::lower_zeta()},
                           {13, Direction<3>::lower_zeta()},
                           {14, Direction<3>::lower_zeta()},
                           {15, Direction<3>::lower_zeta()},
                           {16, Direction<3>::lower_zeta()},
                           {17, Direction<3>::lower_zeta()}}});
    if (with_time_dependence) {
      expected_excision_spheres.at("ExcisionSphereB")
          .inject_time_dependent_maps(
              final_envelope_block.moving_mesh_grid_to_inertial_map()
                  .get_clone());
    }
  }

  const auto& excision_spheres = domain.excision_spheres();
  CHECK(excision_spheres == expected_excision_spheres);

  const auto check_excision_sphere_map =
      [&binary_compact_object](const ExcisionSphere<3>& excision_sphere) {
        const double time = 1.0;
        // Taken from option string above
        const auto functions_of_time =
            binary_compact_object->functions_of_time();
        const auto& center = excision_sphere.center();
        const auto& map = excision_sphere.moving_mesh_grid_to_inertial_map();

        const auto mapped_point = map(center, time, functions_of_time);

        // Should be same point
        for (size_t i = 0; i < 3; i++) {
          CHECK(center.get(i) == approx(mapped_point.get(i)));
        }
      };

  if (with_time_dependence) {
    check_excision_sphere_map(excision_spheres.at("ExcisionSphereA"));
    if (excise_B) {
      check_excision_sphere_map(excision_spheres.at("ExcisionSphereB"));
    }
  }
}

void test_binary_factory() {
  MAKE_GENERATOR(gen);
  const auto check_impl = [](const std::string& opt_string,
                             const bool with_boundary_conditions) {
    const auto binary_compact_object = [&opt_string,
                                        &with_boundary_conditions]() {
      if (with_boundary_conditions) {
        return TestHelpers::test_option_tag<
            domain::OptionTags::DomainCreator<3>, Metavariables<3, true>>(
            opt_string);
      } else {
        return TestHelpers::test_option_tag<
            domain::OptionTags::DomainCreator<3>, Metavariables<3, false>>(
            opt_string);
      }
    }();
    TestHelpers::domain::creators::test_domain_creator(
        *binary_compact_object, with_boundary_conditions);
  };
  for (const auto& [excise_A, excise_B, use_log_maps, use_equiangular_map,
                    additional_refinement_outer, additional_refinement_A,
                    additional_refinement_B, opening_angle,
                    with_boundary_conds] :
       random_sample<5>(
           cartesian_product(make_array(true, false), make_array(true, false),
                             make_array(true, false), make_array(true, false),
                             make_array(0_st, 1_st), make_array(0_st, 2_st),
                             make_array(0_st, 3_st),
                             make_array(60.0, 90.0, 120.0),
                             make_array(true, false)),
           make_not_null(&gen))) {
    if (use_log_maps and not(excise_A and excise_B)) {
      // Log maps in the object interiors only work with excisions
      continue;
    }
    check_impl(create_option_string(
                   excise_A, excise_B, false, use_log_maps,
                   opening_angle == 90.0 ? use_equiangular_map : true,
                   additional_refinement_outer, additional_refinement_A,
                   additional_refinement_B, opening_angle, with_boundary_conds),
               with_boundary_conds);
  }
}

void test_parse_errors() {
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          Object{0.3, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, Distribution::Linear, 120.0, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "The x-coordinate of ObjectA's center is expected to be positive."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.5, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          Object{0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, Distribution::Linear, 120.0, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "The x-coordinate of ObjectB's center is expected to be negative."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 1.0, 8.0, {{create_inner_boundary_condition()}}, false},
          Object{0.5, 1.0, -7.0, {{create_inner_boundary_condition()}}, false},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, Distribution::Linear, 120.0, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "The radius for the enveloping cube is too "
          "small! The Frustums will be malformed."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 1.0, 8.0, {{create_inner_boundary_condition()}}, false},
          Object{0.5, 1.0, -7.0, {{create_inner_boundary_condition()}}, false},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 0.5, 2_st, 6_st, true,
          Distribution::Projective, Distribution::Linear, 120.0, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "The cube length should be greater than or equal to the initial "
          "separation between the two objects."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          Object{1.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, Distribution::Linear, 120.0, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "ObjectB's inner radius must be less than its outer radius."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{3.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          Object{0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, Distribution::Linear, 120.0, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "ObjectA's inner radius must be less than its outer radius."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          Object{0.5, 1.0, -1.0, std::nullopt, true},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, true, 6_st,
          Distribution::Projective, Distribution::Linear, 120.0, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Using a logarithmically spaced radial grid in the "
          "part of Layer 1 enveloping Object B requires excising the interior "
          "of Object B"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 1.0, 1.0, std::nullopt, true},
          Object{0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, Distribution::Linear, 120.0, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Using a logarithmically spaced radial grid in the "
          "part of Layer 1 enveloping Object A requires excising the interior "
          "of Object A"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          Object{0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0,
          std::vector<std::array<size_t, 3>>{}, 6_st, true,
          Distribution::Projective, Distribution::Linear, 120.0, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("Invalid 'InitialRefinement'"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          Object{0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st,
          std::vector<std::array<size_t, 3>>{}, true, Distribution::Projective,
          Distribution::Linear, 120.0, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("Invalid 'InitialGridPoints'"));
  // Note: the boundary condition-related parse errors are checked in the
  // test_connectivity function.
}

void test_kerr_horizon_conforming() {
  INFO(
      "Check that inner radius is deformed to constant Boyer-Lindquist radius");
  const double mass_A = 0.8;
  const double mass_B = 1.2;
  const std::array<double, 3> spin_A{{0.0, 0.0, 0.9}};
  const std::array<double, 3> spin_B{{0.0, 0.2, 0.4}};
  const double r_plus_A = mass_A * (1. + sqrt(1. - dot(spin_A, spin_A)));
  const double r_plus_B = mass_B * (1. + sqrt(1. - dot(spin_B, spin_B)));
  const double inner_radius_A = r_plus_A;
  const double inner_radius_B = 0.89 * r_plus_B;
  const double x_pos_A = 8;
  const double x_pos_B = -8;
  const double y_offset = 0.1;
  const double z_offset = 0.2;
  domain::creators::BinaryCompactObject domain_creator{
      Object{inner_radius_A, 4., x_pos_A, true, true},
      Object{inner_radius_B, 4., x_pos_B, true, true},
      std::array<double, 2>{{0.1, 0.2}},
      40.,
      200.,
      1.0,
      0_st,
      6_st,
      true,
      Distribution::Projective,
      Distribution::Inverse,
      120.,
      domain::creators::bco::TimeDependentMapOptions<false>{
          0.,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          {{32_st,
            domain::creators::time_dependent_options::
                KerrSchildFromBoyerLindquist{mass_A, spin_A},
            std::nullopt}},
          {{32_st,
            domain::creators::time_dependent_options::
                KerrSchildFromBoyerLindquist{mass_B, spin_B},
            std::nullopt}}}};
  const auto domain = domain_creator.create_domain();
  const auto functions_of_time = domain_creator.functions_of_time();
  // Set up coordinates on an ellipsoid of constant Boyer-Lindquist radius
  const size_t num_points = 10;
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<double> dist_phi{0., 2. * M_PI};
  std::uniform_real_distribution<double> dist_theta{0., M_PI};
  const std::array<DataVector, 2> theta_phi{
      {make_with_random_values<DataVector>(
           make_not_null(&gen), make_not_null(&dist_theta), num_points),
       make_with_random_values<DataVector>(
           make_not_null(&gen), make_not_null(&dist_phi), num_points)}};
  const auto radius_A =
      get(gr::Solutions::kerr_schild_radius_from_boyer_lindquist(
          inner_radius_A, theta_phi, mass_A, spin_A));
  const auto radius_B =
      get(gr::Solutions::kerr_schild_radius_from_boyer_lindquist(
          inner_radius_B, theta_phi, mass_B, spin_B));
  tnsr::I<DataVector, 3> x_A{};
  tnsr::I<DataVector, 3> x_B{};
  get<0>(x_A) =
      x_pos_A + radius_A * sin(get<0>(theta_phi)) * cos(get<1>(theta_phi));
  get<1>(x_A) =
      y_offset + radius_A * sin(get<0>(theta_phi)) * sin(get<1>(theta_phi));
  get<2>(x_A) = z_offset + radius_A * cos(get<0>(theta_phi));
  get<0>(x_B) =
      x_pos_B + radius_B * sin(get<0>(theta_phi)) * cos(get<1>(theta_phi));
  get<1>(x_B) =
      y_offset + radius_B * sin(get<0>(theta_phi)) * sin(get<1>(theta_phi));
  get<2>(x_B) = z_offset + radius_B * cos(get<0>(theta_phi));
  // Map the coordinates through the domain. They should lie at the lower zeta
  // boundary of their block.
  const auto x_logical_A =
      block_logical_coordinates(domain, x_A, 0., functions_of_time);
  const auto x_logical_B =
      block_logical_coordinates(domain, x_B, 0., functions_of_time);
  for (size_t i = 0; i < num_points; ++i) {
    {
      CAPTURE(x_logical_A[i]);
      REQUIRE(x_logical_A[i].has_value());
      CHECK(get<2>(x_logical_A[i]->data) == approx(-1.));
    }
    {
      CAPTURE(x_logical_B[i]);
      REQUIRE(x_logical_B[i].has_value());
      CHECK(get<2>(x_logical_B[i]->data) == approx(-1.));
    }
  }
}

}  // namespace

// [[TimeOut, 30]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObject",
                  "[Domain][Unit]") {
  test_connectivity();
  test_bns_domain_with_cubes();
  for (const auto& [with_bc, add_time_dep, excise_B] :
       cartesian_product(make_array(true, false), make_array(true, false),
                         make_array(true, false))) {
    test_bbh_time_dependent_factory(with_bc, add_time_dep, excise_B);
  }
  test_binary_factory();
  test_parse_errors();
  test_kerr_horizon_conforming();
}
