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
#include <variant>

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
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrHorizon.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime

namespace {
using ExpirationTimeMap = std::unordered_map<std::string, double>;
using Object = domain::creators::BinaryCompactObject::Object;
using CartesianCubeAtXCoord =
    domain::creators::BinaryCompactObject::CartesianCubeAtXCoord;
using Excision = domain::creators::BinaryCompactObject::Excision;
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
      std::variant<Distribution, std::vector<Distribution>>
          variant_radial_distribution_outer_shells =
              radial_distribution_outer_shell;
      if (radial_distribution_outer_shell == Distribution::Linear) {
        variant_radial_distribution_outer_shells = std::vector{
            radial_distribution_outer_shell, radial_distribution_outer_shell};
      }
      const bool multiple_outer_shells =
          std::holds_alternative<std::vector<Distribution>>(
              variant_radial_distribution_outer_shells);
      const std::vector<double> radial_partitioning_outer_shells =
          multiple_outer_shells
              ? std::vector<double>{0.5 * (envelope_radius + outer_radius)}
              : std::vector<double>{};
      std::unordered_map<std::string, std::array<size_t, 3>> refinement{
          {"ObjectAShell", {{1, 1, 1}}},
          {"ObjectACube", {{1, 1, 1}}},
          {"ObjectBShell", {{1, 1, 1}}},
          {"ObjectBCube", {{1, 1, 1}}},
          {"Envelope", {{1, 1, 1}}},
          // Add some radial refinement in outer shell
          {"OuterShell0", {{1, 1, 4}}}};
      if (multiple_outer_shells) {
        refinement["OuterShell1"] = std::array{1_st, 1_st, 2_st};
      }
      if (not excise_interiorA) {
        refinement["ObjectAInterior"] = std::array<size_t, 3>{{1, 1, 1}};
      }
      if (not excise_interiorB) {
        refinement["ObjectBInterior"] = std::array<size_t, 3>{{1, 1, 1}};
      }
      CAPTURE(radial_partitioning_outer_shells);
      CAPTURE(variant_radial_distribution_outer_shells);
      CAPTURE(radial_distribution_envelope);
      const domain::creators::BinaryCompactObject binary_compact_object{
          Object{inner_radius_objectA, outer_radius_objectA, xcoord_objectA,
                 excise_interiorA ? std::make_optional(Excision{
                                        with_boundary_conditions
                                            ? create_inner_boundary_condition()
                                            : nullptr})
                                  : std::nullopt,
                 std::nullopt},
          Object{inner_radius_objectB, outer_radius_objectB, xcoord_objectB,
                 excise_interiorB ? std::make_optional(Excision{
                                        with_boundary_conditions
                                            ? create_inner_boundary_condition()
                                            : nullptr})
                                  : std::nullopt,
                 std::nullopt},
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
          radial_partitioning_outer_shells,
          variant_radial_distribution_outer_shells,
          opening_angle,
          false,
          false,
          std::nullopt,
          with_boundary_conditions ? create_outer_boundary_condition()
                                   : nullptr};

      const auto domain = TestHelpers::domain::creators::test_domain_creator(
          binary_compact_object, with_boundary_conditions);

      std::vector<std::string> expected_block_names{
          "ObjectAShellUpperZ",    "ObjectAShellLowerZ",
          "ObjectAShellUpperY",    "ObjectAShellLowerY",
          "ObjectAShellUpperX",    "ObjectAShellLowerX",
          "ObjectACubeUpperZ",     "ObjectACubeLowerZ",
          "ObjectACubeUpperY",     "ObjectACubeLowerY",
          "ObjectACubeUpperX",     "ObjectACubeLowerX",
          "ObjectBShellUpperZ",    "ObjectBShellLowerZ",
          "ObjectBShellUpperY",    "ObjectBShellLowerY",
          "ObjectBShellUpperX",    "ObjectBShellLowerX",
          "ObjectBCubeUpperZ",     "ObjectBCubeLowerZ",
          "ObjectBCubeUpperY",     "ObjectBCubeLowerY",
          "ObjectBCubeUpperX",     "ObjectBCubeLowerX",
          "EnvelopeUpperZLeft",    "EnvelopeUpperZRight",
          "EnvelopeLowerZLeft",    "EnvelopeLowerZRight",
          "EnvelopeUpperYLeft",    "EnvelopeUpperYRight",
          "EnvelopeLowerYLeft",    "EnvelopeLowerYRight",
          "EnvelopeUpperX",        "EnvelopeLowerX",
          "OuterShell0UpperZLeft", "OuterShell0UpperZRight",
          "OuterShell0LowerZLeft", "OuterShell0LowerZRight",
          "OuterShell0UpperYLeft", "OuterShell0UpperYRight",
          "OuterShell0LowerYLeft", "OuterShell0LowerYRight",
          "OuterShell0UpperX",     "OuterShell0LowerX"};
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
              {"OuterShell0",
               {"OuterShell0UpperZLeft", "OuterShell0UpperZRight",
                "OuterShell0LowerZLeft", "OuterShell0LowerZRight",
                "OuterShell0UpperYLeft", "OuterShell0UpperYRight",
                "OuterShell0LowerYLeft", "OuterShell0LowerYRight",
                "OuterShell0UpperX", "OuterShell0LowerX"}}};
      if (multiple_outer_shells) {
        const std::vector<std::string> shell_1_names = {
            "OuterShell1UpperZLeft", "OuterShell1UpperZRight",
            "OuterShell1LowerZLeft", "OuterShell1LowerZRight",
            "OuterShell1UpperYLeft", "OuterShell1UpperYRight",
            "OuterShell1LowerYLeft", "OuterShell1LowerYRight",
            "OuterShell1UpperX",     "OuterShell1LowerX"};
        expected_block_groups["OuterShell1"] = std::unordered_set<std::string>{
            shell_1_names.begin(), shell_1_names.end()};
        expected_block_names.insert(expected_block_names.end(),
                                    shell_1_names.begin(), shell_1_names.end());
      }
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
                    std::nullopt},
                domain::creators::BinaryCompactObject::Object{
                    inner_radius_objectB, outer_radius_objectB, xcoord_objectB,
                    excise_interiorB ? std::make_optional(Excision{
                                           create_inner_boundary_condition()})
                                     : std::nullopt,
                    std::nullopt},
                center_of_mass_offset, envelope_radius, outer_radius,
                (excise_interiorA and excise_interiorB) ? cube_scale
                                                        : cube_scales[0],
                refinement, grid_points, use_equiangular_map,
                radial_distribution_envelope, radial_partitioning_outer_shells,
                variant_radial_distribution_outer_shells, opening_angle, false,
                false, std::nullopt, std::make_unique<PeriodicBc>(),
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
                         std::nullopt},
                  Object{inner_radius_objectB, outer_radius_objectB,
                         xcoord_objectB,
                         excise_interiorB ? std::make_optional(Excision{
                                                std::make_unique<PeriodicBc>()})
                                          : std::nullopt,
                         std::nullopt},
                  center_of_mass_offset, envelope_radius, outer_radius,
                  (excise_interiorA and excise_interiorB) ? cube_scale
                                                          : cube_scales[0],
                  refinement, grid_points, use_equiangular_map,
                  radial_distribution_envelope,
                  radial_partitioning_outer_shells,
                  variant_radial_distribution_outer_shells, opening_angle,
                  false, false, std::nullopt, create_outer_boundary_condition(),
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
                         std::nullopt},
                  Object{inner_radius_objectB, outer_radius_objectB,
                         xcoord_objectB,
                         excise_interiorB
                             ? std::make_optional(
                                   Excision{create_inner_boundary_condition()})
                             : std::nullopt,
                         std::nullopt},
                  center_of_mass_offset, envelope_radius, outer_radius,
                  (excise_interiorA and excise_interiorB) ? cube_scale
                                                          : cube_scales[0],
                  refinement, grid_points, use_equiangular_map,
                  radial_distribution_envelope,
                  radial_partitioning_outer_shells,
                  variant_radial_distribution_outer_shells, opening_angle,
                  false, false, std::nullopt, nullptr,
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
                         std::nullopt},
                  Object{inner_radius_objectB, outer_radius_objectB,
                         xcoord_objectB,
                         excise_interiorB
                             ? std::make_optional(Excision{nullptr})
                             : std::nullopt,
                         std::nullopt},
                  center_of_mass_offset, envelope_radius, outer_radius,
                  (excise_interiorA and excise_interiorB) ? cube_scale
                                                          : cube_scales[0],
                  refinement, grid_points, use_equiangular_map,
                  radial_distribution_envelope,
                  radial_partitioning_outer_shells,
                  variant_radial_distribution_outer_shells, opening_angle,
                  false, false, std::nullopt, create_outer_boundary_condition(),
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
    const double opening_angle, const bool add_boundary_condition,
    const bool use_spherical_harmonics_shell_a = false,
    const bool use_spherical_harmonics_shell_b = false) {
  const std::string cube_scale =
      (excise_A and excise_B and opening_angle == 90) ? "1.5" : "1.0";
  const std::string time_dependence{
      add_time_dependence
          ? ("  TimeDependentMaps:\n"
             "    GridCenters:\n"
             "      ScaleInspiralRateBy: 0.8\n"
             "      SpecEvolutionParametersPerlFile: "s +
             unit_test_src_path() +
             "/../InputFiles/GrMhd/GhValenciaDivClean/"
             "EvolutionParameters.perl\n"
             "    InitialTime: 1.0\n"
             "    ExpansionMap: \n"
             "      InitialValues: [1.0, -0.1, 0.0]\n"
             "      AsymptoticVelocityOuterBoundary: -0.1\n"
             "      DecayTimescaleOuterBoundary: 5.0\n"
             "    RotationMap:\n"
             "      InitialAngularVelocity: [0.0, 0.0, -0.2]\n"
             "    TranslationMap:\n"
             "      InitialValues: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], "
             "      [0.0, 0.0, 0.0]]\n"s
             "    SkewMap:\n"
             "      InitialValuesY: [0.0, -0.01, 0.0]\n"
             "      InitialValuesZ: [0.0, -0.01, 0.0]\n" +
             (excise_A ? "    ShapeMapA:\n"
                         "      LMax: 8\n"
                         "      CoefficientTruncationLimit: 0.\n"
                         "      InitialValues: Spherical\n"
                         "      SizeInitialValues: [0.0, -0.1, 0.01]\n"
                         "      TransitionEndsAtCube: false\n"s
                       : "    ShapeMapA: None\n"s) +
             (excise_B
                  ? "    ShapeMapB:\n"
                    "      LMax: 8\n"
                    "      CoefficientTruncationLimit: 0.\n"
                    "      InitialValues: Spherical\n"
                    "      SizeInitialValues: [0.0, -0.2, 0.02]\n"
                    // a spherical-harmonic object shell needs the shape
                    // transition to end at the sphere
                    "      TransitionEndsAtCube: "s +
                        (use_spherical_harmonics_shell_b ? "false"s : "true"s)
                  : "    ShapeMapB: None"s))
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
         interior_A + "    ShellLogMapStrength: " +
         (use_logarithmic_map_AB ? "1.0"s : "None"s) +
         "\n"
         "    CubeLogMapStrength: None\n"
         "    UseSphericalHarmonics: " +
         stringize(use_spherical_harmonics_shell_a) +
         "\n"
         "  ObjectB:\n"
         "    InnerRadius: 0.2\n"
         "    OuterRadius: 1.0\n"
         "    XCoord: -2.0\n" +
         interior_B + "    ShellLogMapStrength: " +
         (use_logarithmic_map_AB ? "1.0"s : "None"s) +
         "\n"
         "    CubeLogMapStrength: None\n"
         "    UseSphericalHarmonics: " +
         stringize(use_spherical_harmonics_shell_b) +
         "\n"
         "  CenterOfMassOffset: [0.1, 0.2]\n"
         "  Envelope:\n"
         "    Radius: 22.0\n"
         "    RadialDistribution: Projective\n"
         "  OuterShell:\n"
         "    Radius: 25.0\n"
         "    RadialPartitioning: [" +
         (excise_B ? "23.5" : "") +
         "]\n"
         "    RadialDistribution: " +
         (excise_B ? "[Logarithmic, Linear]" : "Linear") + "\n" +
         "    OpeningAngle: " + std::to_string(opening_angle) + "\n" +
         "    UseSphericalHarmonics: false\n" + outer_boundary_condition +
         "  InitialRefinement:\n" +
         (excise_A ? "" : "    ObjectAInterior: [1, 1, 1]\n") +
         (excise_B ? "" : "    ObjectBInterior: [1, 1, 1]\n") +
         // spherical-harmonic shell blocks only support radial refinement,
         // specified as a single number
         (use_spherical_harmonics_shell_a
              ? "    ObjectAShell: 1\n"s
              : "    ObjectAShell: [1, 1, " +
                    std::to_string(1 + additional_refinement_A) + "]\n") +
         (use_spherical_harmonics_shell_b
              ? "    ObjectBShell: 1\n"s
              : "    ObjectBShell: [1, 1, " +
                    std::to_string(1 + additional_refinement_B) + "]\n") +
         "    ObjectACube: [1, 1, 1]\n"
         "    ObjectBCube: [1, 1, 1]\n"
         "    Envelope: [1, 1, 1]\n"
         "    OuterShell0: [1, 1, " +
         std::to_string(1 + additional_refinement_outer) + "]\n" +
         (excise_B ? "    OuterShell1: [1, 1, 0]\n" : "") +
         // spherical-harmonic shell blocks require grid points as
         // [radial_points, L_max] with L_max >= 6
         ((use_spherical_harmonics_shell_a or use_spherical_harmonics_shell_b)
              ? "  InitialGridPoints:\n" +
                    (excise_A ? ""s : "    ObjectAInterior: [3, 3, 3]\n"s) +
                    (excise_B ? ""s : "    ObjectBInterior: [3, 3, 3]\n"s) +
                    (use_spherical_harmonics_shell_a
                         ? "    ObjectAShell: [3, 7]\n"s
                         : "    ObjectAShell: [3, 3, 3]\n"s) +
                    (use_spherical_harmonics_shell_b
                         ? "    ObjectBShell: [3, 7]\n"s
                         : "    ObjectBShell: [3, 3, 3]\n"s) +
                    "    ObjectACube: [3, 3, 3]\n"
                    "    ObjectBCube: [3, 3, 3]\n"
                    "    Envelope: [3, 3, 3]\n"
                    "    OuterShell0: [3, 3, 3]\n"s +
                    (excise_B ? "    OuterShell1: [3, 3, 3]\n"s : ""s)
              : "  InitialGridPoints: 3\n"s) +
         "  CubeScale: " + cube_scale +
         "\n"
         "  UseEquiangularMap: " +
         stringize(use_equiangular_map) +
         "\n"
         "  UseWorldtube: false\n" +
         time_dependence;
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
      {"OuterShell0", {{1, 1, 4}}},
      {"OuterShell1", {{1, 1, 2}}}};

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
        std::vector<double>{0.49 * (envelope_radius + outer_radius)},
        radial_distribution_outer_shell,
        opening_angle,
        false,
        false,
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
                                                  "OuterShell0UpperZLeft",
                                                  "OuterShell0UpperZRight",
                                                  "OuterShell0LowerZLeft",
                                                  "OuterShell0LowerZRight",
                                                  "OuterShell0UpperYLeft",
                                                  "OuterShell0UpperYRight",
                                                  "OuterShell0LowerYLeft",
                                                  "OuterShell0LowerYRight",
                                                  "OuterShell0UpperX",
                                                  "OuterShell0LowerX",
                                                  "OuterShell1UpperZLeft",
                                                  "OuterShell1UpperZRight",
                                                  "OuterShell1LowerZLeft",
                                                  "OuterShell1LowerZRight",
                                                  "OuterShell1UpperYLeft",
                                                  "OuterShell1UpperYRight",
                                                  "OuterShell1LowerYLeft",
                                                  "OuterShell1LowerYRight",
                                                  "OuterShell1UpperX",
                                                  "OuterShell1LowerX"};
    std::unordered_map<std::string, std::unordered_set<std::string>>
        expected_block_groups{
            {"Envelope",
             {"EnvelopeUpperZRight", "EnvelopeUpperX", "EnvelopeLowerZLeft",
              "EnvelopeLowerZRight", "EnvelopeUpperYRight",
              "EnvelopeLowerYRight", "EnvelopeUpperYLeft", "EnvelopeUpperZLeft",
              "EnvelopeLowerYLeft", "EnvelopeLowerX"}},
            {"OuterShell0",
             {"OuterShell0UpperZLeft", "OuterShell0UpperZRight",
              "OuterShell0LowerZLeft", "OuterShell0LowerZRight",
              "OuterShell0UpperYLeft", "OuterShell0UpperYRight",
              "OuterShell0LowerYLeft", "OuterShell0LowerYRight",
              "OuterShell0UpperX", "OuterShell0LowerX"}},
            {"OuterShell1",
             {"OuterShell1UpperZLeft", "OuterShell1UpperZRight",
              "OuterShell1LowerZLeft", "OuterShell1LowerZRight",
              "OuterShell1UpperYLeft", "OuterShell1UpperYRight",
              "OuterShell1LowerYLeft", "OuterShell1LowerYRight",
              "OuterShell1UpperX", "OuterShell1LowerX"}}};
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

  const auto& excision_spheres = domain.excision_spheres();

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
    // Unfortunately, it'd be very hard to create the grid to inertial map
    // necessary for the excision outside of the TimeDependentOptions class so
    // we just clone the actual one here.
    expected_excision_spheres.at("ExcisionSphereA")
        .inject_time_dependent_maps(excision_spheres.at("ExcisionSphereA")
                                        .moving_mesh_grid_to_inertial_map()
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
      // Unfortunately, it'd be very hard to create the grid to inertial map
      // necessary for the excision outside of the TimeDependentOptions class so
      // we just clone the actual one here.
      expected_excision_spheres.at("ExcisionSphereB")
          .inject_time_dependent_maps(excision_spheres.at("ExcisionSphereB")
                                          .moving_mesh_grid_to_inertial_map()
                                          .get_clone());
    }
  }

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
    const auto functions_of_time = binary_compact_object->functions_of_time();
    CHECK(functions_of_time.find("GridCenters") != functions_of_time.end());
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
          Object{0.5,
                 0.8,
                 1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          Object{0.3,
                 0.8,
                 -1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, std::vector<double>{20.0},
          Distribution::Linear, 120.0, false, false, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "First radial partition must be larger than the envelope radius"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.5,
                 0.8,
                 1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          Object{0.3,
                 0.8,
                 -1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, std::vector<double>{40.0},
          Distribution::Linear, 120.0, false, false, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Last radial partition must be smaller than the outer radius"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.5,
                 0.8,
                 1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          Object{0.3,
                 0.8,
                 -1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, std::vector<double>{28.0, 28.0},
          Distribution::Linear, 120.0, false, false, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Radial partitioning contains duplicate element"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.5,
                 0.8,
                 1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          Object{0.3,
                 0.8,
                 -1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, std::vector<double>{28.0, 29.0},
          std::vector{Distribution::Linear}, 120.0, false, false, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify a 'RadialDistribution' for every spherical shell."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.5,
                 0.8,
                 -1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          Object{0.3,
                 0.8,
                 -1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, std::vector<double>{}, Distribution::Linear,
          120.0, false, false, std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "The x-coordinate of ObjectA's center is expected to be positive."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.5,
                 0.8,
                 1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          Object{0.3,
                 0.8,
                 1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, std::vector<double>{}, Distribution::Linear,
          120.0, false, false, std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "The x-coordinate of ObjectB's center is expected to be negative."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3,
                 1.0,
                 8.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          Object{0.5,
                 1.0,
                 -7.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, std::vector<double>{}, Distribution::Linear,
          120.0, false, false, std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "The radius for the enveloping cube is too "
          "small! The Frustums will be malformed."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3,
                 1.0,
                 8.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          Object{0.5,
                 1.0,
                 -7.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 0.5, 2_st, 6_st, true,
          Distribution::Projective, std::vector<double>{}, Distribution::Linear,
          120.0, false, false, std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "The cube length should be greater than or equal to the initial "
          "separation between the two objects."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3,
                 0.8,
                 1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          Object{1.5,
                 0.8,
                 -1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, std::vector<double>{}, Distribution::Linear,
          120.0, false, false, std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "ObjectB's inner radius must be less than its outer radius."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{3.3,
                 0.8,
                 1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          Object{0.5,
                 0.8,
                 -1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, std::vector<double>{}, Distribution::Linear,
          120.0, false, false, std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "ObjectA's inner radius must be less than its outer radius."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3,
                 1.0,
                 1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          Object{0.5,
                 0.8,
                 -1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, std::vector<double>{}, Distribution::Linear,
          120.0, false, false, std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "ObjectA's outer radius is too large for the given separation,  try "
          "using 0.8"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3,
                 0.8,
                 1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          Object{0.5,
                 1.0,
                 -1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, std::vector<double>{}, Distribution::Linear,
          120.0, false, false, std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "ObjectB's outer radius is too large for the given separation,  try "
          "using 0.8"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3,
                 0.8,
                 1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          Object{0.5, 0.8, -1.0, std::nullopt, std::optional<double>{1.0}},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, true, 6_st,
          Distribution::Projective, std::vector<double>{}, Distribution::Linear,
          120.0, false, false, std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Using a logarithmically spaced radial grid in the "
          "part of Layer 1 enveloping Object B requires excising the interior "
          "of Object B"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 0.8, 1.0, std::nullopt, std::optional<double>{1.0}},
          Object{0.5,
                 0.8,
                 -1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st, 6_st, true,
          Distribution::Projective, std::vector<double>{}, Distribution::Linear,
          120.0, false, false, std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Using a logarithmically spaced radial grid in the "
          "part of Layer 1 enveloping Object A requires excising the interior "
          "of Object A"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 0.8, 1.0, false, std::nullopt},
          Object{0.5, 0.8, -1.0, false, std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.2, 2_st, 6_st),
      Catch::Matchers::ContainsSubstring(
          "A filled object cannot be offset within its cube."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3,
                 0.8,
                 1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          Object{0.5,
                 0.8,
                 -1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0,
          std::vector<std::array<size_t, 3>>{}, 6_st, true,
          Distribution::Projective, std::vector<double>{}, Distribution::Linear,
          120.0, false, false, std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("Invalid 'InitialRefinement'"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3,
                 0.8,
                 1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          Object{0.5,
                 0.8,
                 -1.0,
                 {{create_inner_boundary_condition()}},
                 std::nullopt},
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st,
          std::vector<std::array<size_t, 3>>{}, true, Distribution::Projective,
          std::vector<double>{}, Distribution::Linear, 120.0, false, false,
          std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("Invalid 'InitialGridPoints'"));
  // Misuse of SH types when SH is disabled
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object(0.5, 0.8, 1.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          Object(0.3, 0.8, -1.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0, 2_st,
          std::unordered_map<std::string, std::variant<std::array<size_t, 3>,
                                                       std::array<size_t, 2>>>{
              {"OuterShell", std::array<size_t, 2>{5, 4}}},
          true, Distribution::Projective, std::vector<double>{},
          Distribution::Linear, 120.0, false, false, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "only valid for spherical-harmonic shell blocks"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object(0.5, 0.8, 1.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          Object(0.3, 0.8, -1.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          std::array<double, 2>{{0.1, 0.2}}, 25.5, 32.4, 1.0,
          std::unordered_map<std::string,
                             std::variant<std::array<size_t, 3>, size_t>>{
              {"OuterShell", size_t{2}}},
          6_st, true, Distribution::Projective, std::vector<double>{},
          Distribution::Linear, 120.0, false, false, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "only valid for spherical-harmonic shell blocks"));
  // Note: the boundary condition-related parse errors are checked in the
  // test_connectivity function.
}

// Verify that the neighbor graph is two-sided: for every neighbor entry
// A->B in direction `d` with orientation `O`, block B must have the reverse
// entry B->A in direction `O(d).opposite()` with orientation
// `O.inverse_map()`.
void check_two_sided_neighbor_consistency(const Domain<3>& domain) {
  const auto& blocks = domain.blocks();
  for (const auto& block : blocks) {
    for (const auto& [direction, block_neighbors] : block.neighbors()) {
      for (const size_t neighbor_id : block_neighbors.ids()) {
        CAPTURE(block.id());
        CAPTURE(direction);
        CAPTURE(neighbor_id);
        const auto& orientation = block_neighbors.orientation(neighbor_id);
        const auto direction_in_neighbor = orientation(direction).opposite();
        CAPTURE(direction_in_neighbor);
        const auto& neighbor = blocks[neighbor_id];
        REQUIRE(neighbor.neighbors().count(direction_in_neighbor) == 1);
        const auto& reverse_neighbors =
            neighbor.neighbors().at(direction_in_neighbor);
        REQUIRE(reverse_neighbors.ids().count(block.id()) == 1);
        CHECK(reverse_neighbors.orientation(block.id()) ==
              orientation.inverse_map());
      }
    }
  }
}

void test_spherical_harmonics_wavezone() {
  INFO("Test BinaryCompactObject with SphericalHarmonicsInWavezone=true");
  // Both objects excised, no radial partitioning -> 1 SH outer shell block.
  // Block count: 24 (object shells+cubes) + 10 (envelope) + 1 (SH shell) = 35.
  using GridPointsMap = std::unordered_map<
      std::string, std::variant<std::array<size_t, 3>, std::array<size_t, 2>>>;
  using RefinementMap =
      std::unordered_map<std::string,
                         std::variant<std::array<size_t, 3>, size_t>>;
  const domain::creators::BinaryCompactObject bco_sh{
      Object(0.3, 1.0, 3.0, {{create_inner_boundary_condition()}},
             std::nullopt),
      Object(0.5, 1.0, -3.0, {{create_inner_boundary_condition()}},
             std::nullopt),
      std::array<double, 2>{{0.0, 0.0}},
      25.5,
      32.4,
      1.0,
      RefinementMap{{"ObjectAShell", std::array<size_t, 3>{1, 1, 1}},
                    {"ObjectACube", std::array<size_t, 3>{1, 1, 1}},
                    {"ObjectBShell", std::array<size_t, 3>{1, 1, 1}},
                    {"ObjectBCube", std::array<size_t, 3>{1, 1, 1}},
                    {"Envelope", std::array<size_t, 3>{1, 1, 1}},
                    {"OuterShell0", size_t{2}}},
      GridPointsMap{{"ObjectAShell", std::array<size_t, 3>{3, 3, 3}},
                    {"ObjectACube", std::array<size_t, 3>{3, 3, 3}},
                    {"ObjectBShell", std::array<size_t, 3>{3, 3, 3}},
                    {"ObjectBCube", std::array<size_t, 3>{3, 3, 3}},
                    {"Envelope", std::array<size_t, 3>{3, 3, 3}},
                    {"OuterShell0", std::array<size_t, 2>{4, 7}}},
      true,
      Distribution::Projective,
      std::vector<double>{},
      Distribution::Linear,
      120.0,
      true,
      false,
      std::nullopt,
      create_outer_boundary_condition()};

  // Check block count: 35 total (24 object + 10 envelope + 1 SH shell).
  CHECK(bco_sh.block_names().size() == 35);

  // SH outer shell is a single block named "OuterShell0" (not in any group).
  CHECK(bco_sh.block_names()[34] == "OuterShell0");
  CHECK(not bco_sh.block_groups().contains("OuterShell0"));

  // initial_extents: OuterShell0 stores ell internally ({4, 7, 7}) and
  // converts to collocation points on read: {4, 7+1, 2*7+1} = {4, 8, 15}.
  const auto extents = bco_sh.initial_extents();
  REQUIRE(extents.size() == 35);
  CHECK(extents[34] == std::array<size_t, 3>{4, 8, 15});

  // initial_refinement: OuterShell0 should be {2, 0, 0}.
  const auto refinement = bco_sh.initial_refinement_levels();
  REQUIRE(refinement.size() == 35);
  CHECK(refinement[34] == std::array<size_t, 3>{2, 0, 0});

  // external_boundary_conditions: outer BC on upper_xi of block 34.
  const auto bcs = bco_sh.external_boundary_conditions();
  REQUIRE(bcs.size() == 35);
  CHECK(bcs[34].count(Direction<3>::upper_xi()) == 1);
  // No boundary condition on upper_zeta for the SH shell block.
  CHECK(bcs[34].count(Direction<3>::upper_zeta()) == 0);

  // Verify the domain can be constructed and inspect neighbor topology.
  const auto domain_sh = bco_sh.create_domain();
  check_two_sided_neighbor_consistency(domain_sh);
  const auto& blocks = domain_sh.blocks();
  REQUIRE(blocks.size() == 35);

  // OuterShell0 (block 34): lower_xi has 10 envelope neighbors (non-conforming)
  // and no upper_xi neighbor (only 1 shell).
  const auto& sh_block = blocks[34];
  CHECK(sh_block.neighbors().count(Direction<3>::lower_xi()) == 1);
  CHECK(sh_block.neighbors().at(Direction<3>::lower_xi()).ids().size() == 10);
  CHECK(sh_block.neighbors().count(Direction<3>::upper_xi()) == 0);

  // Each envelope block (24-33): upper_zeta should point to block 34.
  for (size_t j = 24; j < 34; ++j) {
    CAPTURE(j);
    CHECK(blocks[j].neighbors().count(Direction<3>::upper_zeta()) == 1);
    CHECK(*blocks[j].neighbors().at(Direction<3>::upper_zeta()).ids().begin() ==
          34);
  }

  // Check orientations at the envelope<->shell boundary.
  // Shell's xi = frustum's zeta (radial); angular axes use self().
  const OrientationMap<3> expected_shell_to_frustum{
      {{Direction<3>::upper_zeta(), Direction<3>::self(),
        Direction<3>::self()}}};
  const auto expected_frustum_to_shell =
      expected_shell_to_frustum.inverse_map();
  const auto& lower_xi_neighbors =
      sh_block.neighbors().at(Direction<3>::lower_xi());
  for (size_t j = 24; j < 34; ++j) {
    CAPTURE(j);
    CHECK(lower_xi_neighbors.orientation(j) == expected_shell_to_frustum);
  }
  for (size_t j = 24; j < 34; ++j) {
    CAPTURE(j);
    const auto& uz_neighbors =
        blocks[j].neighbors().at(Direction<3>::upper_zeta());
    CHECK(uz_neighbors.orientation(34) == expected_frustum_to_shell);
  }

  // 2-shell test: add a radial partition and verify the shell-to-shell
  // neighbor.
  const domain::creators::BinaryCompactObject bco_sh2{
      Object(0.3, 1.0, 3.0, {{create_inner_boundary_condition()}},
             std::nullopt),
      Object(0.5, 1.0, -3.0, {{create_inner_boundary_condition()}},
             std::nullopt),
      std::array<double, 2>{{0.0, 0.0}},
      25.5,
      32.4,
      1.0,
      RefinementMap{{"ObjectAShell", std::array<size_t, 3>{1, 1, 1}},
                    {"ObjectACube", std::array<size_t, 3>{1, 1, 1}},
                    {"ObjectBShell", std::array<size_t, 3>{1, 1, 1}},
                    {"ObjectBCube", std::array<size_t, 3>{1, 1, 1}},
                    {"Envelope", std::array<size_t, 3>{1, 1, 1}},
                    {"OuterShell0", size_t{2}},
                    {"OuterShell1", size_t{2}}},
      GridPointsMap{{"ObjectAShell", std::array<size_t, 3>{3, 3, 3}},
                    {"ObjectACube", std::array<size_t, 3>{3, 3, 3}},
                    {"ObjectBShell", std::array<size_t, 3>{3, 3, 3}},
                    {"ObjectBCube", std::array<size_t, 3>{3, 3, 3}},
                    {"Envelope", std::array<size_t, 3>{3, 3, 3}},
                    {"OuterShell0", std::array<size_t, 2>{4, 7}},
                    {"OuterShell1", std::array<size_t, 2>{4, 7}}},
      true,
      Distribution::Projective,
      std::vector<double>{28.0},
      Distribution::Linear,
      120.0,
      true,
      false,
      std::nullopt,
      create_outer_boundary_condition()};

  const auto domain_sh2 = bco_sh2.create_domain();
  check_two_sided_neighbor_consistency(domain_sh2);
  const auto& blocks2 = domain_sh2.blocks();
  REQUIRE(blocks2.size() == 36);

  // OuterShell0 (block 34): lower_xi -> 10 envelope blocks, upper_xi -> {35}.
  CHECK(blocks2[34].neighbors().count(Direction<3>::lower_xi()) == 1);
  CHECK(blocks2[34].neighbors().at(Direction<3>::lower_xi()).ids().size() ==
        10);
  CHECK(blocks2[34].neighbors().count(Direction<3>::upper_xi()) == 1);
  CHECK(*blocks2[34].neighbors().at(Direction<3>::upper_xi()).ids().begin() ==
        35);

  // OuterShell1 (block 35): lower_xi -> {34}, no upper_xi.
  CHECK(blocks2[35].neighbors().count(Direction<3>::lower_xi()) == 1);
  CHECK(*blocks2[35].neighbors().at(Direction<3>::lower_xi()).ids().begin() ==
        34);
  CHECK(blocks2[35].neighbors().count(Direction<3>::upper_xi()) == 0);

  // external_boundary_conditions for 2-shell case: outer BC on upper_xi of
  // block 35 (outermost shell); block 34 has no upper_xi BC.
  const auto bcs2 = bco_sh2.external_boundary_conditions();
  REQUIRE(bcs2.size() == 36);
  CHECK(bcs2[35].count(Direction<3>::upper_xi()) == 1);
  CHECK(bcs2[34].count(Direction<3>::upper_xi()) == 0);

  // Error cases: strict validation rejects invalid SH block input.
  // Convenience maps used in several tests below.
  const RefinementMap valid_refinement{
      {"ObjectAShell", std::array<size_t, 3>{1, 1, 1}},
      {"ObjectACube", std::array<size_t, 3>{1, 1, 1}},
      {"ObjectBShell", std::array<size_t, 3>{1, 1, 1}},
      {"ObjectBCube", std::array<size_t, 3>{1, 1, 1}},
      {"Envelope", std::array<size_t, 3>{1, 1, 1}},
      {"OuterShell0", size_t{2}}};
  const GridPointsMap valid_grid_points{
      {"ObjectAShell", std::array<size_t, 3>{3, 3, 3}},
      {"ObjectACube", std::array<size_t, 3>{3, 3, 3}},
      {"ObjectBShell", std::array<size_t, 3>{3, 3, 3}},
      {"ObjectBCube", std::array<size_t, 3>{3, 3, 3}},
      {"Envelope", std::array<size_t, 3>{3, 3, 3}},
      {"OuterShell0", std::array<size_t, 2>{4, 7}}};
  // Global scalar refinement (non-zero) -> post-expansion angular error.
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object(0.3, 1.0, 3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          Object(0.5, 1.0, -3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          std::array<double, 2>{{0.0, 0.0}}, 25.5, 32.4, 1.0, 1_st,
          valid_grid_points, true, Distribution::Projective,
          std::vector<double>{}, Distribution::Linear, 120.0, true, false,
          std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "angular h-refinement is not supported"));
  // Global scalar grid points (ell too small) -> post-expansion ell>=6 error.
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object(0.3, 1.0, 3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          Object(0.5, 1.0, -3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          std::array<double, 2>{{0.0, 0.0}}, 25.5, 32.4, 1.0, valid_refinement,
          4_st, true, Distribution::Projective, std::vector<double>{},
          Distribution::Linear, 120.0, true, false, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("at least ell=6"));
  // Per-block [radial_points, L_max] with L_max < 6 -> post-expansion
  // ell>=6 error. This is the canonical user-facing path to this error.
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object(0.3, 1.0, 3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          Object(0.5, 1.0, -3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          std::array<double, 2>{{0.0, 0.0}}, 25.5, 32.4, 1.0, valid_refinement,
          GridPointsMap{{"ObjectAShell", std::array<size_t, 3>{3, 3, 3}},
                        {"ObjectACube", std::array<size_t, 3>{3, 3, 3}},
                        {"ObjectBShell", std::array<size_t, 3>{3, 3, 3}},
                        {"ObjectBCube", std::array<size_t, 3>{3, 3, 3}},
                        {"Envelope", std::array<size_t, 3>{3, 3, 3}},
                        {"OuterShell0", std::array<size_t, 2>{4, 5}}},
          true, Distribution::Projective, std::vector<double>{},
          Distribution::Linear, 120.0, true, false, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("at least ell=6"));
  // Global array<3> with l_max != m_max -> post-expansion single-ell error.
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object(0.3, 1.0, 3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          Object(0.5, 1.0, -3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          std::array<double, 2>{{0.0, 0.0}}, 25.5, 32.4, 1.0, valid_refinement,
          std::array<size_t, 3>{4, 8, 10}, true, Distribution::Projective,
          std::vector<double>{}, Distribution::Linear, 120.0, true, false,
          std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("specified by a single ell"));
  // Map refinement with array<3> for OuterShell0 -> pre-expansion error.
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object(0.3, 1.0, 3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          Object(0.5, 1.0, -3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          std::array<double, 2>{{0.0, 0.0}}, 25.5, 32.4, 1.0,
          RefinementMap{{"ObjectAShell", std::array<size_t, 3>{1, 1, 1}},
                        {"ObjectACube", std::array<size_t, 3>{1, 1, 1}},
                        {"ObjectBShell", std::array<size_t, 3>{1, 1, 1}},
                        {"ObjectBCube", std::array<size_t, 3>{1, 1, 1}},
                        {"Envelope", std::array<size_t, 3>{1, 1, 1}},
                        {"OuterShell0", std::array<size_t, 3>{2, 0, 0}}},
          valid_grid_points, true, Distribution::Projective,
          std::vector<double>{}, Distribution::Linear, 120.0, true, false,
          std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify its refinement as a single number"));
  // Map grid points with array<3> for OuterShell0 -> pre-expansion error.
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object(0.3, 1.0, 3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          Object(0.5, 1.0, -3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          std::array<double, 2>{{0.0, 0.0}}, 25.5, 32.4, 1.0, valid_refinement,
          GridPointsMap{{"ObjectAShell", std::array<size_t, 3>{3, 3, 3}},
                        {"ObjectACube", std::array<size_t, 3>{3, 3, 3}},
                        {"ObjectBShell", std::array<size_t, 3>{3, 3, 3}},
                        {"ObjectBCube", std::array<size_t, 3>{3, 3, 3}},
                        {"Envelope", std::array<size_t, 3>{3, 3, 3}},
                        {"OuterShell0", std::array<size_t, 3>{4, 8, 15}}},
          true, Distribution::Projective, std::vector<double>{},
          Distribution::Linear, 120.0, true, false, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Specify its grid points as [radial_points, L_max]"));
  // Map refinement with size_t on a non-SH block -> pre-expansion error.
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object(0.3, 1.0, 3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          Object(0.5, 1.0, -3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          std::array<double, 2>{{0.0, 0.0}}, 25.5, 32.4, 1.0,
          RefinementMap{{"Envelope", size_t{1}}, {"OuterShell0", size_t{2}}},
          valid_grid_points, true, Distribution::Projective,
          std::vector<double>{}, Distribution::Linear, 120.0, true, false,
          std::nullopt, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "only valid for spherical-harmonic shell blocks"));
  // Map grid points with array<2> on a non-SH block -> pre-expansion error.
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object(0.3, 1.0, 3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          Object(0.5, 1.0, -3.0, {{create_inner_boundary_condition()}},
                 std::nullopt),
          std::array<double, 2>{{0.0, 0.0}}, 25.5, 32.4, 1.0, valid_refinement,
          GridPointsMap{{"Envelope", std::array<size_t, 2>{3, 4}},
                        {"OuterShell0", std::array<size_t, 2>{4, 7}}},
          true, Distribution::Projective, std::vector<double>{},
          Distribution::Linear, 120.0, true, false, std::nullopt,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "only valid for spherical-harmonic shell blocks"));
}

void test_spherical_harmonics_object_shells() {
  INFO("Test BinaryCompactObject with spherical-harmonic object shells");
  using GridPointsMap = std::unordered_map<
      std::string, std::variant<std::array<size_t, 3>, std::array<size_t, 2>>>;
  using RefinementMap =
      std::unordered_map<std::string,
                         std::variant<std::array<size_t, 3>, size_t>>;

  const domain::creators::BinaryCompactObject bco_a_sh{
      Object{
          0.3, 1.0, 3.0, {{create_inner_boundary_condition()}}, 2.0, 1.5, true},
      Object{0.5,
             1.0,
             -3.0,
             {{create_inner_boundary_condition()}},
             std::nullopt,
             std::nullopt,
             false},
      std::array<double, 2>{{0.0, 0.0}},
      25.5,
      32.4,
      1.0,
      RefinementMap{{"ObjectAShell", size_t{2}},
                    {"ObjectACube", std::array<size_t, 3>{1, 1, 1}},
                    {"ObjectBShell", std::array<size_t, 3>{1, 1, 1}},
                    {"ObjectBCube", std::array<size_t, 3>{1, 1, 1}},
                    {"Envelope", std::array<size_t, 3>{1, 1, 1}},
                    {"OuterShell0", std::array<size_t, 3>{1, 1, 2}}},
      GridPointsMap{{"ObjectAShell", std::array<size_t, 2>{4, 7}},
                    {"ObjectACube", std::array<size_t, 3>{3, 3, 3}},
                    {"ObjectBShell", std::array<size_t, 3>{3, 3, 3}},
                    {"ObjectBCube", std::array<size_t, 3>{3, 3, 3}},
                    {"Envelope", std::array<size_t, 3>{3, 3, 3}},
                    {"OuterShell0", std::array<size_t, 3>{3, 3, 3}}},
      true,
      Distribution::Projective,
      std::vector<double>{},
      Distribution::Linear,
      120.0,
      false,
      false,
      std::nullopt,
      create_outer_boundary_condition()};

  CHECK(bco_a_sh.block_names().size() == 39);
  CHECK(bco_a_sh.block_names()[0] == "ObjectAShell0");
  REQUIRE(bco_a_sh.block_groups().contains("ObjectAShell"));
  CHECK(bco_a_sh.block_groups().at("ObjectAShell") ==
        std::unordered_set<std::string>{"ObjectAShell0"});
  CHECK(bco_a_sh.initial_extents()[0] == std::array<size_t, 3>{4, 8, 15});
  CHECK(bco_a_sh.initial_refinement_levels()[0] ==
        std::array<size_t, 3>{2, 0, 0});

  const auto domain_a_sh = bco_a_sh.create_domain();
  check_two_sided_neighbor_consistency(domain_a_sh);
  const auto& blocks_a_sh = domain_a_sh.blocks();
  REQUIRE(blocks_a_sh.size() == 39);
  const auto& shell_map = blocks_a_sh[0].stationary_map();
  const auto inner_jacobian = shell_map.jacobian(
      tnsr::I<double, 3, Frame::BlockLogical>{{-1.0, 1.0, 0.7}});
  const auto outer_jacobian = shell_map.jacobian(
      tnsr::I<double, 3, Frame::BlockLogical>{{1.0, 1.0, 0.7}});
  // The ratio of radial Jacobians specifies the logarithmic-map strength.
  CHECK(std::hypot(get<0, 0>(outer_jacobian), get<1, 0>(outer_jacobian),
                   get<2, 0>(outer_jacobian)) /
            std::hypot(get<0, 0>(inner_jacobian), get<1, 0>(inner_jacobian),
                       get<2, 0>(inner_jacobian)) ==
        approx(2.0 * 1.0 / 0.3));
  CHECK(blocks_a_sh[0].neighbors().count(Direction<3>::lower_xi()) == 0);
  CHECK(blocks_a_sh[0].neighbors().count(Direction<3>::upper_xi()) == 1);
  CHECK(blocks_a_sh[0].neighbors().at(Direction<3>::upper_xi()).ids().size() ==
        6);
  // The shell's radial +xi axis is aligned with the cube wedges' radial
  // +zeta axis (both point outward); angular axes use self().
  const OrientationMap<3> expected_shell_to_cube{
      {{Direction<3>::upper_zeta(), Direction<3>::self(),
        Direction<3>::self()}}};
  const auto expected_cube_to_shell = expected_shell_to_cube.inverse_map();
  for (size_t i = 1; i <= 6; ++i) {
    CAPTURE(i);
    CHECK(blocks_a_sh[0]
              .neighbors()
              .at(Direction<3>::upper_xi())
              .orientation(i) == expected_shell_to_cube);
    CHECK(blocks_a_sh[i].neighbors().count(Direction<3>::lower_zeta()) == 1);
    const auto& lower_zeta_neighbors =
        blocks_a_sh[i].neighbors().at(Direction<3>::lower_zeta());
    CHECK(*lower_zeta_neighbors.ids().begin() == 0);
    CHECK(lower_zeta_neighbors.orientation(0) == expected_cube_to_shell);
  }
  const auto bcs_a_sh = bco_a_sh.external_boundary_conditions();
  REQUIRE(bcs_a_sh.size() == 39);
  CHECK(bcs_a_sh[0].count(Direction<3>::lower_xi()) == 1);
  CHECK(bcs_a_sh[0].count(Direction<3>::lower_zeta()) == 0);

  const domain::creators::BinaryCompactObject bco_b_sh{
      Object{0.3,
             1.0,
             3.0,
             {{create_inner_boundary_condition()}},
             std::nullopt,
             std::nullopt,
             false},
      Object{0.5,
             1.0,
             -3.0,
             {{create_inner_boundary_condition()}},
             std::nullopt,
             std::nullopt,
             true},
      std::array<double, 2>{{0.0, 0.0}},
      25.5,
      32.4,
      1.0,
      RefinementMap{{"ObjectAShell", std::array<size_t, 3>{1, 1, 1}},
                    {"ObjectACube", std::array<size_t, 3>{1, 1, 1}},
                    {"ObjectBShell", size_t{2}},
                    {"ObjectBCube", std::array<size_t, 3>{1, 1, 1}},
                    {"Envelope", std::array<size_t, 3>{1, 1, 1}},
                    {"OuterShell0", std::array<size_t, 3>{1, 1, 2}}},
      GridPointsMap{{"ObjectAShell", std::array<size_t, 3>{3, 3, 3}},
                    {"ObjectACube", std::array<size_t, 3>{3, 3, 3}},
                    {"ObjectBShell", std::array<size_t, 2>{4, 7}},
                    {"ObjectBCube", std::array<size_t, 3>{3, 3, 3}},
                    {"Envelope", std::array<size_t, 3>{3, 3, 3}},
                    {"OuterShell0", std::array<size_t, 3>{3, 3, 3}}},
      true,
      Distribution::Projective,
      std::vector<double>{},
      Distribution::Linear,
      120.0,
      false,
      false,
      std::nullopt,
      create_outer_boundary_condition()};

  CHECK(bco_b_sh.block_names().size() == 39);
  CHECK(bco_b_sh.block_names()[12] == "ObjectBShell0");
  CHECK(bco_b_sh.initial_extents()[12] == std::array<size_t, 3>{4, 8, 15});
  CHECK(bco_b_sh.initial_refinement_levels()[12] ==
        std::array<size_t, 3>{2, 0, 0});
  check_two_sided_neighbor_consistency(bco_b_sh.create_domain());
  const auto bcs_b_sh = bco_b_sh.external_boundary_conditions();
  REQUIRE(bcs_b_sh.size() == 39);
  CHECK(bcs_b_sh[12].count(Direction<3>::lower_xi()) == 1);
  CHECK(bcs_b_sh[12].count(Direction<3>::lower_zeta()) == 0);

  const domain::creators::BinaryCompactObject bco_both_sh{
      Object{0.3,
             1.0,
             3.0,
             {{create_inner_boundary_condition()}},
             std::nullopt,
             std::nullopt,
             true},
      Object{0.5,
             1.0,
             -3.0,
             {{create_inner_boundary_condition()}},
             std::nullopt,
             std::nullopt,
             true},
      std::array<double, 2>{{0.0, 0.0}},
      25.5,
      32.4,
      1.0,
      RefinementMap{{"ObjectAShell", size_t{1}},
                    {"ObjectACube", std::array<size_t, 3>{1, 1, 1}},
                    {"ObjectBShell", size_t{1}},
                    {"ObjectBCube", std::array<size_t, 3>{1, 1, 1}},
                    {"Envelope", std::array<size_t, 3>{1, 1, 1}},
                    {"OuterShell0", size_t{2}}},
      GridPointsMap{{"ObjectAShell", std::array<size_t, 2>{4, 7}},
                    {"ObjectACube", std::array<size_t, 3>{3, 3, 3}},
                    {"ObjectBShell", std::array<size_t, 2>{4, 7}},
                    {"ObjectBCube", std::array<size_t, 3>{3, 3, 3}},
                    {"Envelope", std::array<size_t, 3>{3, 3, 3}},
                    {"OuterShell0", std::array<size_t, 2>{4, 7}}},
      true,
      Distribution::Projective,
      std::vector<double>{},
      Distribution::Linear,
      120.0,
      true,
      false,
      std::nullopt,
      create_outer_boundary_condition()};

  CHECK(bco_both_sh.block_names().size() == 25);
  CHECK(bco_both_sh.block_names()[0] == "ObjectAShell0");
  CHECK(bco_both_sh.block_names()[7] == "ObjectBShell0");
  CHECK(bco_both_sh.block_names()[24] == "OuterShell0");
  CHECK(not bco_both_sh.block_groups().contains("OuterShell0"));
  check_two_sided_neighbor_consistency(bco_both_sh.create_domain());

  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 1.0, 3.0, false, std::nullopt, std::nullopt, true},
          Object{0.5, 1.0, -3.0, true, std::nullopt, std::nullopt, false},
          std::array<double, 2>{{0.0, 0.0}}, 25.5, 32.4, 1.0, 0_st, 4_st, true,
          Distribution::Projective, std::vector<double>{}, Distribution::Linear,
          120.0, false, false, std::nullopt, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("requires excising Object A"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 1.0, 3.0, true, std::nullopt, std::nullopt, true},
          Object{0.5, 1.0, -3.0, true, std::nullopt, std::nullopt, false},
          std::array<double, 2>{{0.0, 0.0}}, 25.5, 32.4, 1.2, 0_st, 4_st, true,
          Distribution::Projective, std::vector<double>{}, Distribution::Linear,
          120.0, false, false, std::nullopt, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("requires CubeScale to be 1.0"));

  const auto sh_time_dep_creator =
      TestHelpers::test_option_tag<domain::OptionTags::DomainCreator<3>,
                                   Metavariables<3, true>>(create_option_string(
          true, true, true, false, true, 0, 0, 0, 120.0, true, true, false));
  REQUIRE(sh_time_dep_creator != nullptr);
  CHECK_NOTHROW(sh_time_dep_creator->create_domain());
  CHECK(not sh_time_dep_creator->functions_of_time().empty());

  auto invalid_transition_options = create_option_string(
      true, true, true, false, true, 0, 0, 0, 120.0, true, true, false);
  const std::string valid_transition = "TransitionEndsAtCube: false";
  invalid_transition_options.replace(
      invalid_transition_options.find(valid_transition),
      valid_transition.size(), "TransitionEndsAtCube: true");
  const auto invalid_transition_creator =
      TestHelpers::test_option_tag<domain::OptionTags::DomainCreator<3>,
                                   Metavariables<3, true>>(
          invalid_transition_options);
  CHECK_THROWS_WITH(
      invalid_transition_creator->create_domain(),
      Catch::Matchers::ContainsSubstring(
          "Set 'TransitionEndsAtCube' to 'false' for this object"));

  // Both objects with a spherical-harmonic shell AND a shape map each. This
  // is the configuration used to evolve a binary with one Ylm shell per
  // object, so both shape maps must transition to the sphere.
  const auto sh_both_time_dep_creator =
      TestHelpers::test_option_tag<domain::OptionTags::DomainCreator<3>,
                                   Metavariables<3, true>>(create_option_string(
          true, true, true, false, true, 0, 0, 0, 120.0, true, true, true));
  REQUIRE(sh_both_time_dep_creator != nullptr);
  const auto time_dependent_domain = sh_both_time_dep_creator->create_domain();
  check_two_sided_neighbor_consistency(time_dependent_domain);
  auto functions_of_time = sh_both_time_dep_creator->functions_of_time();
  for (const auto& name : {"ShapeA", "ShapeB"}) {
    auto initial_values = functions_of_time.at(name)->func_and_2_derivs(1.0);
    initial_values[0][2] = 0.02;
    functions_of_time.at(name) =
        std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
            1.0, std::move(initial_values),
            std::numeric_limits<double>::infinity());
  }
  const auto& time_dependent_blocks = time_dependent_domain.blocks();
  for (const size_t shell_id : {0_st, 7_st}) {
    CAPTURE(shell_id);
    const auto& shell = time_dependent_blocks[shell_id];
    REQUIRE(shell.has_distorted_frame());
    const auto inner_grid_coords = shell.moving_mesh_logical_to_grid_map()(
        tnsr::I<double, 3, Frame::BlockLogical>{{-1.0, 1.0, 0.7}});
    const auto inner_distorted_coords =
        shell.moving_mesh_grid_to_distorted_map()(inner_grid_coords, 1.2,
                                                  functions_of_time);
    CHECK_FALSE(equal_within_roundoff(get<0>(inner_grid_coords),
                                      get<0>(inner_distorted_coords)));
    for (size_t cube_id = shell_id + 1; cube_id <= shell_id + 6; ++cube_id) {
      CAPTURE(cube_id);
      const auto& cube = time_dependent_blocks[cube_id];
      CHECK_FALSE(cube.has_distorted_frame());
      const auto cube_grid_coords = cube.moving_mesh_logical_to_grid_map()(
          tnsr::I<double, 3, Frame::BlockLogical>{{0.2, -0.4, -1.0}});
      const auto shell_logical_coords =
          shell.moving_mesh_logical_to_grid_map().inverse(cube_grid_coords);
      REQUIRE(shell_logical_coords.has_value());
      CHECK(get<0>(*shell_logical_coords) == approx(1.0));
      const auto shell_grid_coords =
          shell.moving_mesh_logical_to_grid_map()(*shell_logical_coords);
      CHECK_ITERABLE_APPROX(shell_grid_coords, cube_grid_coords);
      for (const double time : {1.0, 1.2}) {
        CAPTURE(time);
        const auto distorted_coords = shell.moving_mesh_grid_to_distorted_map()(
            shell_grid_coords, time, functions_of_time);
        CHECK_ITERABLE_APPROX(
            make_array(get<0>(distorted_coords), get<1>(distorted_coords),
                       get<2>(distorted_coords)),
            make_array(get<0>(shell_grid_coords), get<1>(shell_grid_coords),
                       get<2>(shell_grid_coords)));
        const auto inertial_coords = shell.moving_mesh_grid_to_inertial_map()(
            shell_grid_coords, time, functions_of_time);
        CHECK_ITERABLE_APPROX(inertial_coords,
                              cube.moving_mesh_grid_to_inertial_map()(
                                  cube_grid_coords, time, functions_of_time));
        CHECK_ITERABLE_APPROX(inertial_coords,
                              shell.moving_mesh_distorted_to_inertial_map()(
                                  distorted_coords, time, functions_of_time));
      }
    }
  }
}

template <domain::ObjectLabel Object>
using HardcodedShape =
    domain::creators::time_dependent_options::ShapeMapOptions<true, Object>;

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
  const domain::creators::BinaryCompactObject domain_creator{
      Object{inner_radius_A, 4., x_pos_A, true, std::optional<double>{1.0}},
      Object{inner_radius_B, 4., x_pos_B, true, std::optional<double>{1.0}},
      std::array<double, 2>{{0.1, 0.2}},
      40.,
      200.,
      1.0,
      0_st,
      6_st,
      true,
      Distribution::Projective,
      std::vector<double>{},
      Distribution::Inverse,
      120.,
      false,
      false,
      domain::creators::bco::TimeDependentMapOptions<false>{
          0., std::nullopt, std::nullopt, std::nullopt, std::nullopt,
          HardcodedShape<domain::ObjectLabel::A>{
              32_st, domain::creators::time_dependent_options::
                         KerrSchildFromBoyerLindquist{mass_A, spin_A}},
          HardcodedShape<domain::ObjectLabel::B>{
              32_st, domain::creators::time_dependent_options::
                         KerrSchildFromBoyerLindquist{mass_B, spin_B}},
          std::nullopt}};
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

// [[TimeOut, 45]]
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
  test_spherical_harmonics_wavezone();
  test_spherical_harmonics_object_shells();
  test_kerr_horizon_conforming();
}
