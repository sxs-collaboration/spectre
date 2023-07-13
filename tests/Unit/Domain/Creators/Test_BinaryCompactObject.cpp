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
#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Domain/Block.hpp"                       // IWYU pragma: keep
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/Protocols/Metavariables.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
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
using Object = domain::creators::BinaryCompactObject::Object;
using CartesianCubeAtXCoord =
    domain::creators::BinaryCompactObject::CartesianCubeAtXCoord;
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
  MAKE_GENERATOR(gen);

  // ObjectA:
  constexpr double inner_radius_objectA = 0.3;
  constexpr double outer_radius_objectA = 1.0;
  constexpr double xcoord_objectA = 3.0;

  // ObjectB:
  constexpr double inner_radius_objectB = 0.5;
  constexpr double outer_radius_objectB = 1.0;
  constexpr double xcoord_objectB = -3.0;

  // Envelope:
  constexpr double envelope_radius = 25.5;

  // Outer shell:
  constexpr double outer_radius = 32.4;

  // Misc.:
  constexpr size_t grid_points = 3;
  constexpr bool use_equiangular_map = true;

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
        envelope_radius,
        outer_radius,
        refinement,
        grid_points,
        use_equiangular_map,
        radial_distribution_envelope,
        radial_distribution_outer_shell,
        opening_angle,
        with_boundary_conditions ? create_outer_boundary_condition() : nullptr};

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
             {"ObjectAShellLowerZ", "ObjectAShellUpperX", "ObjectAShellLowerX",
              "ObjectAShellUpperY", "ObjectAShellUpperZ",
              "ObjectAShellLowerY"}},
            {"ObjectBShell",
             {"ObjectBShellLowerZ", "ObjectBShellUpperX", "ObjectBShellLowerX",
              "ObjectBShellUpperY", "ObjectBShellUpperZ",
              "ObjectBShellLowerY"}},
            {"ObjectACube",
             {"ObjectACubeLowerY", "ObjectACubeLowerZ", "ObjectACubeUpperY",
              "ObjectACubeUpperX", "ObjectACubeUpperZ", "ObjectACubeLowerX"}},
            {"ObjectBCube",
             {"ObjectBCubeLowerY", "ObjectBCubeLowerZ", "ObjectBCubeUpperY",
              "ObjectBCubeUpperX", "ObjectBCubeUpperZ", "ObjectBCubeLowerX"}},
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
          ExcisionSphere<3>{
              inner_radius_objectA,
              tnsr::I<double, 3, Frame::Grid>{{xcoord_objectA, 0.0, 0.0}},
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
          ExcisionSphere<3>{
              inner_radius_objectB,
              tnsr::I<double, 3, Frame::Grid>{{xcoord_objectB, 0.0, 0.0}},
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
              Object{inner_radius_objectA, outer_radius_objectA, xcoord_objectA,
                     excise_interiorA ? std::make_optional(Excision{
                                            create_inner_boundary_condition()})
                                      : std::nullopt,
                     false},
              domain::creators::BinaryCompactObject::Object{
                  inner_radius_objectB, outer_radius_objectB, xcoord_objectB,
                  excise_interiorB ? std::make_optional(Excision{
                                         create_inner_boundary_condition()})
                                   : std::nullopt,
                  false},
              envelope_radius, outer_radius, refinement, grid_points,
              use_equiangular_map, radial_distribution_envelope,
              radial_distribution_outer_shell, opening_angle,
              std::make_unique<PeriodicBc>(),
              Options::Context{false, {}, 1, 1}),
          Catch::Matchers::Contains("Cannot have periodic boundary "
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
                envelope_radius, outer_radius, refinement, grid_points,
                use_equiangular_map, radial_distribution_envelope,
                radial_distribution_outer_shell, opening_angle,
                create_outer_boundary_condition(),
                Options::Context{false, {}, 1, 1}),
            Catch::Matchers::Contains("Cannot have periodic boundary "
                                      "conditions with a binary domain"));
        CHECK_THROWS_WITH(
            domain::creators::BinaryCompactObject(
                Object{
                    inner_radius_objectA, outer_radius_objectA, xcoord_objectA,
                    excise_interiorA ? std::make_optional(Excision{
                                           create_inner_boundary_condition()})
                                     : std::nullopt,
                    false},
                Object{
                    inner_radius_objectB, outer_radius_objectB, xcoord_objectB,
                    excise_interiorB ? std::make_optional(Excision{
                                           create_inner_boundary_condition()})
                                     : std::nullopt,
                    false},
                envelope_radius, outer_radius, refinement, grid_points,
                use_equiangular_map, radial_distribution_envelope,
                radial_distribution_outer_shell, opening_angle, nullptr,
                Options::Context{false, {}, 1, 1}),
            Catch::Matchers::Contains(
                "Must specify either both inner and outer boundary "
                "conditions or neither."));
        CHECK_THROWS_WITH(
            domain::creators::BinaryCompactObject(
                Object{inner_radius_objectA, outer_radius_objectA,
                       xcoord_objectA,
                       excise_interiorA ? std::make_optional(Excision{nullptr})
                                        : std::nullopt,
                       false},
                Object{inner_radius_objectB, outer_radius_objectB,
                       xcoord_objectB,
                       excise_interiorB ? std::make_optional(Excision{nullptr})
                                        : std::nullopt,
                       false},
                envelope_radius, outer_radius, refinement, grid_points,
                use_equiangular_map, radial_distribution_envelope,
                radial_distribution_outer_shell, opening_angle,
                create_outer_boundary_condition(),
                Options::Context{false, {}, 1, 1}),
            Catch::Matchers::Contains(
                "Must specify either both inner and outer boundary "
                "conditions or neither."));
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
  const std::string time_dependence{
      add_time_dependence ? "  TimeDependentMaps:\n"
                            "    InitialTime: 1.0\n"
                            "    ExpansionMap: \n"
                            "      InitialValues: [1.0, -0.1]\n"
                            "      AsymptoticVelocityOuterBoundary: -0.1\n"
                            "      DecayTimescaleOuterBoundaryVelocity: 5.0\n"
                            "    RotationMap:\n"
                            "      InitialAngularVelocity: [0.0, 0.0, -0.2]\n"
                            "    SizeMapA:\n"
                            "      InitialValues: [0.0, -0.1, 0.01]\n"
                            "    SizeMapB:\n"
                            "      InitialValues: [0.0, -0.2, 0.02]\n"
                            "    ShapeMapA:\n"
                            "      LMax: 8\n"
                            "    ShapeMapB:\n"
                            "      LMax: 8"
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
         "  InitialGridPoints: 3\n"
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

  // Envelope:
  constexpr double envelope_radius = 25.5;
  constexpr auto radial_distribution_envelope = Distribution::Projective;

  // Outer shell:
  constexpr double outer_radius = 32.4;
  constexpr double opening_angle = 90.0;

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
  const domain::creators::BinaryCompactObject binary_compact_object{
      CartesianCubeAtXCoord{xcoord_objectA},
      CartesianCubeAtXCoord{xcoord_objectB},
      envelope_radius,
      outer_radius,
      refinement,
      grid_points,
      use_equiangular_map,
      radial_distribution_envelope,
      radial_distribution_outer_shell,
      opening_angle,
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
            "EnvelopeLowerZRight", "EnvelopeUpperYRight", "EnvelopeLowerYRight",
            "EnvelopeUpperYLeft", "EnvelopeUpperZLeft", "EnvelopeLowerYLeft",
            "EnvelopeLowerX"}},
          {"OuterShell",
           {"OuterShellUpperZLeft", "OuterShellUpperZRight",
            "OuterShellLowerZLeft", "OuterShellLowerZRight",
            "OuterShellUpperYLeft", "OuterShellUpperYRight",
            "OuterShellLowerYLeft", "OuterShellLowerYRight", "OuterShellUpperX",
            "OuterShellLowerX"}}};
  CHECK(binary_compact_object.block_names() == expected_block_names);
  CHECK(binary_compact_object.block_groups() == expected_block_groups);
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
          create_option_string(true, true, true, false, true, 0, 0, 0, 120.0,
                               with_boundary_conditions));
    } else {
      return TestHelpers::test_option_tag<domain::OptionTags::DomainCreator<3>,
                                          Metavariables<3, true, false>>(
          create_option_string(true, true, true, false, true, 0, 0, 0, 120.0,
                               with_boundary_conditions));
    }
  }();

  const std::vector<double> times_to_check{{1., 2.3}};
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      *binary_compact_object, with_boundary_conditions, false, times_to_check);
}

void test_binary_factory() {
  MAKE_GENERATOR(gen);
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
  const bool add_time_dependence = false;
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
                   excise_A, excise_B, add_time_dependence, use_log_maps,
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
          25.5, 32.4, 2_st, 6_st, true, Distribution::Projective,
          Distribution::Linear, 120.0, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "The x-coordinate of ObjectA's center is expected to be positive."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.5, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          Object{0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          25.5, 32.4, 2_st, 6_st, true, Distribution::Projective,
          Distribution::Linear, 120.0, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "The x-coordinate of ObjectB's center is expected to be negative."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 1.0, 8.0, {{create_inner_boundary_condition()}}, false},
          Object{0.5, 1.0, -7.0, {{create_inner_boundary_condition()}}, false},
          25.5, 32.4, 2_st, 6_st, true, Distribution::Projective,
          Distribution::Linear, 120.0, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("The radius for the enveloping cube is too "
                                "small! The Frustums will be malformed."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          Object{1.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          25.5, 32.4, 2_st, 6_st, true, Distribution::Projective,
          Distribution::Linear, 120.0, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "ObjectB's inner radius must be less than its outer radius."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{3.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          Object{0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          25.5, 32.4, 2_st, 6_st, true, Distribution::Projective,
          Distribution::Linear, 120.0, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "ObjectA's inner radius must be less than its outer radius."));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          Object{0.5, 1.0, -1.0, std::nullopt, true}, 25.5, 32.4, 2_st, 6_st,
          true, Distribution::Projective, Distribution::Linear, 120.0,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Using a logarithmically spaced radial grid in the "
          "part of Layer 1 enveloping Object B requires excising the interior "
          "of Object B"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 1.0, 1.0, std::nullopt, true},
          Object{0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          25.5, 32.4, 2_st, 6_st, true, Distribution::Projective,
          Distribution::Linear, 120.0, create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "Using a logarithmically spaced radial grid in the "
          "part of Layer 1 enveloping Object A requires excising the interior "
          "of Object A"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          Object{0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          25.5, 32.4, std::vector<std::array<size_t, 3>>{}, 6_st, true,
          Distribution::Projective, Distribution::Linear, 120.0,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("Invalid 'InitialRefinement'"));
  CHECK_THROWS_WITH(
      domain::creators::BinaryCompactObject(
          Object{0.3, 1.0, 1.0, {{create_inner_boundary_condition()}}, false},
          Object{0.5, 1.0, -1.0, {{create_inner_boundary_condition()}}, false},
          25.5, 32.4, 2_st, std::vector<std::array<size_t, 3>>{}, true,
          Distribution::Projective, Distribution::Linear, 120.0,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("Invalid 'InitialGridPoints'"));
  // Note: the boundary condition-related parse errors are checked in the
  // test_connectivity function.
}
}  // namespace

// [[TimeOut, 30]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObject",
                  "[Domain][Unit]") {
  test_connectivity();
  test_bns_domain_with_cubes();
  test_bbh_time_dependent_factory(true, true);
  test_bbh_time_dependent_factory(true, false);
  test_bbh_time_dependent_factory(false, true);
  test_bbh_time_dependent_factory(false, false);
  test_binary_factory();
  test_parse_errors();
}
