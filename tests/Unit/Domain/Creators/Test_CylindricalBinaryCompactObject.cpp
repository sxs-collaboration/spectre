// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <pup.h>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Creators/CylindricalBinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using ExpirationTimeMap = std::unordered_map<std::string, double>;
using CylBCO = ::domain::creators::CylindricalBinaryCompactObject;
using TimeDepOptions = domain::creators::bco::TimeDependentMapOptions<true>;

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

std::pair<std::vector<std::string>,
          std::unordered_map<std::string, std::unordered_set<std::string>>>
block_names_and_groups(const bool include_inner_sphere_A,
                       const bool include_inner_sphere_B,
                       const bool include_outer_sphere) {
  std::vector<std::string> block_names{
      "CAFilledCylinderCenter", "CAFilledCylinderEast",
      "CAFilledCylinderNorth",  "CAFilledCylinderWest",
      "CAFilledCylinderSouth",  "CACylinderEast",
      "CACylinderNorth",        "CACylinderWest",
      "CACylinderSouth",        "EAFilledCylinderCenter",
      "EAFilledCylinderEast",   "EAFilledCylinderNorth",
      "EAFilledCylinderWest",   "EAFilledCylinderSouth",
      "EACylinderEast",         "EACylinderNorth",
      "EACylinderWest",         "EACylinderSouth",
      "EBFilledCylinderCenter", "EBFilledCylinderEast",
      "EBFilledCylinderNorth",  "EBFilledCylinderWest",
      "EBFilledCylinderSouth",  "EBCylinderEast",
      "EBCylinderNorth",        "EBCylinderWest",
      "EBCylinderSouth",        "MAFilledCylinderCenter",
      "MAFilledCylinderEast",   "MAFilledCylinderNorth",
      "MAFilledCylinderWest",   "MAFilledCylinderSouth",
      "MBFilledCylinderCenter", "MBFilledCylinderEast",
      "MBFilledCylinderNorth",  "MBFilledCylinderWest",
      "MBFilledCylinderSouth",  "CBFilledCylinderCenter",
      "CBFilledCylinderEast",   "CBFilledCylinderNorth",
      "CBFilledCylinderWest",   "CBFilledCylinderSouth",
      "CBCylinderEast",         "CBCylinderNorth",
      "CBCylinderWest",         "CBCylinderSouth"};
  std::unordered_map<std::string, std::unordered_set<std::string>> block_groups{
      {"Outer",
       {{"CAFilledCylinderCenter", "CBCylinderEast", "CAFilledCylinderEast",
         "CAFilledCylinderNorth", "CBFilledCylinderNorth", "CACylinderEast",
         "CBFilledCylinderEast", "CAFilledCylinderSouth", "CACylinderNorth",
         "CAFilledCylinderWest", "CACylinderWest", "CACylinderSouth",
         "CBFilledCylinderCenter", "CBFilledCylinderWest",
         "CBFilledCylinderSouth", "CBCylinderNorth", "CBCylinderWest",
         "CBCylinderSouth"}}},
      {"InnerA",
       {"EAFilledCylinderCenter", "MAFilledCylinderNorth", "EACylinderSouth",
        "EAFilledCylinderSouth", "EACylinderNorth", "EACylinderWest",
        "MAFilledCylinderCenter", "EAFilledCylinderNorth",
        "EAFilledCylinderWest", "MAFilledCylinderSouth", "EACylinderEast",
        "MAFilledCylinderEast", "EAFilledCylinderEast",
        "MAFilledCylinderWest"}},
      {"InnerB",
       {"EBFilledCylinderEast", "MBFilledCylinderEast", "EBFilledCylinderSouth",
        "EBFilledCylinderNorth", "EBFilledCylinderWest", "EBCylinderEast",
        "MBFilledCylinderWest", "EBCylinderNorth", "EBCylinderSouth",
        "EBCylinderWest", "MBFilledCylinderCenter", "EBFilledCylinderCenter",
        "MBFilledCylinderNorth", "MBFilledCylinderSouth"}}};

  if (include_inner_sphere_A) {
    block_names.insert(
        block_names.end(),
        {"InnerSphereEAFilledCylinderCenter", "InnerSphereEAFilledCylinderEast",
         "InnerSphereEAFilledCylinderNorth", "InnerSphereEAFilledCylinderWest",
         "InnerSphereEAFilledCylinderSouth",
         "InnerSphereMAFilledCylinderCenter", "InnerSphereMAFilledCylinderEast",
         "InnerSphereMAFilledCylinderNorth", "InnerSphereMAFilledCylinderWest",
         "InnerSphereMAFilledCylinderSouth", "InnerSphereEACylinderEast",
         "InnerSphereEACylinderNorth", "InnerSphereEACylinderWest",
         "InnerSphereEACylinderSouth"});
    block_groups.insert(
        {"InnerSphereA",
         {"InnerSphereEAFilledCylinderCenter",
          "InnerSphereEAFilledCylinderEast", "InnerSphereEAFilledCylinderNorth",
          "InnerSphereEAFilledCylinderWest", "InnerSphereEAFilledCylinderSouth",
          "InnerSphereMAFilledCylinderCenter",
          "InnerSphereMAFilledCylinderEast", "InnerSphereMAFilledCylinderNorth",
          "InnerSphereMAFilledCylinderWest", "InnerSphereMAFilledCylinderSouth",
          "InnerSphereEACylinderEast", "InnerSphereEACylinderNorth",
          "InnerSphereEACylinderWest", "InnerSphereEACylinderSouth"}});
  }
  if (include_inner_sphere_B) {
    block_names.insert(
        block_names.end(),
        {"InnerSphereEBFilledCylinderCenter", "InnerSphereEBFilledCylinderEast",
         "InnerSphereEBFilledCylinderNorth", "InnerSphereEBFilledCylinderWest",
         "InnerSphereEBFilledCylinderSouth",
         "InnerSphereMBFilledCylinderCenter", "InnerSphereMBFilledCylinderEast",
         "InnerSphereMBFilledCylinderNorth", "InnerSphereMBFilledCylinderWest",
         "InnerSphereMBFilledCylinderSouth", "InnerSphereEBCylinderEast",
         "InnerSphereEBCylinderNorth", "InnerSphereEBCylinderWest",
         "InnerSphereEBCylinderSouth"});
    block_groups.insert(
        {"InnerSphereB",
         {"InnerSphereEBFilledCylinderCenter",
          "InnerSphereEBFilledCylinderEast", "InnerSphereEBFilledCylinderNorth",
          "InnerSphereEBFilledCylinderWest", "InnerSphereEBFilledCylinderSouth",
          "InnerSphereMBFilledCylinderCenter",
          "InnerSphereMBFilledCylinderEast", "InnerSphereMBFilledCylinderNorth",
          "InnerSphereMBFilledCylinderWest", "InnerSphereMBFilledCylinderSouth",
          "InnerSphereEBCylinderEast", "InnerSphereEBCylinderNorth",
          "InnerSphereEBCylinderWest", "InnerSphereEBCylinderSouth"}});
  }
  if (include_outer_sphere) {
    block_names.insert(
        block_names.end(),
        {"OuterSphereCAFilledCylinderCenter", "OuterSphereCAFilledCylinderEast",
         "OuterSphereCAFilledCylinderNorth", "OuterSphereCAFilledCylinderWest",
         "OuterSphereCAFilledCylinderSouth",
         "OuterSphereCBFilledCylinderCenter", "OuterSphereCBFilledCylinderEast",
         "OuterSphereCBFilledCylinderNorth", "OuterSphereCBFilledCylinderWest",
         "OuterSphereCBFilledCylinderSouth", "OuterSphereCACylinderEast",
         "OuterSphereCACylinderNorth", "OuterSphereCACylinderWest",
         "OuterSphereCACylinderSouth", "OuterSphereCBCylinderEast",
         "OuterSphereCBCylinderNorth", "OuterSphereCBCylinderWest",
         "OuterSphereCBCylinderSouth"});
    block_groups.insert(
        {"OuterSphere",
         {"OuterSphereCAFilledCylinderCenter",
          "OuterSphereCAFilledCylinderEast", "OuterSphereCAFilledCylinderNorth",
          "OuterSphereCAFilledCylinderWest", "OuterSphereCAFilledCylinderSouth",
          "OuterSphereCBFilledCylinderCenter",
          "OuterSphereCBFilledCylinderEast", "OuterSphereCBFilledCylinderNorth",
          "OuterSphereCBFilledCylinderWest", "OuterSphereCBFilledCylinderSouth",
          "OuterSphereCACylinderEast", "OuterSphereCACylinderNorth",
          "OuterSphereCACylinderWest", "OuterSphereCACylinderSouth",
          "OuterSphereCBCylinderEast", "OuterSphereCBCylinderNorth",
          "OuterSphereCBCylinderWest", "OuterSphereCBCylinderSouth"}});
  }

  return std::make_pair(block_names, block_groups);
}

std::string stringize(const bool t) { return t ? "true" : "false"; }

static constexpr int precision = std::numeric_limits<double>::max_digits10;

std::string stringize(const double t) {
  std::stringstream ss{};
  ss << std::setprecision(precision) << t;
  return ss.str();
}

std::string stringize(const std::array<double, 3>& t) {
  std::stringstream result{};
  result << std::setprecision(precision) << "[";
  bool first = true;
  for (const auto& item : t) {
    if (not first) {
      result << ", ";
    }
    result << item;
    first = false;
  }
  result << "]";
  return result.str();
}

std::string create_option_string(
    const bool add_time_dependence,
    const bool with_additional_outer_radial_refinement,
    const bool with_additional_grid_points, const bool include_outer_sphere,
    const bool include_inner_sphere_A, const bool include_inner_sphere_B,
    const bool add_boundary_condition, const bool use_equiangular_map,
    const std::array<double, 3>& center_objectA,
    const std::array<double, 3>& center_objectB,
    const double inner_radius_objectA, const double inner_radius_objectB,
    const double outer_radius) {
  const std::string time_dependence{
      add_time_dependence
          ? "  TimeDependentMaps:\n"
            "    InitialTime: 1.0\n"
            "    ExpansionMap: None\n"
            "    RotationMap:\n"
            "      InitialQuaternions: [[1.0, 0.0, 0.0, 0.0]]\n"
            "      InitialAngles: [[0.0, 0.0, 0.0], [0.0, 0.0, -0.2]]\n"
            "      DecayTimescale: Auto\n"
            "    TranslationMap: None\n"
            "    ShapeMapA:\n"
            "      LMax: 8\n"
            "      InitialValues: Spherical\n"
            "      SizeInitialValues: [1.1, 0.0, 0.0]\n"
            "    ShapeMapB:\n"
            "      LMax: 8\n"
            "      InitialValues: Spherical\n"
            "      SizeInitialValues: [1.2, 0.0, 0.0]\n"
          : "  TimeDependentMaps: None\n"};

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

  const auto initial_structure =
      [&include_outer_sphere, &include_inner_sphere_A, &include_inner_sphere_B](
          const bool include_extra, const size_t value) {
        const std::string same = "[" + get_output(value) + "," +
                                 get_output(value) + "," + get_output(value) +
                                 "]";
        const std::string one_more = "[" + get_output(value + 1) + "," +
                                     get_output(value) + "," +
                                     get_output(value) + "]";
        std::string result{};
        if (include_extra) {
          result += "\n    Outer: " + one_more;
          result += "\n    InnerA: " + same;
          result += "\n    InnerB: " + one_more;
          if (include_outer_sphere) {
            result += "\n    OuterSphere: " + one_more;
          }
          if (include_inner_sphere_A) {
            result += "\n    InnerSphereA: " + same;
          }
          if (include_inner_sphere_B) {
            result += "\n    InnerSphereB: " + same;
          }
        } else {
          result = " " + get_output(value);
        }
        return result;
      };

  return "CylindricalBinaryCompactObject:"
         "\n  CenterA: " +
         stringize(center_objectA) +
         "\n  RadiusA: " + stringize(inner_radius_objectA) +
         "\n  CenterB: " + stringize(center_objectB) +
         "\n  RadiusB: " + stringize(inner_radius_objectB) +
         "\n  OuterRadius: " + stringize(outer_radius) +
         "\n  UseEquiangularMap: " + stringize(use_equiangular_map) +
         "\n  IncludeInnerSphereA: " + stringize(include_inner_sphere_A) +
         "\n  IncludeInnerSphereB: " + stringize(include_inner_sphere_B) +
         "\n  IncludeOuterSphere: " + stringize(include_outer_sphere) +
         "\n  InitialRefinement:" +
         initial_structure(with_additional_outer_radial_refinement, 1) +
         "\n  InitialGridPoints:" +
         initial_structure(with_additional_grid_points, 3) + "\n" +
         time_dependence + boundary_conditions;
}

void test_construction(const CylBCO& creator,
                       const bool with_boundary_conditions,
                       const bool include_inner_sphere_A,
                       const bool include_inner_sphere_B,
                       const bool include_outer_sphere,
                       const double inner_radius_objectA,
                       const double inner_radius_objectB,
                       const std::array<double, 3>& center_objectA,
                       const std::array<double, 3>& center_objectB,
                       const std::vector<double>& times_to_check) {
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      creator, with_boundary_conditions, false, times_to_check);

  const auto& [block_names, block_groups] = block_names_and_groups(
      include_inner_sphere_A, include_inner_sphere_B, include_outer_sphere);

  CHECK(creator.block_names() == block_names);
  CHECK(creator.block_groups() == block_groups);

  CHECK(domain.excision_spheres().size() == 2);
  const auto& excision_sphere_a =
      domain.excision_spheres().at("ExcisionSphereA");
  CHECK(excision_sphere_a.radius() == inner_radius_objectA);
  const auto& excision_sphere_b =
      domain.excision_spheres().at("ExcisionSphereB");
  CHECK(excision_sphere_b.radius() == inner_radius_objectB);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(excision_sphere_a.center().get(i) == center_objectA.at(i));
    CHECK(excision_sphere_b.center().get(i) == center_objectB.at(i));
  }
  const auto& block = domain.blocks()[0];
  CHECK(block.is_time_dependent() == excision_sphere_a.is_time_dependent());
  CHECK(block.is_time_dependent() == excision_sphere_b.is_time_dependent());

  if (block.is_time_dependent()) {
    // Taken from option string above
    const double initial_time = 1.0;
    const double time = 2.0;
    const double z_angle = (time - initial_time) * -0.2;
    const auto functions_of_time = creator.functions_of_time();
    const auto& center_a = excision_sphere_a.center();
    const auto& center_b = excision_sphere_b.center();
    const auto& map_a = excision_sphere_a.moving_mesh_grid_to_inertial_map();
    const auto& map_b = excision_sphere_b.moving_mesh_grid_to_inertial_map();

    const auto mapped_point_a = map_a(center_a, time, functions_of_time);
    const auto mapped_point_b = map_b(center_b, time, functions_of_time);

    // Just a rotation
    const tnsr::I<double, 3> expected_mapped_point_a{
        {get<0>(center_a) * cos(z_angle) - get<1>(center_a) * sin(z_angle),
         get<0>(center_a) * sin(z_angle) + get<1>(center_a) * cos(z_angle),
         get<2>(center_a)}};
    const tnsr::I<double, 3> expected_mapped_point_b{
        {get<0>(center_b) * cos(z_angle) - get<1>(center_b) * sin(z_angle),
         get<0>(center_b) * sin(z_angle) + get<1>(center_b) * cos(z_angle),
         get<2>(center_b)}};

    CHECK_ITERABLE_APPROX(expected_mapped_point_a, mapped_point_a);
    CHECK_ITERABLE_APPROX(expected_mapped_point_b, mapped_point_b);
  }
}

TimeDepOptions construct_time_dependent_options() {
  constexpr double expected_time = 1.0;  // matches InitialTime: 1.0 above

  // Matches RotationMap:InitialAngularVelocity above
  const DataVector initial_angular_velocity{{0.0, 0.0, -0.2}};

  // Hardcoded in CylindricalBinaryCompactObject.cpp
  std::array<DataVector, 1> initial_quaternion_coefs{{{1.0, 0.0, 0.0, 0.0}}};

  // Rotation map has internally another FunctionOfTime for the
  // rotation angle.
  std::array<DataVector, 4> initial_rotation_angle_coefs{
      {{3, 0.0}, initial_angular_velocity, {3, 0.0}, {3, 0.0}}};

  // Matches SizeMap{A,B}::InitialValues above
  std::array<DataVector, 4> initial_size_A_coefs{{{1.1}, {0.0}, {0.0}, {0.0}}};
  std::array<DataVector, 4> initial_size_B_coefs{{{1.2}, {0.0}, {0.0}, {0.0}}};

  // No expansion map options because of above option string
  return TimeDepOptions{
      expected_time,
      std::nullopt,
      TimeDepOptions::RotationMapOptions{
          std::vector{std::array{10.0, 0.0, 0.0, 0.0}},
          std::vector{std::array{0.0, 0.0, 0.0},
                      std::array{initial_angular_velocity[0],
                                 initial_angular_velocity[1],
                                 initial_angular_velocity[2]}},
          std::nullopt},
      std::nullopt,
      TimeDepOptions::ShapeMapOptions<domain::ObjectLabel::A>{
          8_st,
          std::nullopt,
          {{initial_size_A_coefs[0][0], initial_size_A_coefs[1][0],
            initial_size_A_coefs[1][0]}}},
      TimeDepOptions::ShapeMapOptions<domain::ObjectLabel::B>{
          8_st,
          std::nullopt,
          {{initial_size_B_coefs[0][0], initial_size_B_coefs[1][0],
            initial_size_B_coefs[1][0]}}}};
}

void test_parse_errors() {
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 1.0, 0.4, false, false, false,
          1.0, false, 1_st, 3_st, std::nullopt,
          create_inner_boundary_condition(), create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("OuterRadius is too small"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{-2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 1.0, 0.4, false, false, false,
          25.0, false, 1_st, 3_st, std::nullopt,
          create_inner_boundary_condition(), create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "The x-coordinate of the input CenterA is expected to be positive"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {5.0, 0.05, 0.0}, 1.0, 0.4, false, false, false,
          25.0, false, 1_st, 3_st, std::nullopt,
          create_inner_boundary_condition(), create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "The x-coordinate of the input CenterB is expected to be negative"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, -1.0, 0.4, false, false, false,
          25.0, false, 1_st, 3_st, std::nullopt,
          create_inner_boundary_condition(), create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("RadiusA and RadiusB are expected "
                                         "to be positive"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 1.0, -0.4, false, false, false,
          25.0, false, 1_st, 3_st, std::nullopt,
          create_inner_boundary_condition(), create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("RadiusA and RadiusB are expected "
                                         "to be positive"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 0.15, 0.4, false, false, false,
          25.0, false, 1_st, 3_st, std::nullopt,
          create_inner_boundary_condition(), create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "RadiusA should not be smaller than RadiusB"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-1.0, 0.05, 0.0}, 1.0, 0.4, false, false, false,
          25.0, false, 1_st, 3_st, std::nullopt,
          create_inner_boundary_condition(), create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("We expect |x_A| <= |x_B|"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{4.0, 0.0, 0.0}}, {-4.0, 0.0, 0.0}, 1.0, 1.0, false, false, false,
          25.0, false, 1_st, 3_st,
          TimeDepOptions{0.0, std::nullopt,
                         TimeDepOptions::RotationMapOptions{
                             std::vector{std::array{1.0, 0.0, 0.0, 0.0}},
                             std::nullopt, std::nullopt},
                         std::nullopt, std::nullopt, std::nullopt},
          create_inner_boundary_condition(), create_outer_boundary_condition(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "To use the CylindricalBBH domain with time-dependent maps"));
  // Boundary condition errors
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 1.0, 0.4, false, false, false,
          25.0, false, 1_st, 3_st, std::nullopt,
          create_inner_boundary_condition(),
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("Cannot have periodic boundary "
                                         "conditions with a binary domain"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 1.0, 0.4, false, false, false,
          25.0, false, 1_st, 3_st, std::nullopt,
          std::make_unique<TestHelpers::domain::BoundaryConditions::
                               TestPeriodicBoundaryCondition<3>>(),
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring("Cannot have periodic boundary "
                                         "conditions with a binary domain"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 1.0, 0.4, false, false, false,
          25.0, false, 1_st, 3_st, std::nullopt, nullptr,
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Must specify either both inner and outer boundary "
          "conditions or neither."));
  CHECK_THROWS_WITH(domain::creators::CylindricalBinaryCompactObject(
                        {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 1.0, 0.4, false,
                        false, false, 25.0, false, 1_st, 3_st, std::nullopt,
                        create_inner_boundary_condition(), nullptr,
                        Options::Context{false, {}, 1, 1}),
                    Catch::Matchers::ContainsSubstring(
                        "Must specify either both inner and outer boundary "
                        "conditions or neither."));
}

// This matches the structure in the option string
std::unordered_map<std::string, std::array<size_t, 3>> make_initial_structure(
    const size_t initial_value, const bool include_inner_sphere_A,
    const bool include_inner_sphere_B, const bool include_outer_sphere) {
  std::unordered_map<std::string, std::array<size_t, 3>> initial_map;
  const std::array<size_t, 3> same{initial_value, initial_value, initial_value};
  const std::array<size_t, 3> one_more{initial_value + 1, initial_value,
                                       initial_value};
  initial_map["Outer"] = one_more;
  initial_map["InnerA"] = same;
  initial_map["InnerB"] = one_more;
  if (include_inner_sphere_A) {
    initial_map["InnerSphereA"] = same;
  }
  if (include_inner_sphere_B) {
    initial_map["InnerSphereB"] = same;
  }
  if (include_outer_sphere) {
    initial_map["OuterSphere"] = one_more;
  }

  return initial_map;
}

void test_cylindrical_bbh() {
  MAKE_GENERATOR(gen);

  const std::vector<double> times_to_check{{1.0, 2.3}};

  constexpr size_t refinement = 1;
  constexpr size_t grid_points = 3;

  const double separation = 9.0;
  const double y_offset = 0.05;

  // When we add sphere_e support we will make the following
  // loop go over {true, false}
  const bool with_sphere_e = false;
  for (auto [include_outer_sphere, include_inner_sphere_A,
             include_inner_sphere_B, use_equiangular_map,
             with_additional_outer_radial_refinement,
             with_additional_grid_points, with_time_dependence,
             with_control_systems, with_boundary_conditions] :
       random_sample<5>(
           cartesian_product(make_array(true, false), make_array(true, false),
                             make_array(true, false), make_array(true, false),
                             make_array(true, false), make_array(true, false),
                             make_array(true, false), make_array(true, false),
                             make_array(true, false)),
           make_not_null(&gen))) {
    CAPTURE(with_sphere_e);
    CAPTURE(include_outer_sphere);
    CAPTURE(use_equiangular_map);
    CAPTURE(with_boundary_conditions);
    CAPTURE(with_additional_outer_radial_refinement);
    CAPTURE(with_additional_grid_points);
    CAPTURE(with_time_dependence);
    if (with_time_dependence) {
      include_inner_sphere_A = true;
      include_inner_sphere_B = true;
      include_outer_sphere = true;
    } else {
      // With no time dependence, can't have control systems
      with_control_systems = false;
    }
    CAPTURE(include_inner_sphere_A);
    CAPTURE(include_inner_sphere_B);
    CAPTURE(with_control_systems);

    const double outer_radius = include_outer_sphere ? 100.0 : 30.0;
    const double mass_ratio = with_sphere_e ? 4 : 1.2;
    // Set centers so that the Newtonian COM is at the origin,
    // except offset slightly in the y direction.
    const std::array<double, 3> center_objectA = {
        separation / (1.0 + mass_ratio), y_offset, 0.0};
    const std::array<double, 3> center_objectB = {
        -separation * mass_ratio / (1.0 + mass_ratio), y_offset, 0.0};

    // Set radius = 2m for each object.
    // Formula comes from assuming m_A+m_B=1.
    const double inner_radius_objectA = 2.0 * mass_ratio / (1.0 + mass_ratio);
    const double inner_radius_objectB = 2.0 / (1.0 + mass_ratio);

    CylBCO::InitialRefinement::type initial_refinement{};
    CylBCO::InitialGridPoints::type initial_grid_points{};

    if (with_additional_outer_radial_refinement) {
      initial_refinement =
          make_initial_structure(refinement, include_inner_sphere_A,
                                 include_inner_sphere_B, include_outer_sphere);
    } else {
      initial_refinement = refinement;
    }
    if (with_additional_grid_points) {
      initial_grid_points =
          make_initial_structure(grid_points, include_inner_sphere_A,
                                 include_inner_sphere_B, include_outer_sphere);
    } else {
      initial_grid_points = grid_points;
    }

    CylBCO cyl_binary_compact_object{};
    std::optional<TimeDepOptions> time_dep_opts{};
    if (with_time_dependence) {
      time_dep_opts = construct_time_dependent_options();
    }
    cyl_binary_compact_object = CylBCO{
        center_objectA,
        center_objectB,
        inner_radius_objectA,
        inner_radius_objectB,
        include_inner_sphere_A,
        include_inner_sphere_B,
        include_outer_sphere,
        outer_radius,
        use_equiangular_map,
        initial_refinement,
        initial_grid_points,
        std::move(time_dep_opts),
        with_boundary_conditions ? create_inner_boundary_condition() : nullptr,
        with_boundary_conditions ? create_outer_boundary_condition() : nullptr};

    test_construction(cyl_binary_compact_object, with_boundary_conditions,
                      include_inner_sphere_A, include_inner_sphere_B,
                      include_outer_sphere, inner_radius_objectA,
                      inner_radius_objectB, center_objectA, center_objectB,
                      times_to_check);
    TestHelpers::domain::creators::test_creation(
        create_option_string(
            with_time_dependence, with_additional_outer_radial_refinement,
            with_additional_grid_points, include_outer_sphere,
            include_inner_sphere_A, include_inner_sphere_B,
            with_boundary_conditions, use_equiangular_map, center_objectA,
            center_objectB, inner_radius_objectA, inner_radius_objectB,
            outer_radius),
        cyl_binary_compact_object, with_boundary_conditions);
  }
}
}  // namespace

// [[TimeOut, 80]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.CylindricalBinaryCompactObject",
                  "[Domain][Unit]") {
  test_cylindrical_bbh();
  test_parse_errors();
}
