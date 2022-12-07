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
using Translation = domain::CoordinateMaps::TimeDependent::Translation<1>;
using Translation3D = domain::CoordinateMaps::TimeDependent::Translation<3>;
using BoundaryCondVector = std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>;
using ExpirationTimeMap = std::unordered_map<std::string, double>;

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

void test_connectivity_once(const bool with_sphere_e,
                            const bool include_inner_sphere_A,
                            const bool include_inner_sphere_B,
                            const bool include_outer_sphere) {
  // Misc.:
  constexpr size_t refinement = 1;
  constexpr size_t grid_points = 3;
  const double outer_radius = include_outer_sphere ? 100.0 : 30.0;

  const double mass_ratio = with_sphere_e ? 4 : 1.2;
  const double separation = 9.0;
  const double y_offset = 0.05;
  // Set centers so that the Newtonian COM is at the origin,
  // except offset slightly in the y direction.
  const std::array<double, 3> center_objectA = {separation / (1.0 + mass_ratio),
                                                y_offset, 0.0};
  const std::array<double, 3> center_objectB = {
      -separation * mass_ratio / (1.0 + mass_ratio), y_offset, 0.0};

  // Set radius = 2m for each object.
  // Formula comes from assuming m_A+m_B=1.
  const double inner_radius_objectA = 2.0 * mass_ratio / (1.0 + mass_ratio);
  const double inner_radius_objectB = 2.0 / (1.0 + mass_ratio);

  for (const bool with_boundary_conditions : {true, false}) {
    CAPTURE(with_boundary_conditions);
    const domain::creators::CylindricalBinaryCompactObject
        binary_compact_object{
            center_objectA,
            center_objectB,
            inner_radius_objectA,
            inner_radius_objectB,
            include_inner_sphere_A,
            include_inner_sphere_B,
            include_outer_sphere,
            outer_radius,
            refinement,
            grid_points,
            nullptr,
            with_boundary_conditions ? create_inner_boundary_condition()
                                     : nullptr,
            with_boundary_conditions ? create_outer_boundary_condition()
                                     : nullptr};

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
    std::unordered_map<std::string, std::unordered_set<std::string>>
        block_groups{
            {"Outer",
             {{"CAFilledCylinderCenter", "CBCylinderEast",
               "CAFilledCylinderEast", "CAFilledCylinderNorth",
               "CBFilledCylinderNorth", "CACylinderEast",
               "CBFilledCylinderEast", "CAFilledCylinderSouth",
               "CACylinderNorth", "CAFilledCylinderWest", "CACylinderWest",
               "CACylinderSouth", "CBFilledCylinderCenter",
               "CBFilledCylinderWest", "CBFilledCylinderSouth",
               "CBCylinderNorth", "CBCylinderWest", "CBCylinderSouth"}}},
            {"InnerA",
             {"EAFilledCylinderCenter", "MAFilledCylinderNorth",
              "EACylinderSouth", "EAFilledCylinderSouth", "EACylinderNorth",
              "EACylinderWest", "MAFilledCylinderCenter",
              "EAFilledCylinderNorth", "EAFilledCylinderWest",
              "MAFilledCylinderSouth", "EACylinderEast", "MAFilledCylinderEast",
              "EAFilledCylinderEast", "MAFilledCylinderWest"}},
            {"InnerB",
             {"EBFilledCylinderEast", "MBFilledCylinderEast",
              "EBFilledCylinderSouth", "EBFilledCylinderNorth",
              "EBFilledCylinderWest", "EBCylinderEast", "MBFilledCylinderWest",
              "EBCylinderNorth", "EBCylinderSouth", "EBCylinderWest",
              "MBFilledCylinderCenter", "EBFilledCylinderCenter",
              "MBFilledCylinderNorth", "MBFilledCylinderSouth"}}};

    if (include_inner_sphere_A) {
      block_names.insert(
          block_names.end(),
          {"InnerSphereEAFilledCylinderCenter",
           "InnerSphereEAFilledCylinderEast",
           "InnerSphereEAFilledCylinderNorth",
           "InnerSphereEAFilledCylinderWest",
           "InnerSphereEAFilledCylinderSouth",
           "InnerSphereMAFilledCylinderCenter",
           "InnerSphereMAFilledCylinderEast",
           "InnerSphereMAFilledCylinderNorth",
           "InnerSphereMAFilledCylinderWest",
           "InnerSphereMAFilledCylinderSouth", "InnerSphereEACylinderEast",
           "InnerSphereEACylinderNorth", "InnerSphereEACylinderWest",
           "InnerSphereEACylinderSouth"});
      block_groups.insert(
          {"InnerSphereA",
           {"InnerSphereEAFilledCylinderCenter",
            "InnerSphereEAFilledCylinderEast",
            "InnerSphereEAFilledCylinderNorth",
            "InnerSphereEAFilledCylinderWest",
            "InnerSphereEAFilledCylinderSouth",
            "InnerSphereMAFilledCylinderCenter",
            "InnerSphereMAFilledCylinderEast",
            "InnerSphereMAFilledCylinderNorth",
            "InnerSphereMAFilledCylinderWest",
            "InnerSphereMAFilledCylinderSouth", "InnerSphereEACylinderEast",
            "InnerSphereEACylinderNorth", "InnerSphereEACylinderWest",
            "InnerSphereEACylinderSouth"}});
    }
    if (include_inner_sphere_B) {
      block_names.insert(
          block_names.end(),
          {"InnerSphereEBFilledCylinderCenter",
           "InnerSphereEBFilledCylinderEast",
           "InnerSphereEBFilledCylinderNorth",
           "InnerSphereEBFilledCylinderWest",
           "InnerSphereEBFilledCylinderSouth",
           "InnerSphereMBFilledCylinderCenter",
           "InnerSphereMBFilledCylinderEast",
           "InnerSphereMBFilledCylinderNorth",
           "InnerSphereMBFilledCylinderWest",
           "InnerSphereMBFilledCylinderSouth", "InnerSphereEBCylinderEast",
           "InnerSphereEBCylinderNorth", "InnerSphereEBCylinderWest",
           "InnerSphereEBCylinderSouth"});
      block_groups.insert(
          {"InnerSphereB",
           {"InnerSphereEBFilledCylinderCenter",
            "InnerSphereEBFilledCylinderEast",
            "InnerSphereEBFilledCylinderNorth",
            "InnerSphereEBFilledCylinderWest",
            "InnerSphereEBFilledCylinderSouth",
            "InnerSphereMBFilledCylinderCenter",
            "InnerSphereMBFilledCylinderEast",
            "InnerSphereMBFilledCylinderNorth",
            "InnerSphereMBFilledCylinderWest",
            "InnerSphereMBFilledCylinderSouth", "InnerSphereEBCylinderEast",
            "InnerSphereEBCylinderNorth", "InnerSphereEBCylinderWest",
            "InnerSphereEBCylinderSouth"}});
    }
    if (include_outer_sphere) {
      block_names.insert(
          block_names.end(),
          {"OuterSphereCAFilledCylinderCenter",
           "OuterSphereCAFilledCylinderEast",
           "OuterSphereCAFilledCylinderNorth",
           "OuterSphereCAFilledCylinderWest",
           "OuterSphereCAFilledCylinderSouth",
           "OuterSphereCBFilledCylinderCenter",
           "OuterSphereCBFilledCylinderEast",
           "OuterSphereCBFilledCylinderNorth",
           "OuterSphereCBFilledCylinderWest",
           "OuterSphereCBFilledCylinderSouth", "OuterSphereCACylinderEast",
           "OuterSphereCACylinderNorth", "OuterSphereCACylinderWest",
           "OuterSphereCACylinderSouth", "OuterSphereCBCylinderEast",
           "OuterSphereCBCylinderNorth", "OuterSphereCBCylinderWest",
           "OuterSphereCBCylinderSouth"});
      block_groups.insert(
          {"OuterSphere",
           {"OuterSphereCAFilledCylinderCenter",
            "OuterSphereCAFilledCylinderEast",
            "OuterSphereCAFilledCylinderNorth",
            "OuterSphereCAFilledCylinderWest",
            "OuterSphereCAFilledCylinderSouth",
            "OuterSphereCBFilledCylinderCenter",
            "OuterSphereCBFilledCylinderEast",
            "OuterSphereCBFilledCylinderNorth",
            "OuterSphereCBFilledCylinderWest",
            "OuterSphereCBFilledCylinderSouth", "OuterSphereCACylinderEast",
            "OuterSphereCACylinderNorth", "OuterSphereCACylinderWest",
            "OuterSphereCACylinderSouth", "OuterSphereCBCylinderEast",
            "OuterSphereCBCylinderNorth", "OuterSphereCBCylinderWest",
            "OuterSphereCBCylinderSouth"}});
    }
    CHECK(binary_compact_object.block_names() == block_names);
    CHECK(binary_compact_object.block_groups() == block_groups);

    TestHelpers::domain::creators::test_domain_creator(
        binary_compact_object, with_boundary_conditions);

    if (with_boundary_conditions) {
      CHECK_THROWS_WITH(
          domain::creators::CylindricalBinaryCompactObject(
              center_objectA, center_objectB, inner_radius_objectA,
              inner_radius_objectB, include_inner_sphere_A,
              include_inner_sphere_B, include_outer_sphere, outer_radius,
              refinement, grid_points, nullptr,
              create_inner_boundary_condition(),
              std::make_unique<TestHelpers::domain::BoundaryConditions::
                                   TestPeriodicBoundaryCondition<3>>(),
              Options::Context{false, {}, 1, 1}),
          Catch::Matchers::Contains("Cannot have periodic boundary "
                                    "conditions with a binary domain"));
      CHECK_THROWS_WITH(
          domain::creators::CylindricalBinaryCompactObject(
              center_objectA, center_objectB, inner_radius_objectA,
              inner_radius_objectB, include_inner_sphere_A,
              include_inner_sphere_B, include_outer_sphere, outer_radius,
              refinement, grid_points, nullptr,
              std::make_unique<TestHelpers::domain::BoundaryConditions::
                                   TestPeriodicBoundaryCondition<3>>(),
              create_outer_boundary_condition(),
              Options::Context{false, {}, 1, 1}),
          Catch::Matchers::Contains("Cannot have periodic boundary "
                                    "conditions with a binary domain"));
      CHECK_THROWS_WITH(
          domain::creators::CylindricalBinaryCompactObject(
              center_objectA, center_objectB, inner_radius_objectA,
              inner_radius_objectB, include_inner_sphere_A,
              include_inner_sphere_B, include_outer_sphere, outer_radius,
              refinement, grid_points, nullptr, nullptr,
              create_outer_boundary_condition(),
              Options::Context{false, {}, 1, 1}),
          Catch::Matchers::Contains(
              "Must specify either both inner and outer boundary "
              "conditions or neither."));
      CHECK_THROWS_WITH(
          domain::creators::CylindricalBinaryCompactObject(
              center_objectA, center_objectB, inner_radius_objectA,
              inner_radius_objectB, include_inner_sphere_A,
              include_inner_sphere_B, include_outer_sphere, outer_radius,
              refinement, grid_points, nullptr,
              create_inner_boundary_condition(), nullptr,
              Options::Context{false, {}, 1, 1}),
          Catch::Matchers::Contains(
              "Must specify either both inner and outer boundary "
              "conditions or neither."));
    }
  }
}

void test_connectivity() {
  // When we add sphere_e support we will make the following
  // loop go over {true, false}
  for (const bool with_sphere_e : {false}) {
    CAPTURE(with_sphere_e);
    for (const bool include_outer_sphere : {true, false}) {
      CAPTURE(include_outer_sphere);
      for (const bool include_inner_sphere_A : {true, false}) {
        CAPTURE(include_inner_sphere_A);
        for (const bool include_inner_sphere_B : {true, false}) {
          CAPTURE(include_inner_sphere_B);
          test_connectivity_once(with_sphere_e, include_inner_sphere_A,
                                 include_inner_sphere_B, include_outer_sphere);
        }
      }
    }
  }
}

std::string create_option_string(
    const bool add_time_dependence,
    const bool with_additional_outer_radial_refinement,
    const bool with_additional_grid_points, const bool add_boundary_condition) {
  const std::string time_dependence{add_time_dependence
                                        ? "  TimeDependence:\n"
                                          "    UniformTranslation:\n"
                                          "      InitialTime: 1.0\n"
                                          "      Velocity: [2.3, -0.3, 0.5]\n"
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
         "  IncludeInnerSphereA: False\n"
         "  IncludeInnerSphereB: False\n"
         "  IncludeOuterSphere: False\n"
         "  InitialRefinement:" +
         (with_additional_outer_radial_refinement
              ? std::string(" 1\n")
              : std::string("\n    Outer: [2, 1, 1]"
                            "\n    InnerA: [1, 1, 1]"
                            "\n    InnerB: [2, 1, 1]\n")) +
         "  InitialGridPoints:" +
         (with_additional_grid_points
              ? std::string(" 3\n")
              : std::string("\n    Outer: [4, 3, 3]"
                            "\n    InnerA: [3, 3, 3]"
                            "\n    InnerB: [5, 3, 3]\n")) +
         time_dependence + boundary_conditions;
}

void test_bbh_time_dependent_factory(const bool with_boundary_conditions,
                                     const bool with_control_systems) {
  const auto binary_compact_object = [&with_boundary_conditions]() {
    if (with_boundary_conditions) {
      return TestHelpers::test_option_tag<
          domain::OptionTags::DomainCreator<3>,
          TestHelpers::domain::BoundaryConditions::
              MetavariablesWithBoundaryConditions<
                  3, domain::creators::CylindricalBinaryCompactObject>>(
          create_option_string(true, false, false, with_boundary_conditions));
    } else {
      return TestHelpers::test_option_tag<
          domain::OptionTags::DomainCreator<3>,
          TestHelpers::domain::BoundaryConditions::
              MetavariablesWithoutBoundaryConditions<
                  3, domain::creators::CylindricalBinaryCompactObject>>(
          create_option_string(true, false, false, with_boundary_conditions));
    }
  }();
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      *binary_compact_object, with_boundary_conditions);

  const std::array<double, 4> times_to_check{{0.0, 4.4, 7.8}};

  constexpr double initial_time = 0.0;
  constexpr double expected_time = 1.0;  // matches InitialTime: 1.0 above
  const DataVector velocity{{2.3, -0.3, 0.5}};
  // This name must match the hard coded one in UniformTranslation
  const std::string f_of_t_name = "Translation";
  std::array<DataVector, 3> function_of_time_coefficients{
      {{3, 0.0}, velocity, {3, 0.0}}};
  ExpirationTimeMap initial_expiration_times{};
  initial_expiration_times[f_of_t_name] =
      with_control_systems ? 10.0 : std::numeric_limits<double>::infinity();

  const std::tuple<
      std::pair<std::string, domain::FunctionsOfTime::PiecewisePolynomial<2>>>
      expected_functions_of_time = std::make_tuple(
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<2>>{
              f_of_t_name,
              {expected_time, function_of_time_coefficients,
               initial_expiration_times[f_of_t_name]}});
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time[f_of_t_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, function_of_time_coefficients,
          initial_expiration_times[f_of_t_name]);

  TestHelpers::domain::creators::test_functions_of_time(
      *binary_compact_object, expected_functions_of_time,
      initial_expiration_times);
  for (const double time : times_to_check) {
    test_det_jac_positive(domain.blocks(), time, functions_of_time);
    test_physical_separation(domain.blocks(), time, functions_of_time);
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
            TestHelpers::domain::BoundaryConditions::
                MetavariablesWithBoundaryConditions<
                    3, domain::creators::CylindricalBinaryCompactObject>>(
            opt_string);
      } else {
        return TestHelpers::test_option_tag<
            domain::OptionTags::DomainCreator<3>,
            TestHelpers::domain::BoundaryConditions::
                MetavariablesWithoutBoundaryConditions<
                    3, domain::creators::CylindricalBinaryCompactObject>>(
            opt_string);
      }
    }();
    TestHelpers::domain::creators::test_domain_creator(
        *binary_compact_object, with_boundary_conditions);
  };
  for (const bool with_boundary_conds : {true, false}) {
    for (const bool with_additional_outer_radial_refinement : {false, true}) {
      for (const bool with_additional_grid_points : {false, true}) {
        check_impl(create_option_string(
                       false, with_additional_outer_radial_refinement,
                       with_additional_grid_points, with_boundary_conds),
                   with_boundary_conds);
      }
    }
  }
}

void test_parse_errors() {
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 1.0, 0.4, false, false, false,
          1.0, 1_st, 3_st, nullptr, create_inner_boundary_condition(),
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("OuterRadius is too small"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{-2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 1.0, 0.4, false, false, false,
          25.0, 1_st, 3_st, nullptr, create_inner_boundary_condition(),
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "The x-coordinate of the input CenterA is expected to be positive"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {5.0, 0.05, 0.0}, 1.0, 0.4, false, false, false,
          25.0, 1_st, 3_st, nullptr, create_inner_boundary_condition(),
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains(
          "The x-coordinate of the input CenterB is expected to be negative"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, -1.0, 0.4, false, false, false,
          25.0, 1_st, 3_st, nullptr, create_inner_boundary_condition(),
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("RadiusA and RadiusB are expected "
                                "to be positive"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 1.0, -0.4, false, false, false,
          25.0, 1_st, 3_st, nullptr, create_inner_boundary_condition(),
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("RadiusA and RadiusB are expected "
                                "to be positive"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-5.0, 0.05, 0.0}, 0.15, 0.4, false, false, false,
          25.0, 1_st, 3_st, nullptr, create_inner_boundary_condition(),
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("RadiusA should not be smaller than RadiusB"));
  CHECK_THROWS_WITH(
      domain::creators::CylindricalBinaryCompactObject(
          {{2.0, 0.05, 0.0}}, {-1.0, 0.05, 0.0}, 1.0, 0.4, false, false, false,
          25.0, 1_st, 3_st, nullptr, create_inner_boundary_condition(),
          create_outer_boundary_condition(), Options::Context{false, {}, 1, 1}),
      Catch::Matchers::Contains("We expect |x_A| <= |x_B|"));
  // Note: the boundary condition-related parse errors are checked in the
  // test_connectivity function.
}
}  // namespace

// [[TimeOut, 40]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.CylindricalBinaryCompactObject",
                  "[Domain][Unit]") {
  test_connectivity();
  test_bbh_time_dependent_factory(true, true);
  test_bbh_time_dependent_factory(true, false);
  test_bbh_time_dependent_factory(false, true);
  test_bbh_time_dependent_factory(false, false);
  test_binary_factory();
  test_parse_errors();
}
