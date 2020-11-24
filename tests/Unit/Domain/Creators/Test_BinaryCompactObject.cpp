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
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"

namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime

namespace {
using Translation = domain::CoordinateMaps::TimeDependent::Translation;
using Translation3D = domain::CoordinateMaps::TimeDependent::ProductOf3Maps<
    Translation, Translation, Translation>;

template <typename... FuncsOfTime>
void test_binary_compact_object_construction(
    const domain::creators::BinaryCompactObject& binary_compact_object,
    double time = std::numeric_limits<double>::signaling_NaN(),
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time = {},
    const std::tuple<std::pair<std::string, FuncsOfTime>...>&
        expected_functions_of_time = {}) {
  const auto domain = binary_compact_object.create_domain();
  test_initial_domain(domain,
                      binary_compact_object.initial_refinement_levels());
  test_physical_separation(binary_compact_object.create_domain().blocks(), time,
                           functions_of_time);

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

  for (const bool excise_interiorA : {true, false}) {
    for (const bool excise_interiorB : {true, false}) {
      for (const bool use_equiangular_map : {true, false}) {
        for (const bool use_logarithmic_map_outer_spherical_shell :
             {true, false}) {
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
              use_equiangular_map,
              use_projective_map,
              use_logarithmic_map_outer_spherical_shell,
              addition_to_outer_layer_radial_refinement_level};
          test_binary_compact_object_construction(binary_compact_object);

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
          const double layer_5_inner_radius = get(
              magnitude(std::move(map)->operator()(logical_point)));
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
        }
      }
  }
}
}
void test_bbh_time_dependent_factory() {
  const auto binary_compact_object =
      TestHelpers::test_factory_creation<DomainCreator<3>>(
          "  BinaryCompactObject:\n"
          "    InnerRadiusObjectA: 0.2\n"
          "    OuterRadiusObjectA: 1.0\n"
          "    XCoordObjectA: -2.0\n"
          "    ExciseInteriorA: true\n"
          "    InnerRadiusObjectB: 1.0\n"
          "    OuterRadiusObjectB: 2.0\n"
          "    XCoordObjectB: 3.0\n"
          "    ExciseInteriorB: true\n"
          "    RadiusOuterCube: 22.0\n"
          "    RadiusOuterSphere: 25.0\n"
          "    InitialRefinement: 1\n"
          "    InitialGridPoints: 3\n"
          "    UseEquiangularMap: true\n"
          "    TimeDependence:\n"
          "      UniformTranslation:\n"
          "        InitialTime: 1.0\n"
          "        InitialExpirationDeltaT: 9.0\n"
          "        Velocity: [2.3, -0.3, 0.5]\n"
          "        FunctionOfTimeNames: [TranslationX, TranslationY, "
          "TranslationZ]");
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
        time, functions_of_time, expected_functions_of_time);
  }
}

void test_bbh_equiangular_factory() {
  const auto binary_compact_object =
      TestHelpers::test_factory_creation<DomainCreator<3>>(
          "  BinaryCompactObject:\n"
          "    InnerRadiusObjectA: 0.2\n"
          "    OuterRadiusObjectA: 1.0\n"
          "    XCoordObjectA: -2.0\n"
          "    ExciseInteriorA: true\n"
          "    InnerRadiusObjectB: 1.0\n"
          "    OuterRadiusObjectB: 2.0\n"
          "    XCoordObjectB: 3.0\n"
          "    ExciseInteriorB: true\n"
          "    RadiusOuterCube: 22.0\n"
          "    RadiusOuterSphere: 25.0\n"
          "    InitialRefinement: 1\n"
          "    InitialGridPoints: 3\n"
          "    UseEquiangularMap: true\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_bbh_2_outer_radial_refinements_linear_map_factory() {
  const auto binary_compact_object =
      TestHelpers::test_factory_creation<DomainCreator<3>>(
          "  BinaryCompactObject:\n"
          "    InnerRadiusObjectA: 0.2\n"
          "    OuterRadiusObjectA: 1.0\n"
          "    XCoordObjectA: -2.0\n"
          "    ExciseInteriorA: true\n"
          "    InnerRadiusObjectB: 1.0\n"
          "    OuterRadiusObjectB: 2.0\n"
          "    XCoordObjectB: 3.0\n"
          "    ExciseInteriorB: true\n"
          "    RadiusOuterCube: 22.0\n"
          "    RadiusOuterSphere: 25.0\n"
          "    InitialRefinement: 1\n"
          "    InitialGridPoints: 3\n"
          "    AdditionToObjectBRadialRefinementLevel: 2\n"
          "    UseEquiangularMap: true\n"
          "    AdditionToOuterLayerRadialRefinementLevel: 2\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_bbh_3_outer_radial_refinements_log_map_factory() {
  const auto binary_compact_object =
      TestHelpers::test_factory_creation<DomainCreator<3>>(
          "  BinaryCompactObject:\n"
          "    InnerRadiusObjectA: 0.2\n"
          "    OuterRadiusObjectA: 1.0\n"
          "    XCoordObjectA: -2.0\n"
          "    ExciseInteriorA: true\n"
          "    InnerRadiusObjectB: 1.0\n"
          "    OuterRadiusObjectB: 2.0\n"
          "    XCoordObjectB: 3.0\n"
          "    ExciseInteriorB: true\n"
          "    RadiusOuterCube: 22.0\n"
          "    RadiusOuterSphere: 25.0\n"
          "    InitialRefinement: 1\n"
          "    InitialGridPoints: 3\n"
          "    UseEquiangularMap: true\n"
          "    UseLogarithmicMapObjectA: true\n"
          "    AdditionToObjectARadialRefinementLevel: 3\n"
          "    UseLogarithmicMapObjectB: true\n"
          "    AdditionToOuterLayerRadialRefinementLevel: 3");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_bbh_equidistant_factory() {
  const auto binary_compact_object =
      TestHelpers::test_factory_creation<DomainCreator<3>>(
          "  BinaryCompactObject:\n"
          "    InnerRadiusObjectA: 0.2\n"
          "    OuterRadiusObjectA: 1.0\n"
          "    XCoordObjectA: -2.0\n"
          "    ExciseInteriorA: true\n"
          "    InnerRadiusObjectB: 1.0\n"
          "    OuterRadiusObjectB: 2.0\n"
          "    XCoordObjectB: 3.0\n"
          "    ExciseInteriorB: true\n"
          "    RadiusOuterCube: 22.0\n"
          "    RadiusOuterSphere: 25.0\n"
          "    InitialRefinement: 1\n"
          "    InitialGridPoints: 3\n"
          "    UseEquiangularMap: false\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_bns_equiangular_factory() {
  const auto binary_compact_object =
      TestHelpers::test_factory_creation<DomainCreator<3>>(
          "  BinaryCompactObject:\n"
          "    InnerRadiusObjectA: 0.2\n"
          "    OuterRadiusObjectA: 1.0\n"
          "    XCoordObjectA: -2.0\n"
          "    ExciseInteriorA: false\n"
          "    InnerRadiusObjectB: 1.0\n"
          "    OuterRadiusObjectB: 2.0\n"
          "    XCoordObjectB: 3.0\n"
          "    ExciseInteriorB: false\n"
          "    RadiusOuterCube: 22.0\n"
          "    RadiusOuterSphere: 25.0\n"
          "    InitialRefinement: 1\n"
          "    InitialGridPoints: 3\n"
          "    UseEquiangularMap: true\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_bns_equidistant_factory() {
  const auto binary_compact_object =
      TestHelpers::test_factory_creation<DomainCreator<3>>(
          "  BinaryCompactObject:\n"
          "    InnerRadiusObjectA: 0.2\n"
          "    OuterRadiusObjectA: 1.0\n"
          "    XCoordObjectA: -2.0\n"
          "    ExciseInteriorA: false\n"
          "    InnerRadiusObjectB: 1.0\n"
          "    OuterRadiusObjectB: 2.0\n"
          "    XCoordObjectB: 3.0\n"
          "    ExciseInteriorB: false\n"
          "    RadiusOuterCube: 22.0\n"
          "    RadiusOuterSphere: 25.0\n"
          "    InitialRefinement: 1\n"
          "    InitialGridPoints: 3\n"
          "    UseEquiangularMap: false\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_bhns_equiangular_factory() {
  const auto binary_compact_object =
      TestHelpers::test_factory_creation<DomainCreator<3>>(
          "  BinaryCompactObject:\n"
          "    InnerRadiusObjectA: 0.2\n"
          "    OuterRadiusObjectA: 1.0\n"
          "    XCoordObjectA: -2.0\n"
          "    ExciseInteriorA: true\n"
          "    InnerRadiusObjectB: 1.0\n"
          "    OuterRadiusObjectB: 2.0\n"
          "    XCoordObjectB: 3.0\n"
          "    ExciseInteriorB: false\n"
          "    RadiusOuterCube: 22.0\n"
          "    RadiusOuterSphere: 25.0\n"
          "    InitialRefinement: 1\n"
          "    InitialGridPoints: 3\n"
          "    UseEquiangularMap: true\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_bhns_equidistant_factory() {
  const auto binary_compact_object =
      TestHelpers::test_factory_creation<DomainCreator<3>>(
          "  BinaryCompactObject:\n"
          "    InnerRadiusObjectA: 0.2\n"
          "    OuterRadiusObjectA: 1.0\n"
          "    XCoordObjectA: -2.0\n"
          "    ExciseInteriorA: true\n"
          "    InnerRadiusObjectB: 1.0\n"
          "    OuterRadiusObjectB: 2.0\n"
          "    XCoordObjectB: 3.0\n"
          "    ExciseInteriorB: false\n"
          "    RadiusOuterCube: 22.0\n"
          "    RadiusOuterSphere: 25.0\n"
          "    InitialRefinement: 1\n"
          "    InitialGridPoints: 3\n"
          "    UseEquiangularMap: false\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_nsbh_equiangular_factory() {
  const auto binary_compact_object =
      TestHelpers::test_factory_creation<DomainCreator<3>>(
          "  BinaryCompactObject:\n"
          "    InnerRadiusObjectA: 0.2\n"
          "    OuterRadiusObjectA: 1.0\n"
          "    XCoordObjectA: -2.0\n"
          "    ExciseInteriorA: false\n"
          "    InnerRadiusObjectB: 1.0\n"
          "    OuterRadiusObjectB: 2.0\n"
          "    XCoordObjectB: 3.0\n"
          "    ExciseInteriorB: true\n"
          "    RadiusOuterCube: 22.0\n"
          "    RadiusOuterSphere: 25.0\n"
          "    InitialRefinement: 1\n"
          "    InitialGridPoints: 3\n"
          "    UseEquiangularMap: true\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}

void test_nsbh_equidistant_factory() {
  const auto binary_compact_object =
      TestHelpers::test_factory_creation<DomainCreator<3>>(
          "  BinaryCompactObject:\n"
          "    InnerRadiusObjectA: 0.2\n"
          "    OuterRadiusObjectA: 1.0\n"
          "    XCoordObjectA: -2.0\n"
          "    ExciseInteriorA: false\n"
          "    InnerRadiusObjectB: 1.0\n"
          "    OuterRadiusObjectB: 2.0\n"
          "    XCoordObjectB: 3.0\n"
          "    ExciseInteriorB: true\n"
          "    RadiusOuterCube: 22.0\n"
          "    RadiusOuterSphere: 25.0\n"
          "    InitialRefinement: 1\n"
          "    InitialGridPoints: 3\n"
          "    UseEquiangularMap: false\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObject.FactoryTests",
                  "[Domain][Unit]") {
  test_connectivity();
  test_bbh_time_dependent_factory();
  test_bbh_2_outer_radial_refinements_linear_map_factory();
  test_bbh_3_outer_radial_refinements_log_map_factory();
  test_bbh_time_dependent_factory();
  test_bbh_equiangular_factory();
  test_bbh_equidistant_factory();
  test_bns_equiangular_factory();
  test_bns_equidistant_factory();
  test_bhns_equiangular_factory();
  test_bhns_equidistant_factory();
  test_nsbh_equiangular_factory();
  test_nsbh_equidistant_factory();
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
  const bool use_equiangular_map = true;

  domain::creators::BinaryCompactObject binary_compact_object{
      inner_radius_objectA,     outer_radius_objectA, xcoord_objectA,
      excise_interiorA,         inner_radius_objectB, outer_radius_objectB,
      xcoord_objectB,           excise_interiorB,     radius_enveloping_cube,
      radius_enveloping_sphere, refinement,           grid_points,
      use_equiangular_map};
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
  const bool use_equiangular_map = true;

  domain::creators::BinaryCompactObject binary_compact_object{
      inner_radius_objectA,     outer_radius_objectA, xcoord_objectA,
      excise_interiorA,         inner_radius_objectB, outer_radius_objectB,
      xcoord_objectB,           excise_interiorB,     radius_enveloping_cube,
      radius_enveloping_sphere, refinement,           grid_points,
      use_equiangular_map};
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
  const bool use_equiangular_map = true;

  domain::creators::BinaryCompactObject binary_compact_object{
      inner_radius_objectA,     outer_radius_objectA, xcoord_objectA,
      excise_interiorA,         inner_radius_objectB, outer_radius_objectB,
      xcoord_objectB,           excise_interiorB,     radius_enveloping_cube,
      radius_enveloping_sphere, refinement,           grid_points,
      use_equiangular_map};
}
