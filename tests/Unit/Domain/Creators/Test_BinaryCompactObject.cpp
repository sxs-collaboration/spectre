// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
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
#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Domain/Block.hpp"                       // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"               // IWYU pragma: keep
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"

namespace domain {
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain

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
  const double inner_radius_objectA = 0.5;
  const double outer_radius_objectA = 1.0;
  const double xcoord_objectA = -3.0;

  // ObjectB:
  const double inner_radius_objectB = 0.3;
  const double outer_radius_objectB = 1.0;
  const double xcoord_objectB = 3.0;

  // Enveloping Cube:
  const double radius_enveloping_cube = 25.5;
  const double radius_enveloping_sphere = 32.4;

  // Misc.:
  const size_t refinement = 2;
  const size_t grid_points = 6;

  for (const bool excise_interiorA : {true, false}) {
    for (const bool excise_interiorB : {true, false}) {
      for (const bool use_equiangular_map : {true, false}) {
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
            use_equiangular_map};
        test_binary_compact_object_construction(binary_compact_object);
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
          "    InitialRefinement: 2\n"
          "    InitialGridPoints: 6\n"
          "    UseEquiangularMap: true\n"
          "    TimeDependence:\n"
          "      UniformTranslation:\n"
          "        InitialTime: 1.0\n"
          "        Velocity: [2.3, -0.3, 0.5]\n"
          "        FunctionOfTimeNames: [TranslationX, TranslationY, "
          "TranslationZ]");
  const std::array<double, 4> times_to_check{{0.0, 4.4, 7.8}};

  constexpr double initial_time = 0.0;
  constexpr double expected_time = 1.0; // matches InitialTime: 1.0 above
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
              {expected_time, function_of_time_coefficients_x}},
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<2>>{
              "TranslationY"s,
              {expected_time, function_of_time_coefficients_y}},
          std::pair<std::string,
                    domain::FunctionsOfTime::PiecewisePolynomial<2>>{
              "TranslationZ"s,
              {expected_time, function_of_time_coefficients_z}});
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time["TranslationX"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, function_of_time_coefficients_x);
  functions_of_time["TranslationY"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, function_of_time_coefficients_y);
  functions_of_time["TranslationZ"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, function_of_time_coefficients_z);

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
          "    InitialRefinement: 2\n"
          "    InitialGridPoints: 6\n"
          "    UseEquiangularMap: true\n");
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
          "    InitialRefinement: 2\n"
          "    InitialGridPoints: 6\n"
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
          "    InitialRefinement: 2\n"
          "    InitialGridPoints: 6\n"
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
          "    InitialRefinement: 2\n"
          "    InitialGridPoints: 6\n"
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
          "    InitialRefinement: 2\n"
          "    InitialGridPoints: 6\n"
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
          "    InitialRefinement: 2\n"
          "    InitialGridPoints: 6\n"
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
          "    InitialRefinement: 2\n"
          "    InitialGridPoints: 6\n"
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
          "    InitialRefinement: 2\n"
          "    InitialGridPoints: 6\n"
          "    UseEquiangularMap: false\n");
  test_binary_compact_object_construction(
      dynamic_cast<const domain::creators::BinaryCompactObject&>(
          *binary_compact_object));
}
}  // namespace

// Test times out sometimes, increase timeout to make it pass reliably.
// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.Domain.Creators.BinaryCompactObject.FactoryTests",
                  "[Domain][Unit]") {
  test_connectivity();
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
