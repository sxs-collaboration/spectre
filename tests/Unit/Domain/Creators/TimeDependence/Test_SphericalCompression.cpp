// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/Creators/TimeDependence/SphericalCompression.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/Creators/TimeDependence/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators::time_dependence {

namespace {
using SphericalCompressionMap =
    domain::CoordinateMaps::TimeDependent::SphericalCompression<false>;

using ConcreteMap = domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                          SphericalCompressionMap>;

ConcreteMap create_coord_map(const std::string& f_of_t_name,
                             const double min_radius, const double max_radius,
                             const std::array<double, 3>& center) {
  return ConcreteMap{
      {SphericalCompressionMap{f_of_t_name, min_radius, max_radius, center}}};
}

void test(const std::unique_ptr<TimeDependence<3>>& time_dep_unique_ptr,
          const double initial_time, const std::string& f_of_t_name,
          const double min_radius, const double max_radius,
          const std::array<double, 3>& center) {
  MAKE_GENERATOR(gen);
  CAPTURE(initial_time);
  CAPTURE(f_of_t_name);
  CAPTURE(min_radius);
  CAPTURE(max_radius);
  CAPTURE(center);

  CHECK_FALSE(time_dep_unique_ptr->is_none());

  // We downcast to the expected derived class to make sure that factory
  // creation worked correctly. In order to maximize code reuse this check is
  // done here as opposed to separately elsewhere.
  const auto* const time_dep =
      dynamic_cast<const SphericalCompression*>(time_dep_unique_ptr.get());
  REQUIRE(time_dep != nullptr);

  // Test coordinate maps
  UniformCustomDistribution<size_t> dist_size_t{1, 10};
  const size_t num_blocks = dist_size_t(gen);
  CAPTURE(num_blocks);

  const auto expected_block_map =
      create_coord_map(f_of_t_name, min_radius, max_radius, center);

  const auto block_maps = time_dep_unique_ptr->block_maps(num_blocks);
  for (const auto& block_map_unique_ptr : block_maps) {
    const auto* const block_map =
        dynamic_cast<const ConcreteMap*>(block_map_unique_ptr.get());
    REQUIRE(block_map != nullptr);
    CHECK(*block_map == expected_block_map);
  }

  // Test functions of time
  const auto functions_of_time = time_dep_unique_ptr->functions_of_time();
  REQUIRE(functions_of_time.size() == 1);

  // Test map for composition
  CHECK(time_dep->map_for_composition() == expected_block_map);

  // For a random point at a random time check that the values agree. This is to
  // check that the internals were assigned the correct function of times.
  // The points are are drawn from the positive x/y/z quadrant in a that
  // guarantees that they are in the region where the map is valid (min_radius
  // <= radius <= max_radius), provided that min_radius is sufficiently small
  // compared to max_radius. This could be generalized, but in practice,
  // this map will not be used in cases where min_radius and max_radius
  // are very close to each other.
  TIME_DEPENDENCE_GENERATE_COORDS(make_not_null(&gen), 3, min_radius * 1.001,
                                  max_radius * 0.5);

  for (const auto& block_map : block_maps) {
    // We've checked equivalence above
    // (CHECK(*block_map == expected_block_map);), but have sometimes been
    // burned by incorrect operator== implementations so we check that the
    // mappings behave as expected.
    const double time_offset = dist(gen) + 1.2;
    CHECK_ITERABLE_APPROX(
        expected_block_map(grid_coords_dv, initial_time + time_offset,
                           functions_of_time),
        (*block_map)(grid_coords_dv, initial_time + time_offset,
                     functions_of_time));
    CHECK_ITERABLE_APPROX(
        expected_block_map(grid_coords_double, initial_time + time_offset,
                           functions_of_time),
        (*block_map)(grid_coords_double, initial_time + time_offset,
                     functions_of_time));

    CHECK_ITERABLE_APPROX(
        *expected_block_map.inverse(inertial_coords_double,
                                    initial_time + time_offset,
                                    functions_of_time),
        *block_map->inverse(inertial_coords_double, initial_time + time_offset,
                            functions_of_time));

    CHECK_ITERABLE_APPROX(
        expected_block_map.inv_jacobian(
            grid_coords_dv, initial_time + time_offset, functions_of_time),
        block_map->inv_jacobian(grid_coords_dv, initial_time + time_offset,
                                functions_of_time));
    CHECK_ITERABLE_APPROX(
        expected_block_map.inv_jacobian(
            grid_coords_double, initial_time + time_offset, functions_of_time),
        block_map->inv_jacobian(grid_coords_double, initial_time + time_offset,
                                functions_of_time));

    CHECK_ITERABLE_APPROX(
        expected_block_map.jacobian(grid_coords_dv, initial_time + time_offset,
                                    functions_of_time),
        block_map->jacobian(grid_coords_dv, initial_time + time_offset,
                            functions_of_time));
    CHECK_ITERABLE_APPROX(
        expected_block_map.jacobian(
            grid_coords_double, initial_time + time_offset, functions_of_time),
        block_map->jacobian(grid_coords_double, initial_time + time_offset,
                            functions_of_time));
  }
}

void test_equivalence() {
  SphericalCompression sc0{
      1.0, 2.5, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7, "LambdaFactorA0"};
  SphericalCompression sc1{
      1.0, 2.6, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7, "LambdaFactorA0"};
  SphericalCompression sc2{
      1.0, 2.5, 0.3, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7, "LambdaFactorA0"};
  SphericalCompression sc3{
      1.0, 2.5, 0.4, 4.1, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7, "LambdaFactorA0"};
  SphericalCompression sc4{
      1.0, 2.5, 0.4, 4.0, {{0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7, "LambdaFactorA0"};
  SphericalCompression sc5{
      1.0, 2.5, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.8, -4.6, 5.7, "LambdaFactorA0"};
  SphericalCompression sc6{
      1.0, 2.5, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, 4.6, 5.7, "LambdaFactorA0"};
  SphericalCompression sc7{
      1.0, 2.5, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.9, "LambdaFactorA0"};
  SphericalCompression sc8{
      1.0, 2.5, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7, "LambdaFactorB0"};

  CHECK(sc0 == sc0);
  CHECK_FALSE(sc0 != sc0);
  CHECK(sc0 != sc1);
  CHECK_FALSE(sc0 == sc1);
  CHECK(sc0 != sc2);
  CHECK_FALSE(sc0 == sc2);
  CHECK(sc0 != sc3);
  CHECK_FALSE(sc0 == sc3);
  CHECK(sc0 != sc4);
  CHECK_FALSE(sc0 == sc4);
  CHECK(sc0 != sc5);
  CHECK_FALSE(sc0 == sc5);
  CHECK(sc0 != sc6);
  CHECK_FALSE(sc0 == sc6);
  CHECK(sc0 != sc7);
  CHECK_FALSE(sc0 == sc7);
  CHECK(sc0 != sc8);
  CHECK_FALSE(sc0 == sc8);
}

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependence.SphericalCompression",
                  "[Domain][Unit]") {
  constexpr double initial_time{1.3};
  constexpr double update_delta_t{11.0};
  constexpr double min_radius{0.4};
  constexpr double max_radius{4.0};
  const std::array<double, 3> center{{-0.02, 0.013, 0.024}};
  constexpr double initial_value{1.0};
  constexpr double initial_velocity{-0.1};
  constexpr double initial_acceleration{0.01};
  const std::string f_of_t_name{"LambdaFactorA0"};

  const std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dep = std::make_unique<SphericalCompression>(
          initial_time, update_delta_t, min_radius, max_radius, center,
          initial_value, initial_velocity, initial_acceleration, f_of_t_name);
  test(time_dep, initial_time, f_of_t_name, min_radius, max_radius, center);
  test(time_dep->get_clone(), initial_time, f_of_t_name, min_radius, max_radius,
       center);

  test(TestHelpers::test_creation<std::unique_ptr<TimeDependence<3>>>(
           "SphericalCompression:\n"
           "  InitialTime: 1.3\n"
           "  InitialExpirationDeltaT: 11.0\n"
           "  MinRadius: 0.4\n"
           "  MaxRadius: 4.0\n"
           "  Center: [-0.02, 0.013, 0.024]\n"
           "  InitialValue: 1.0\n"
           "  InitialVelocity: -0.1\n"
           "  InitialAcceleration: 0.01\n"
           "  FunctionOfTimeName: LambdaFactorA0\n"),
       initial_time, f_of_t_name, min_radius, max_radius, center);

  test_equivalence();
}

// [[OutputRegex, Tried to create a SphericalCompression TimeDependence]]
SPECTRE_TEST_CASE(
    "Unit.Domain.Creators.TimeDependence.SphericalCompression.ErrorTest",
    "[Domain][Unit]") {
  ERROR_TEST();
  TestHelpers::test_creation<std::unique_ptr<TimeDependence<3>>>(
      "SphericalCompression:\n"
      "  InitialTime: 1.3\n"
      "  InitialExpirationDeltaT: Auto\n"
      "  MinRadius: 4.0\n"
      "  MaxRadius: 0.4\n"
      "  Center: [-0.01, 0.02, 0.01]\n"
      "  InitialValue: 3.5\n"
      "  InitialVelocity: -4.6\n"
      "  InitialAcceleration: 5.7\n"
      "  FunctionOfTimeName: LambdaFactorA0\n");
}
}  // namespace

}  // namespace domain::creators::time_dependence
