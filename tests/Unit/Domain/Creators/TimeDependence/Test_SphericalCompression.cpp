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

template <typename InputFrame = Frame::Grid,
          typename OutputFrame = Frame::Inertial>
using ConcreteMapSimple =
    domain::CoordinateMap<InputFrame, OutputFrame, SphericalCompressionMap>;

using ConcreteMapCombined =
    domain::CoordinateMap<Frame::Grid, Frame::Inertial, SphericalCompressionMap,
                          SphericalCompressionMap>;

template <typename InputFrame = Frame::Grid,
          typename OutputFrame = Frame::Inertial>
ConcreteMapSimple<InputFrame, OutputFrame> create_coord_map(
    const std::string& f_of_t_name, const double min_radius,
    const double max_radius, const std::array<double, 3>& center) {
  return ConcreteMapSimple<InputFrame, OutputFrame>{
      {SphericalCompressionMap{f_of_t_name, min_radius, max_radius, center}}};
}

ConcreteMapCombined create_coord_map_grid_to_inertial(
    const std::string& f_of_t_grid_to_distorted,
    const std::string& f_of_t_distorted_to_inertial, const double min_radius,
    const double max_radius, const std::array<double, 3>& center) {
  return ConcreteMapCombined{
      SphericalCompressionMap{f_of_t_grid_to_distorted, min_radius, max_radius,
                              center},
      SphericalCompressionMap{f_of_t_distorted_to_inertial, min_radius,
                              max_radius, center}};
}

template <typename InputFrame, typename OutputFrame, typename BlockMaps,
          typename BlockMap, typename Gen>
void test_maps(const std::unique_ptr<TimeDependence<3>>& time_dep_unique_ptr,
               const double initial_time,
               const tnsr::I<DataVector, 3, InputFrame>& input_coords_dv,
               const tnsr::I<double, 3, InputFrame>& input_coords_double,
               const tnsr::I<double, 3, OutputFrame>& output_coords_double,
               const BlockMaps& block_maps, const BlockMap& expected_block_map,
               std::uniform_real_distribution<double>& dist, Gen& gen) {
  const auto functions_of_time = time_dep_unique_ptr->functions_of_time();
  for (const auto& block_map : block_maps) {
    // We've checked equivalence above
    // (CHECK(*block_map == expected_block_map);), but have sometimes been
    // burned by incorrect operator== implementations so we check that the
    // mappings behave as expected.
    const double time_offset = dist(gen) + 1.2;
    CHECK_ITERABLE_APPROX(
        expected_block_map(input_coords_dv, initial_time + time_offset,
                           functions_of_time),
        (*block_map)(input_coords_dv, initial_time + time_offset,
                     functions_of_time));
    CHECK_ITERABLE_APPROX(
        expected_block_map(input_coords_double, initial_time + time_offset,
                           functions_of_time),
        (*block_map)(input_coords_double, initial_time + time_offset,
                     functions_of_time));

    CHECK_ITERABLE_APPROX(
        *expected_block_map.inverse(output_coords_double,
                                    initial_time + time_offset,
                                    functions_of_time),
        *block_map->inverse(output_coords_double, initial_time + time_offset,
                            functions_of_time));

    CHECK_ITERABLE_APPROX(
        expected_block_map.inv_jacobian(
            input_coords_dv, initial_time + time_offset, functions_of_time),
        block_map->inv_jacobian(input_coords_dv, initial_time + time_offset,
                                functions_of_time));
    CHECK_ITERABLE_APPROX(
        expected_block_map.inv_jacobian(
            input_coords_double, initial_time + time_offset, functions_of_time),
        block_map->inv_jacobian(input_coords_double, initial_time + time_offset,
                                functions_of_time));

    CHECK_ITERABLE_APPROX(
        expected_block_map.jacobian(input_coords_dv, initial_time + time_offset,
                                    functions_of_time),
        block_map->jacobian(input_coords_dv, initial_time + time_offset,
                            functions_of_time));
    CHECK_ITERABLE_APPROX(
        expected_block_map.jacobian(
            input_coords_double, initial_time + time_offset, functions_of_time),
        block_map->jacobian(input_coords_double, initial_time + time_offset,
                            functions_of_time));
  }
}

void test_common(
    const std::unique_ptr<TimeDependence<3>>& time_dep_unique_ptr) {
  CHECK_FALSE(time_dep_unique_ptr->is_none());

  // We downcast to the expected derived class to make sure that factory
  // creation worked correctly. In order to maximize code reuse this check is
  // done here as opposed to separately elsewhere.
  const auto* const time_dep =
      dynamic_cast<const SphericalCompression*>(time_dep_unique_ptr.get());
  REQUIRE(time_dep != nullptr);
}

void test_f_of_time(
    const std::unique_ptr<TimeDependence<3>>& time_dep_unique_ptr,
    const std::vector<std::string>& f_of_t_names) {
  // Test without expiration times
  const auto functions_of_time = time_dep_unique_ptr->functions_of_time();
  REQUIRE(functions_of_time.size() == f_of_t_names.size());
  for (const auto& f_of_t_name : f_of_t_names) {
    CHECK(functions_of_time.count(f_of_t_name) == 1);
    CHECK(functions_of_time.at(f_of_t_name)->time_bounds()[1] ==
          std::numeric_limits<double>::infinity());
  }

  // Test with expiration times
  const double init_expr_time = 5.0;
  std::unordered_map<std::string, double> init_expr_times{};
  for (const auto& f_of_t_name : f_of_t_names) {
    init_expr_times[f_of_t_name] = init_expr_time;
  }
  const auto functions_of_time_with_expr_times =
      time_dep_unique_ptr->functions_of_time(init_expr_times);
  REQUIRE(functions_of_time_with_expr_times.size() == f_of_t_names.size());
  for (const auto& f_of_t_name : f_of_t_names) {
    CHECK(functions_of_time_with_expr_times.count(f_of_t_name) == 1);
    CHECK(functions_of_time_with_expr_times.at(f_of_t_name)->time_bounds()[1] ==
          init_expr_time);
  }
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

  test_common(time_dep_unique_ptr);

  // Test coordinate maps
  UniformCustomDistribution<size_t> dist_size_t{1, 10};
  const size_t num_blocks = dist_size_t(gen);
  CAPTURE(num_blocks);

  const auto expected_block_map =
      create_coord_map(f_of_t_name, min_radius, max_radius, center);

  const auto block_maps =
      time_dep_unique_ptr->block_maps_grid_to_inertial(num_blocks);
  for (const auto& block_map_unique_ptr : block_maps) {
    const auto* const block_map =
        dynamic_cast<const ConcreteMapSimple<Frame::Grid, Frame::Inertial>*>(
            block_map_unique_ptr.get());
    REQUIRE(block_map != nullptr);
    CHECK(*block_map == expected_block_map);
  }

  // Check that maps involving distorted frame are empty.
  CHECK(time_dep_unique_ptr->block_maps_grid_to_distorted(1)[0].get() ==
        nullptr);
  CHECK(time_dep_unique_ptr->block_maps_distorted_to_inertial(1)[0].get() ==
        nullptr);

  test_f_of_time(time_dep_unique_ptr, {f_of_t_name});

  // For a random point at a random time check that the values agree. This is
  // to check that the internals were assigned the correct function of times.
  // The points are are drawn from the positive x/y/z quadrant in a that
  // guarantees that they are in the region where the map is valid (min_radius
  // <= radius <= max_radius), provided that min_radius is sufficiently small
  // compared to max_radius. This could be generalized, but in practice,
  // this map will not be used in cases where min_radius and max_radius
  // are very close to each other.
  TIME_DEPENDENCE_GENERATE_COORDS(make_not_null(&gen), 3, min_radius * 1.001,
                                  max_radius * 0.5);

  test_maps(time_dep_unique_ptr, initial_time, grid_coords_dv,
            grid_coords_double, inertial_coords_double, block_maps,
            expected_block_map, dist, gen);
}

void test_with_distorted_frame(
    const std::unique_ptr<TimeDependence<3>>& time_dep_unique_ptr,
    const double initial_time, const std::string& f_of_t_grid_to_distorted,
    const std::string& f_of_t_distorted_to_inertial, const double min_radius,
    const double max_radius, const std::array<double, 3>& center) {
  MAKE_GENERATOR(gen);
  CAPTURE(initial_time);
  CAPTURE(f_of_t_grid_to_distorted);
  CAPTURE(f_of_t_distorted_to_inertial);
  CAPTURE(min_radius);
  CAPTURE(max_radius);
  CAPTURE(center);

  test_common(time_dep_unique_ptr);

  // Test coordinate maps
  UniformCustomDistribution<size_t> dist_size_t{1, 10};
  const size_t num_blocks = dist_size_t(gen);
  CAPTURE(num_blocks);

  const auto expected_block_map_grid_to_inertial =
      create_coord_map_grid_to_inertial(f_of_t_grid_to_distorted,
                                        f_of_t_distorted_to_inertial,
                                        min_radius, max_radius, center);

  const auto block_maps_grid_to_inertial =
      time_dep_unique_ptr->block_maps_grid_to_inertial(num_blocks);
  for (const auto& block_map_unique_ptr : block_maps_grid_to_inertial) {
    const auto* const block_map =
        dynamic_cast<const ConcreteMapCombined*>(block_map_unique_ptr.get());
    REQUIRE(block_map != nullptr);
    CHECK(*block_map == expected_block_map_grid_to_inertial);
  }

  const auto expected_block_map_grid_to_distorted =
      create_coord_map<Frame::Grid, Frame::Distorted>(
          f_of_t_grid_to_distorted, min_radius, max_radius, center);
  const auto block_maps_grid_to_distorted =
      time_dep_unique_ptr->block_maps_grid_to_distorted(num_blocks);
  for (const auto& block_map_unique_ptr : block_maps_grid_to_distorted) {
    const auto* const block_map =
        dynamic_cast<const ConcreteMapSimple<Frame::Grid, Frame::Distorted>*>(
            block_map_unique_ptr.get());
    REQUIRE(block_map != nullptr);
    CHECK(*block_map == expected_block_map_grid_to_distorted);
  }

  const auto expected_block_map_distorted_to_inertial =
      create_coord_map<Frame::Distorted, Frame::Inertial>(
          f_of_t_distorted_to_inertial, min_radius, max_radius, center);
  const auto block_maps_distorted_to_inertial =
      time_dep_unique_ptr->block_maps_distorted_to_inertial(num_blocks);
  for (const auto& block_map_unique_ptr : block_maps_distorted_to_inertial) {
    const auto* const block_map = dynamic_cast<
        const ConcreteMapSimple<Frame::Distorted, Frame::Inertial>*>(
        block_map_unique_ptr.get());
    REQUIRE(block_map != nullptr);
    CHECK(*block_map == expected_block_map_distorted_to_inertial);
  }

  test_f_of_time(time_dep_unique_ptr,
                 {f_of_t_grid_to_distorted, f_of_t_distorted_to_inertial});

  // For a random point at a random time check that the values agree. This is
  // to check that the internals were assigned the correct function of times.
  // The points are are drawn from the positive x/y/z quadrant in a that
  // guarantees that they are in the region where the map is valid (min_radius
  // <= radius <= max_radius), provided that min_radius is sufficiently small
  // compared to max_radius. This could be generalized, but in practice,
  // this map will not be used in cases where min_radius and max_radius
  // are very close to each other.
  TIME_DEPENDENCE_GENERATE_COORDS(make_not_null(&gen), 3, min_radius * 1.001,
                                  max_radius * 0.5);

  test_maps(time_dep_unique_ptr, initial_time, grid_coords_dv,
            grid_coords_double, inertial_coords_double,
            block_maps_grid_to_inertial, expected_block_map_grid_to_inertial,
            dist, gen);

  TIME_DEPENDENCE_GENERATE_DISTORTED_COORDS(make_not_null(&gen), dist, 3);

  test_maps(time_dep_unique_ptr, initial_time, grid_coords_dv,
            grid_coords_double, distorted_coords_double,
            block_maps_grid_to_distorted, expected_block_map_grid_to_distorted,
            dist, gen);

  test_maps(time_dep_unique_ptr, initial_time, distorted_coords_dv,
            distorted_coords_double, inertial_coords_double,
            block_maps_distorted_to_inertial,
            expected_block_map_distorted_to_inertial, dist, gen);
}

void test_equivalence() {
  {
    SphericalCompression sc0{1.0, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7};
    SphericalCompression sc1{1.0, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7};
    SphericalCompression sc2{1.0, 0.3, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7};
    SphericalCompression sc3{1.0, 0.4, 4.1, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7};
    SphericalCompression sc4{1.0, 0.4, 4.0, {{0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7};
    SphericalCompression sc5{1.0, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.8, -4.6, 5.7};
    SphericalCompression sc6{1.0, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, 4.6, 5.7};
    SphericalCompression sc7{1.0, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.9};

    CHECK(sc0 == sc0);
    CHECK_FALSE(sc0 != sc0);
    CHECK(sc0 == sc1);
    CHECK_FALSE(sc0 != sc1);
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
  }
  // With distorted frame
  {
    SphericalCompression sc0{1.0, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7,
                             0.2, 3.4, 3.6};
    SphericalCompression sc1{1.0, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7,
                             0.2, 3.4, 3.6};
    SphericalCompression sc2{1.0, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7};
    SphericalCompression sc3{1.0,  0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7,
                             0.21, 3.4, 3.6};
    SphericalCompression sc4{1.0, 0.4,  4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7,
                             0.2, -3.4, 3.6};
    SphericalCompression sc5{1.0, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.7,
                             0.2, 3.4, 3.7};
    SphericalCompression sc6{1.0, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.25, -4.6, 5.7,
                             0.2, 3.4, 3.6};
    SphericalCompression sc7{1.0, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.5, 5.7,
                             0.2, 3.4, 3.6};
    SphericalCompression sc8{1.0, 0.4, 4.0, {{-0.2, 1.3, 2.4}}, 3.5, -4.6, 5.5,
                             0.2, 3.4, 3.6};

    CHECK(sc0 == sc0);
    CHECK_FALSE(sc0 != sc0);
    CHECK(sc0 == sc1);
    CHECK_FALSE(sc0 != sc1);
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
}

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependence.SphericalCompression",
                  "[Domain][Unit]") {
  constexpr double initial_time{1.3};
  constexpr double min_radius{0.4};
  constexpr double max_radius{4.0};
  const std::array<double, 3> center{{-0.02, 0.013, 0.024}};
  constexpr double initial_value{1.0};
  constexpr double initial_velocity{-0.1};
  constexpr double initial_acceleration{0.01};
  // This name must match the hard coded one in SphericalCompression
  const std::string f_of_t_name = "Size";

  CHECK_THROWS_WITH(
      SphericalCompression(initial_time, max_radius, min_radius, center,
                           initial_value, initial_velocity,
                           initial_acceleration),
      Catch::Matchers::Contains(
          "Tried to create a SphericalCompression TimeDependence"));

  const std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dep = std::make_unique<SphericalCompression>(
          initial_time, min_radius, max_radius, center, initial_value,
          initial_velocity, initial_acceleration);
  test(time_dep, initial_time, f_of_t_name, min_radius, max_radius, center);
  test(time_dep->get_clone(), initial_time, f_of_t_name, min_radius, max_radius,
       center);

  test(TestHelpers::test_creation<std::unique_ptr<TimeDependence<3>>>(
           "SphericalCompression:\n"
           "  InitialTime: 1.3\n"
           "  MinRadius: 0.4\n"
           "  MaxRadius: 4.0\n"
           "  Center: [-0.02, 0.013, 0.024]\n"
           "  InitialValue: 1.0\n"
           "  InitialVelocity: -0.1\n"
           "  InitialAcceleration: 0.01\n"),
       initial_time, f_of_t_name, min_radius, max_radius, center);

  test_equivalence();

  // Now distorted-frame tests.

  CHECK_THROWS_WITH(
      SphericalCompression(initial_time, max_radius, min_radius, center,
                           initial_value, initial_velocity,
                           initial_acceleration, initial_value,
                           initial_velocity, initial_acceleration),
      Catch::Matchers::Contains(
          "Tried to create a SphericalCompression TimeDependence"));

  // These names must match the hard coded one in SphericalCompression
  const std::string f_of_t_grid_to_distorted = "Size";
  const std::string f_of_t_distorted_to_inertial = "SizeDistortedToInertial";
  const std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dep_distorted = std::make_unique<SphericalCompression>(
          initial_time, min_radius, max_radius, center, initial_value,
          initial_velocity, initial_acceleration, initial_value,
          initial_velocity, initial_acceleration);
  test_with_distorted_frame(
      time_dep_distorted, initial_time, f_of_t_grid_to_distorted,
      f_of_t_distorted_to_inertial, min_radius, max_radius, center);
  test_with_distorted_frame(
      time_dep_distorted->get_clone(), initial_time, f_of_t_grid_to_distorted,
      f_of_t_distorted_to_inertial, min_radius, max_radius, center);
}
}  // namespace

}  // namespace domain::creators::time_dependence
