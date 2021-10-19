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
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/Creators/TimeDependence/CubicScale.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/Creators/TimeDependence/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators::time_dependence {
namespace {
template <size_t MeshDim>
using CubicScaleMap =
    domain::CoordinateMaps::TimeDependent::CubicScale<MeshDim>;

template <size_t MeshDim>
using CoordMap =
    domain::CoordinateMap<Frame::Grid, Frame::Inertial, CubicScaleMap<MeshDim>>;

template <size_t MeshDim>
CoordMap<MeshDim> create_coord_map(
    const double outer_boundary,
    const std::array<std::string, 2>& f_of_t_names) {
  return CoordMap<MeshDim>{
      CubicScaleMap<MeshDim>{outer_boundary, f_of_t_names[0], f_of_t_names[1]}};
}

template <size_t MeshDim>
void test_impl(
    const std::unique_ptr<TimeDependence<MeshDim>>& time_dep_unique_ptr,
    const double initial_time, const double outer_boundary,
    const bool use_linear_scaling,
    const std::array<double, 2>& initial_expansion,
    const std::array<double, 2>& velocity,
    const std::array<double, 2>& acceleration,
    const std::array<std::string, 2>& f_of_t_names) {
  MAKE_GENERATOR(gen);
  CAPTURE(initial_time);
  CAPTURE(outer_boundary);
  CAPTURE(use_linear_scaling);
  CAPTURE(initial_expansion);
  CAPTURE(velocity);
  CAPTURE(acceleration);
  CAPTURE(f_of_t_names);

  CHECK_FALSE(time_dep_unique_ptr->is_none());

  // We downcast to the expected derived class to make sure that factory
  // creation worked correctly. In order to maximize code reuse this check is
  // done here as opposed to separately elsewhere.
  const auto* const time_dep =
      dynamic_cast<const CubicScale<MeshDim>*>(time_dep_unique_ptr.get());
  REQUIRE(time_dep != nullptr);

  // Test coordinate maps
  UniformCustomDistribution<size_t> dist_size_t{1, 10};
  const size_t num_blocks = dist_size_t(gen);
  CAPTURE(num_blocks);

  const auto expected_block_map =
      create_coord_map<MeshDim>(outer_boundary, f_of_t_names);

  const auto block_maps = time_dep_unique_ptr->block_maps(num_blocks);
  for (const auto& block_map_unique_ptr : block_maps) {
    const auto* const block_map =
        dynamic_cast<const CoordMap<MeshDim>*>(block_map_unique_ptr.get());
    REQUIRE(block_map != nullptr);
    CHECK(*block_map == expected_block_map);
  }

  // Test functions of time without expiration times
  {
    const auto functions_of_time = time_dep_unique_ptr->functions_of_time();
    bool f_of_t_correct_number =
        use_linear_scaling ? functions_of_time.size() == 1
                           : functions_of_time.size() == f_of_t_names.size();
    REQUIRE(f_of_t_correct_number);
    for (const auto& f_of_t_name : f_of_t_names) {
      CHECK(functions_of_time.count(f_of_t_name) == 1);
      CHECK(functions_of_time.at(f_of_t_name)->time_bounds()[1] ==
            std::numeric_limits<double>::infinity());
    }
  }
  // Test functions of time with expiration times
  {
    const double init_expr_time = 5.0;
    std::unordered_map<std::string, double> init_expr_times{};
    for (auto& name : f_of_t_names) {
      init_expr_times[name] = init_expr_time;
    }
    const auto functions_of_time =
        time_dep_unique_ptr->functions_of_time(init_expr_times);
    bool f_of_t_correct_number =
        use_linear_scaling ? functions_of_time.size() == 1
                           : functions_of_time.size() == f_of_t_names.size();
    REQUIRE(f_of_t_correct_number);
    for (const auto& f_of_t_name : f_of_t_names) {
      CHECK(functions_of_time.count(f_of_t_name) == 1);
      CHECK(functions_of_time.at(f_of_t_name)->time_bounds()[1] ==
            init_expr_time);
    }
  }

  const auto functions_of_time = time_dep_unique_ptr->functions_of_time();

  // Test map for composition
  CHECK(time_dep->map_for_composition() == expected_block_map);

  // For a random point at a random time check that the values agree. This is to
  // check that the internals were assigned the correct function of times.
  TIME_DEPENDENCE_GENERATE_COORDS(make_not_null(&gen), MeshDim, -1.0, 1.0);

  for (const auto& block_map : block_maps) {
    // We've checked equivalence above
    // (CHECK(*black_map == expected_block_map);), but have sometimes been
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

template <size_t MeshDim>
void test(const bool use_linear_scaling) {
  const double initial_time = 1.3;
  const double outer_boundary = 10.4;
  const std::array<double, 2> initial_expansion{{1.0, 1.0}};
  const std::array<double, 2> velocity{{-0.1, 0.0}};
  const std::array<double, 2> acceleration{{-0.05, 0.0}};
  // These names must match the hard coded ones in CubicScale
  const std::string f_of_t_name0 =
      "CubicScale"s + (use_linear_scaling ? "" : "A");
  const std::string f_of_t_name1 =
      "CubicScale"s + (use_linear_scaling ? "" : "B");

  const std::unique_ptr<
      domain::creators::time_dependence::TimeDependence<MeshDim>>
      time_dep = std::make_unique<CubicScale<MeshDim>>(
          initial_time, outer_boundary, use_linear_scaling, initial_expansion,
          velocity, acceleration);
  test_impl(time_dep, initial_time, outer_boundary, use_linear_scaling,
            initial_expansion, velocity, acceleration,
            {f_of_t_name0, f_of_t_name1});
  test_impl(time_dep->get_clone(), initial_time, outer_boundary,
            use_linear_scaling, initial_expansion, velocity, acceleration,
            {f_of_t_name0, f_of_t_name1});

  const std::string linear_scaling = use_linear_scaling
                                         ? "  UseLinearScaling: true\n"
                                         : "  UseLinearScaling: false\n";
  test_impl(
      TestHelpers::test_creation<std::unique_ptr<TimeDependence<MeshDim>>>(
          "CubicScale:\n"
          "  InitialTime: 1.3\n"
          "  OuterBoundary: 10.4\n" +
          linear_scaling +
          "  InitialExpansion: [1.0, 1.0]\n"
          "  Velocity: [-0.1, 0.0]\n"
          "  Acceleration: [-0.05, 0.0]\n"),
      initial_time, outer_boundary, use_linear_scaling, initial_expansion,
      velocity, acceleration, {f_of_t_name0, f_of_t_name1});

  INFO("Check equivalence operators");
  CubicScale<MeshDim> cubic_scale0{1.0,          2.0,           false,
                                   {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale1{1.2,          2.0,           false,
                                   {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale2{1.0,          3.0,           false,
                                   {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale3{1.0,          2.0,           true,
                                   {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale4{1.0,          2.0,           false,
                                   {{3.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale5{1.0,          2.0,           false,
                                   {{1.0, 0.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale6{1.0,          2.0,           false,
                                   {{1.0, 2.0}}, {{-2.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale7{1.0,          2.0,           false,
                                   {{1.0, 2.0}}, {{-1.0, 7.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale8{1.0,          2.0,           false,
                                   {{1.0, 2.0}}, {{-1.0, 3.0}}, {{3.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale9{1.0,          2.0,           false,
                                   {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 10.9}}};
  CubicScale<MeshDim> cubic_scale10{1.0,          2.0,           true,
                                    {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};

  CHECK(cubic_scale0 == cubic_scale0);
  CHECK_FALSE(cubic_scale0 != cubic_scale0);
  CHECK(cubic_scale0 != cubic_scale1);
  CHECK_FALSE(cubic_scale0 == cubic_scale1);
  CHECK(cubic_scale0 != cubic_scale2);
  CHECK_FALSE(cubic_scale0 == cubic_scale2);
  CHECK(cubic_scale0 != cubic_scale3);
  CHECK_FALSE(cubic_scale0 == cubic_scale3);
  CHECK(cubic_scale0 != cubic_scale4);
  CHECK_FALSE(cubic_scale0 == cubic_scale4);
  CHECK(cubic_scale0 != cubic_scale5);
  CHECK_FALSE(cubic_scale0 == cubic_scale5);
  CHECK(cubic_scale0 != cubic_scale6);
  CHECK_FALSE(cubic_scale0 == cubic_scale6);
  CHECK(cubic_scale0 != cubic_scale7);
  CHECK_FALSE(cubic_scale0 == cubic_scale7);
  CHECK(cubic_scale0 != cubic_scale8);
  CHECK_FALSE(cubic_scale0 == cubic_scale8);
  CHECK(cubic_scale0 != cubic_scale9);
  CHECK_FALSE(cubic_scale0 == cubic_scale9);
  CHECK(cubic_scale0 != cubic_scale10);
  CHECK_FALSE(cubic_scale0 == cubic_scale10);
}

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependence.CubicScale",
                  "[Domain][Unit]") {
  test<1>(true);
  test<2>(true);
  test<3>(true);
  test<1>(false);
  test<2>(false);
  test<3>(false);
}
}  // namespace
}  // namespace domain::creators::time_dependence
