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
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/Creators/TimeDependence/GenerateCoordinateMap.hpp"
#include "Domain/Creators/TimeDependence/ScalingAndZRotation.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/Creators/TimeDependence/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators::time_dependence {

namespace {
using Identity = domain::CoordinateMaps::Identity<1>;
using Rotation = domain::CoordinateMaps::TimeDependent::Rotation<2>;
template <size_t MeshDim>
using CubicScaleMap =
    domain::CoordinateMaps::TimeDependent::CubicScale<MeshDim>;

template <size_t MeshDim>
using GridToInertialMap = detail::generate_coordinate_map_t<
    Frame::Grid, Frame::Inertial,
    tmpl::list<CubicScaleMap<MeshDim>,
               tmpl::conditional_t<MeshDim == 2, Rotation,
                                   domain::CoordinateMaps::TimeDependent::
                                       ProductOf2Maps<Rotation, Identity>>>>;

template <size_t MeshDim>
GridToInertialMap<MeshDim> create_grid_to_inertial_map(
    const double outer_boundary,
    const std::array<std::string, 3>& f_of_t_names);

template <>
GridToInertialMap<2> create_grid_to_inertial_map<2>(
    const double outer_boundary,
    const std::array<std::string, 3>& f_of_t_names) {
  return GridToInertialMap<2>{
      CubicScaleMap<2>{outer_boundary, f_of_t_names[0], f_of_t_names[1]},
      Rotation{f_of_t_names[2]}};
}

template <>
GridToInertialMap<3> create_grid_to_inertial_map<3>(
    const double outer_boundary,
    const std::array<std::string, 3>& f_of_t_names) {
  return GridToInertialMap<3>{
      CubicScaleMap<3>{outer_boundary, f_of_t_names[0], f_of_t_names[1]},
      {Rotation{f_of_t_names[2]}, Identity{}}};
}

template <size_t MeshDim>
using GridToDistortedMap =
    detail::generate_coordinate_map_t<Frame::Grid, Frame::Distorted,
                                      tmpl::list<CubicScaleMap<MeshDim>>>;

template <size_t MeshDim>
GridToDistortedMap<MeshDim> create_grid_to_distorted_map(
    const double outer_boundary,
    const std::array<std::string, 3>& f_of_t_names) {
  return GridToDistortedMap<MeshDim>{
      CubicScaleMap<MeshDim>{outer_boundary, f_of_t_names[0], f_of_t_names[1]}};
}

template <size_t MeshDim>
using DistortedToInertialMap = detail::generate_coordinate_map_t<
    Frame::Distorted, Frame::Inertial,
    tmpl::list<tmpl::conditional_t<MeshDim == 2, Rotation,
                                   domain::CoordinateMaps::TimeDependent::
                                       ProductOf2Maps<Rotation, Identity>>>>;

template <size_t MeshDim>
DistortedToInertialMap<MeshDim> create_distorted_to_inertial_map(
    const std::array<std::string, 3>& f_of_t_names);

template <>
DistortedToInertialMap<2> create_distorted_to_inertial_map<2>(
    const std::array<std::string, 3>& f_of_t_names) {
  return DistortedToInertialMap<2>{Rotation{f_of_t_names[2]}};
}

template <>
DistortedToInertialMap<3> create_distorted_to_inertial_map<3>(
    const std::array<std::string, 3>& f_of_t_names) {
  return DistortedToInertialMap<3>{{Rotation{f_of_t_names[2]}, Identity{}}};
}

template <size_t MeshDim>
void test_impl(
    const std::unique_ptr<TimeDependence<MeshDim>>& time_dep_unique_ptr,
    const double initial_time, const double outer_boundary,
    const bool use_linear_scaling,
    const std::array<std::string, 3>& f_of_t_names) {
  MAKE_GENERATOR(gen);
  CAPTURE(initial_time);
  CAPTURE(outer_boundary);
  CAPTURE(use_linear_scaling);
  CAPTURE(f_of_t_names);

  CHECK_FALSE(time_dep_unique_ptr->is_none());

  // We downcast to the expected derived class to make sure that factory
  // creation worked correctly. In order to maximize code reuse this check is
  // done here as opposed to separately elsewhere.
  const auto* const time_dep =
      dynamic_cast<const ScalingAndZRotation<MeshDim>*>(
          time_dep_unique_ptr.get());
  REQUIRE(time_dep != nullptr);

  // Test coordinate maps
  UniformCustomDistribution<size_t> dist_size_t{1, 10};
  const size_t num_blocks = dist_size_t(gen);
  CAPTURE(num_blocks);

  const auto expected_grid_to_inertial_map =
      create_grid_to_inertial_map<MeshDim>(outer_boundary, f_of_t_names);

  const auto block_maps_grid_to_inertial =
      time_dep_unique_ptr->block_maps_grid_to_inertial(num_blocks);
  for (const auto& block_map_unique_ptr : block_maps_grid_to_inertial) {
    const auto* const block_map =
        dynamic_cast<const GridToInertialMap<MeshDim>*>(
            block_map_unique_ptr.get());
    REQUIRE(block_map != nullptr);
    CHECK(*block_map == expected_grid_to_inertial_map);
  }

  const auto expected_grid_to_distorted_map =
      create_grid_to_distorted_map<MeshDim>(outer_boundary, f_of_t_names);

  const auto block_maps_grid_to_distorted =
      time_dep_unique_ptr->block_maps_grid_to_distorted(num_blocks);
  for (const auto& block_map_unique_ptr : block_maps_grid_to_distorted) {
    const auto* const block_map =
        dynamic_cast<const GridToDistortedMap<MeshDim>*>(
            block_map_unique_ptr.get());
    REQUIRE(block_map != nullptr);
    CHECK(*block_map == expected_grid_to_distorted_map);
  }

  const auto expected_distorted_to_inertial_map =
      create_distorted_to_inertial_map<MeshDim>(f_of_t_names);

  const auto block_maps_distorted_to_inertial =
      time_dep_unique_ptr->block_maps_distorted_to_inertial(num_blocks);
  for (const auto& block_map_unique_ptr : block_maps_distorted_to_inertial) {
    const auto* const block_map =
        dynamic_cast<const DistortedToInertialMap<MeshDim>*>(
            block_map_unique_ptr.get());
    REQUIRE(block_map != nullptr);
    CHECK(*block_map == expected_distorted_to_inertial_map);
  }

  // Test functions of time without expiration times
  {
    const auto functions_of_time = time_dep_unique_ptr->functions_of_time();
    bool f_of_t_correct_number =
        use_linear_scaling ? functions_of_time.size() == f_of_t_names.size() - 1
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
        use_linear_scaling ? functions_of_time.size() == f_of_t_names.size() - 1
                           : functions_of_time.size() == f_of_t_names.size();
    REQUIRE(f_of_t_correct_number);
    for (const auto& f_of_t_name : f_of_t_names) {
      CHECK(functions_of_time.count(f_of_t_name) == 1);
      CHECK(functions_of_time.at(f_of_t_name)->time_bounds()[1] ==
            init_expr_time);
    }
  }

  const auto functions_of_time = time_dep_unique_ptr->functions_of_time();

  // For a random point at a random time check that the values agree. This is to
  // check that the internals were assigned the correct function of times.
  TIME_DEPENDENCE_GENERATE_COORDS(make_not_null(&gen), MeshDim, -1.0, 1.0);

  for (const auto& block_map : block_maps_grid_to_inertial) {
    // We've checked equivalence above
    // (CHECK(*block_map == expected_grid_to_inertial_map);), but have sometimes
    // been burned by incorrect operator== implementations so we check that the
    // mappings behave as expected.
    const double time_offset = dist(gen) + 1.2;
    CHECK_ITERABLE_APPROX(
        expected_grid_to_inertial_map(
            grid_coords_dv, initial_time + time_offset, functions_of_time),
        (*block_map)(grid_coords_dv, initial_time + time_offset,
                     functions_of_time));
    CHECK_ITERABLE_APPROX(
        expected_grid_to_inertial_map(
            grid_coords_double, initial_time + time_offset, functions_of_time),
        (*block_map)(grid_coords_double, initial_time + time_offset,
                     functions_of_time));

    CHECK_ITERABLE_APPROX(
        *expected_grid_to_inertial_map.inverse(inertial_coords_double,
                                               initial_time + time_offset,
                                               functions_of_time),
        *block_map->inverse(inertial_coords_double, initial_time + time_offset,
                            functions_of_time));

    CHECK_ITERABLE_APPROX(
        expected_grid_to_inertial_map.inv_jacobian(
            grid_coords_dv, initial_time + time_offset, functions_of_time),
        block_map->inv_jacobian(grid_coords_dv, initial_time + time_offset,
                                functions_of_time));
    CHECK_ITERABLE_APPROX(
        expected_grid_to_inertial_map.inv_jacobian(
            grid_coords_double, initial_time + time_offset, functions_of_time),
        block_map->inv_jacobian(grid_coords_double, initial_time + time_offset,
                                functions_of_time));

    CHECK_ITERABLE_APPROX(
        expected_grid_to_inertial_map.jacobian(
            grid_coords_dv, initial_time + time_offset, functions_of_time),
        block_map->jacobian(grid_coords_dv, initial_time + time_offset,
                            functions_of_time));
    CHECK_ITERABLE_APPROX(
        expected_grid_to_inertial_map.jacobian(
            grid_coords_double, initial_time + time_offset, functions_of_time),
        block_map->jacobian(grid_coords_double, initial_time + time_offset,
                            functions_of_time));
  }
}

template <size_t MeshDim>
void test_equivalence() {
  ScalingAndZRotation<MeshDim> map0{
      1.0, 2.4, 2.0, false, {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  ScalingAndZRotation<MeshDim> map1{
      1.2, 2.4, 2.0, false, {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  ScalingAndZRotation<MeshDim> map2{
      1.0, 2.4, 3.0, false, {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  ScalingAndZRotation<MeshDim> map3{
      1.0, 2.4, 2.0, true, {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  ScalingAndZRotation<MeshDim> map4{
      1.0, 2.4, 2.0, false, {{3.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  ScalingAndZRotation<MeshDim> map5{
      1.0, 2.4, 2.0, false, {{1.0, 0.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  ScalingAndZRotation<MeshDim> map6{
      1.0, 2.4, 2.0, false, {{1.0, 2.0}}, {{-2.0, 3.0}}, {{0.2, 0.9}}};
  ScalingAndZRotation<MeshDim> map7{
      1.0, 2.4, 2.0, false, {{1.0, 2.0}}, {{-1.0, 7.0}}, {{0.2, 0.9}}};
  ScalingAndZRotation<MeshDim> map8{
      1.0, 2.4, 2.0, false, {{1.0, 2.0}}, {{-1.0, 3.0}}, {{3.2, 0.9}}};
  ScalingAndZRotation<MeshDim> map9{
      1.0, 2.4, 2.0, false, {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 10.9}}};
  ScalingAndZRotation<MeshDim> map10{
      1.0, 2.4, 2.0, true, {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  ScalingAndZRotation<MeshDim> map11{
      1.0, 2.5, 2.0, false, {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};

  CHECK(map0 == map0);
  CHECK_FALSE(map0 != map0);
  CHECK(map0 != map1);
  CHECK_FALSE(map0 == map1);
  CHECK(map0 != map2);
  CHECK_FALSE(map0 == map2);
  CHECK(map0 != map3);
  CHECK_FALSE(map0 == map3);
  CHECK(map0 != map4);
  CHECK_FALSE(map0 == map4);
  CHECK(map0 != map5);
  CHECK_FALSE(map0 == map5);
  CHECK(map0 != map6);
  CHECK_FALSE(map0 == map6);
  CHECK(map0 != map7);
  CHECK_FALSE(map0 == map7);
  CHECK(map0 != map8);
  CHECK_FALSE(map0 == map8);
  CHECK(map0 != map9);
  CHECK_FALSE(map0 == map9);
  CHECK(map0 != map10);
  CHECK_FALSE(map0 == map10);
  CHECK(map0 != map11);
  CHECK_FALSE(map0 == map11);
}


template <size_t MeshDim>
void test(const bool use_linear_scaling) {
  const double initial_time = 1.3;
  const double angular_velocity = 2.4;
  const double outer_boundary = 10.4;
  const std::array<double, 2> initial_expansion{{1.0, 1.0}};
  const std::array<double, 2> velocity{{-0.1, 0.0}};
  const std::array<double, 2> acceleration{{-0.05, 0.0}};
  // These names must match the hard coded ones in Test_ScalingAndZRotation
  const std::string f_of_t_name0 =
      "CubicScale"s + (use_linear_scaling ? "" : "A");
  const std::string f_of_t_name1 =
      "CubicScale"s + (use_linear_scaling ? "" : "B");
  // This name must match the hard coded one in RotationAboutZAxis
  const std::string f_of_t_name2 = "Rotation";

  const std::unique_ptr<
      domain::creators::time_dependence::TimeDependence<MeshDim>>
      time_dep = std::make_unique<ScalingAndZRotation<MeshDim>>(
          initial_time, angular_velocity, outer_boundary, use_linear_scaling,
          initial_expansion, velocity, acceleration);
  test_impl(time_dep, initial_time, outer_boundary, use_linear_scaling,
            {f_of_t_name0, f_of_t_name1, f_of_t_name2});
  test_impl(time_dep->get_clone(), initial_time, outer_boundary,
            use_linear_scaling, {f_of_t_name0, f_of_t_name1, f_of_t_name2});
  const std::string linear_scaling = use_linear_scaling
                                         ? "  UseLinearScaling: true\n"
                                         : "  UseLinearScaling: false\n";
  test_impl(
      TestHelpers::test_creation<std::unique_ptr<TimeDependence<MeshDim>>>(
          "ScalingAndZRotation:\n"
          "  InitialTime: 1.3\n"
          "  AngularVelocity: 2.4\n"
          "  OuterBoundary: 10.4\n" +
          linear_scaling +
          "  InitialExpansion: [1.0, 1.0]\n"
          "  Velocity: [-0.1, 0.0]\n"
          "  Acceleration: [-0.05, 0.0]\n"),
      initial_time, outer_boundary, use_linear_scaling,
      {f_of_t_name0, f_of_t_name1, f_of_t_name2});

  test_equivalence<MeshDim>();
}

SPECTRE_TEST_CASE(
    "Unit.Domain.Creators.TimeDependence.ScalingAndZRotation",
    "[Domain][Unit]") {
  test<2>(true);
  test<3>(true);
  test<2>(false);
  test<3>(false);
}
}  // namespace
}  // namespace domain::creators::time_dependence
