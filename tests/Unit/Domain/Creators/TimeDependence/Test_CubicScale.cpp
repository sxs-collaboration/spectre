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

namespace domain {
namespace creators {
namespace time_dependence {
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
    const std::array<double, 2>& initial_expansion,
    const std::array<double, 2>& velocity,
    const std::array<double, 2>& acceleration,
    const std::array<std::string, 2>& f_of_t_names) noexcept {
  MAKE_GENERATOR(gen);
  CAPTURE(initial_time);
  CAPTURE(outer_boundary);
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

  // Test functions of time
  const auto functions_of_time = time_dep_unique_ptr->functions_of_time();
  REQUIRE(functions_of_time.size() == f_of_t_names.size());
  for (const auto& f_of_t_name : f_of_t_names) {
    CHECK(functions_of_time.count(f_of_t_name) == 1);
  }

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
void test() noexcept {
  const double initial_time = 1.3;
  const double outer_boundary = 10.4;
  const std::array<double, 2> initial_expansion{{1.0, 1.0}};
  const std::array<double, 2> velocity{{-0.1, 0.0}};
  const std::array<double, 2> acceleration{{-0.05, 0.0}};
  const std::array<std::string, 2> f_of_t_names{{"Expansion0", "Expansion1"}};

  const std::unique_ptr<
      domain::creators::time_dependence::TimeDependence<MeshDim>>
      time_dep = std::make_unique<CubicScale<MeshDim>>(
          initial_time, outer_boundary, f_of_t_names, initial_expansion,
          velocity, acceleration);
  test_impl(time_dep, initial_time, outer_boundary, initial_expansion, velocity,
            acceleration, f_of_t_names);
  test_impl(time_dep->get_clone(), initial_time, outer_boundary,
            initial_expansion, velocity, acceleration, f_of_t_names);

  test_impl(TestHelpers::test_factory_creation<TimeDependence<MeshDim>>(
                "CubicScale:\n"
                "  InitialTime: 1.3\n"
                "  OuterBoundary: 10.4\n"
                "  InitialExpansion: [1.0, 1.0]\n"
                "  Velocity: [-0.1, 0.0]\n"
                "  Acceleration: [0.0, 0.0]\n"
                "  FunctionOfTimeNames: [Expansion0, Expansion1]\n"),
            initial_time, outer_boundary, initial_expansion, velocity,
            acceleration, f_of_t_names);

  test_impl(TestHelpers::test_factory_creation<TimeDependence<MeshDim>>(
                "CubicScale:\n"
                "  InitialTime: 1.3\n"
                "  OuterBoundary: 10.4\n"
                "  InitialExpansion: [1.0, 1.0]\n"
                "  Velocity: [-0.1, 0.0]\n"
                "  Acceleration: [0.0, 0.0]\n"),
            initial_time, outer_boundary, initial_expansion, velocity,
            acceleration, {{"ExpansionA", "ExpansionB"}});

  INFO("Check equivalence operators");
  CubicScale<MeshDim> cubic_scale0{
      1.0,          2.0,           {{"ExpansionA", "ExpansionB"}},
      {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale1{
      1.2,          2.0,           {{"ExpansionA", "ExpansionB"}},
      {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale2{
      1.0,          3.0,           {{"ExpansionA0", "ExpansionB"}},
      {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale3{
      1.0,          2.0,           {{"ExpansionA", "ExpansionB0"}},
      {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale4{
      1.0,          2.0,           {{"ExpansionC", "ExpansionB"}},
      {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale5{
      1.0,          2.0,           {{"ExpansionA", "ExpansionC"}},
      {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale6{
      1.0,          2.0,           {{"ExpansionA", "ExpansionB"}},
      {{3.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale7{
      1.0,          2.0,           {{"ExpansionA", "ExpansionB"}},
      {{1.0, 0.0}}, {{-1.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale8{
      1.0,          2.0,           {{"ExpansionA", "ExpansionB"}},
      {{1.0, 2.0}}, {{-2.0, 3.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale9{
      1.0,          2.0,           {{"ExpansionA", "ExpansionB"}},
      {{1.0, 2.0}}, {{-1.0, 7.0}}, {{0.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale10{
      1.0,          2.0,           {{"ExpansionA", "ExpansionB"}},
      {{1.0, 2.0}}, {{-1.0, 3.0}}, {{3.2, 0.9}}};
  CubicScale<MeshDim> cubic_scale11{
      1.0,          2.0,           {{"ExpansionA", "ExpansionB"}},
      {{1.0, 2.0}}, {{-1.0, 3.0}}, {{0.2, 10.9}}};

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
  CHECK(cubic_scale0 != cubic_scale11);
  CHECK_FALSE(cubic_scale0 == cubic_scale11);
}

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependence.CubicScale",
                  "[Domain][Unit]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
