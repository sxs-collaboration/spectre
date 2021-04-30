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
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Creators/TimeDependence/UniformRotationAboutZAxis.hpp"
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
using ConcreteMap = tmpl::conditional_t<
    MeshDim == 2, domain::CoordinateMap<Frame::Grid, Frame::Inertial, Rotation>,
    domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                          domain::CoordinateMaps::TimeDependent::ProductOf2Maps<
                              Rotation, Identity>>>;

template <size_t MeshDim>
ConcreteMap<MeshDim> create_coord_map(const std::string& f_of_t_name);

template <>
ConcreteMap<2> create_coord_map<2>(const std::string& f_of_t_name) {
  return ConcreteMap<2>{{Rotation{f_of_t_name}}};
}

template <>
ConcreteMap<3> create_coord_map<3>(const std::string& f_of_t_name) {
  return ConcreteMap<3>{{Rotation{f_of_t_name}, Identity{}}};
}

template <size_t MeshDim>
void test(const std::unique_ptr<TimeDependence<MeshDim>>& time_dep_unique_ptr,
          const double initial_time, const std::string& f_of_t_name) noexcept {
  MAKE_GENERATOR(gen);
  CAPTURE(initial_time);
  CAPTURE(f_of_t_name);

  CHECK_FALSE(time_dep_unique_ptr->is_none());

  // We downcast to the expected derived class to make sure that factory
  // creation worked correctly. In order to maximize code reuse this check is
  // done here as opposed to separately elsewhere.
  const auto* const time_dep =
      dynamic_cast<const UniformRotationAboutZAxis<MeshDim>*>(
          time_dep_unique_ptr.get());
  REQUIRE(time_dep != nullptr);

  // Test coordinate maps
  UniformCustomDistribution<size_t> dist_size_t{1, 10};
  const size_t num_blocks = dist_size_t(gen);
  CAPTURE(num_blocks);

  const auto expected_block_map = create_coord_map<MeshDim>(f_of_t_name);

  const auto block_maps = time_dep_unique_ptr->block_maps(num_blocks);
  for (const auto& block_map_unique_ptr : block_maps) {
    const auto* const block_map =
        dynamic_cast<const ConcreteMap<MeshDim>*>(block_map_unique_ptr.get());
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
  TIME_DEPENDENCE_GENERATE_COORDS(make_not_null(&gen), MeshDim, -1.0, 1.0);

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

void test_equivalence() noexcept {
  {
    UniformRotationAboutZAxis<2> ur0{1.0, 2.5, 2.0, "RotationAnglePhi"};
    UniformRotationAboutZAxis<2> ur1{1.2, 2.5, 2.0, "RotationAnglePhi"};
    UniformRotationAboutZAxis<2> ur2{1.0, 2.5, 3.0, "RotationAnglePhi"};
    UniformRotationAboutZAxis<2> ur3{1.0, 2.5, 2.0, "RotationAngleTheta"};
    UniformRotationAboutZAxis<2> ur4{1.0, 2.6, 2.0, "RotationAnglePhi"};
    CHECK(ur0 == ur0);
    CHECK_FALSE(ur0 != ur0);
    CHECK(ur0 != ur1);
    CHECK_FALSE(ur0 == ur1);
    CHECK(ur0 != ur2);
    CHECK_FALSE(ur0 == ur2);
    CHECK(ur0 != ur3);
    CHECK_FALSE(ur0 == ur3);
    CHECK(ur0 != ur4);
    CHECK_FALSE(ur0 == ur4);
  }
  {
    UniformRotationAboutZAxis<3> ur0{1.0, 2.5, 2.0, "RotationAnglePhi"};
    UniformRotationAboutZAxis<3> ur1{1.2, 2.5, 2.0, "RotationAnglePhi"};
    UniformRotationAboutZAxis<3> ur2{1.0, 2.5, 3.0, "RotationAnglePhi"};
    UniformRotationAboutZAxis<3> ur3{1.0, 2.5, 2.0, "RotationAngleTheta"};
    UniformRotationAboutZAxis<3> ur4{1.0, 2.6, 2.0, "RotationAnglePhi"};
    CHECK(ur0 == ur0);
    CHECK_FALSE(ur0 != ur0);
    CHECK(ur0 != ur1);
    CHECK_FALSE(ur0 == ur1);
    CHECK(ur0 != ur2);
    CHECK_FALSE(ur0 == ur2);
    CHECK(ur0 != ur3);
    CHECK_FALSE(ur0 == ur3);
    CHECK(ur0 != ur4);
    CHECK_FALSE(ur0 == ur4);
  }
}

SPECTRE_TEST_CASE(
    "Unit.Domain.Creators.TimeDependence.UniformRotationAboutZAxis",
    "[Domain][Unit]") {
  const double initial_time = 1.3;
  const double update_delta_t = 2.5;
  constexpr double angular_velocity = 2.4;
  const std::string f_of_t_name{"RotationAngle"};
  {
    // 2d
    const std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
        time_dep = std::make_unique<UniformRotationAboutZAxis<2>>(
            initial_time, update_delta_t, angular_velocity, f_of_t_name);
    test(time_dep, initial_time, f_of_t_name);
    test(time_dep->get_clone(), initial_time, f_of_t_name);

    test(TestHelpers::test_creation<std::unique_ptr<TimeDependence<2>>>(
             "UniformRotationAboutZAxis:\n"
             "  InitialTime: 1.3\n"
             "  InitialExpirationDeltaT: 2.5\n"
             "  AngularVelocity: 2.4\n"
             "  FunctionOfTimeName: RotationAngle\n"),
         initial_time, f_of_t_name);
  }

  {
    // 3d
    const std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dep = std::make_unique<UniformRotationAboutZAxis<3>>(
            initial_time, update_delta_t, angular_velocity, f_of_t_name);
    test(time_dep, initial_time, f_of_t_name);
    test(time_dep->get_clone(), initial_time, f_of_t_name);

    test(TestHelpers::test_creation<std::unique_ptr<TimeDependence<3>>>(
             "UniformRotationAboutZAxis:\n"
             "  InitialTime: 1.3\n"
             "  InitialExpirationDeltaT: Auto\n"
             "  AngularVelocity: 2.4\n"
             "  FunctionOfTimeName: RotationAngle\n"),
         initial_time, f_of_t_name);
  }

  test_equivalence();
}
}  // namespace

}  // namespace domain::creators::time_dependence
