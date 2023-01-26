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
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/Creators/TimeDependence/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators::time_dependence {

namespace {
template <size_t MeshDim>
using Translation = domain::CoordinateMaps::TimeDependent::Translation<MeshDim>;

template <size_t MeshDim, typename InputFrame = Frame::Grid,
          typename OutputFrame = Frame::Inertial>
using ConcreteMapSimple =
    domain::CoordinateMap<InputFrame, OutputFrame, Translation<MeshDim>>;

template <size_t MeshDim>
using ConcreteMapCombined =
    domain::CoordinateMap<Frame::Grid, Frame::Inertial, Translation<MeshDim>,
                          Translation<MeshDim>>;

template <size_t MeshDim, typename InputFrame = Frame::Grid,
          typename OutputFrame = Frame::Inertial>
ConcreteMapSimple<MeshDim, InputFrame, OutputFrame> create_coord_map(
    const std::string& f_of_t_name) {
  return ConcreteMapSimple<MeshDim, InputFrame, OutputFrame>{
      Translation<MeshDim>{f_of_t_name}};
}

template <size_t MeshDim>
ConcreteMapCombined<MeshDim> create_coord_map_grid_to_inertial(
    const std::string& f_of_t_grid_to_distorted,
    const std::string& f_of_t_distorted_to_inertial) {
  return ConcreteMapCombined<MeshDim>{
      Translation<MeshDim>{f_of_t_grid_to_distorted},
      Translation<MeshDim>{f_of_t_distorted_to_inertial}};
}

template <size_t MeshDim, typename InputFrame, typename OutputFrame,
          typename BlockMaps, typename BlockMap, typename Gen>
void test_maps(
    const std::unique_ptr<TimeDependence<MeshDim>>& time_dep_unique_ptr,
    const double initial_time,
    const tnsr::I<DataVector, MeshDim, InputFrame>& input_coords_dv,
    const tnsr::I<double, MeshDim, InputFrame>& input_coords_double,
    const tnsr::I<double, MeshDim, OutputFrame>& output_coords_double,
    const BlockMaps& block_maps, const BlockMap& expected_block_map,
    std::uniform_real_distribution<double>& dist, Gen& gen) {
  const auto functions_of_time = time_dep_unique_ptr->functions_of_time();

  for (const auto& block_map : block_maps) {
    // We've checked equivalence above
    // (CHECK(*black_map == expected_block_map);), but have sometimes been
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

template <size_t MeshDim>
void test_common(
    const std::unique_ptr<TimeDependence<MeshDim>>& time_dep_unique_ptr) {
  CHECK_FALSE(time_dep_unique_ptr->is_none());

  // We downcast to the expected derived class to make sure that factory
  // creation worked correctly. In order to maximize code reuse this check is
  // done here as opposed to separately elsewhere.
  const auto* const time_dep = dynamic_cast<const UniformTranslation<MeshDim>*>(
      time_dep_unique_ptr.get());
  REQUIRE(time_dep != nullptr);
}

template <size_t MeshDim>
void test_f_of_time(
    const std::unique_ptr<TimeDependence<MeshDim>>& time_dep_unique_ptr,
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

template <size_t MeshDim>
void test(const std::unique_ptr<TimeDependence<MeshDim>>& time_dep_unique_ptr,
          const double initial_time, const std::string& f_of_t_name) {
  MAKE_GENERATOR(gen);
  CAPTURE(initial_time);
  CAPTURE(f_of_t_name);

  test_common(time_dep_unique_ptr);

  // Test coordinate maps
  UniformCustomDistribution<size_t> dist_size_t{1, 10};
  const size_t num_blocks = dist_size_t(gen);
  CAPTURE(num_blocks);

  const auto expected_block_map = create_coord_map<MeshDim>(f_of_t_name);

  const auto block_maps =
      time_dep_unique_ptr->block_maps_grid_to_inertial(num_blocks);
  for (const auto& block_map_unique_ptr : block_maps) {
    const auto* const block_map =
        dynamic_cast<const ConcreteMapSimple<MeshDim>*>(
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
  TIME_DEPENDENCE_GENERATE_COORDS(make_not_null(&gen), MeshDim, -1.0, 1.0);

  test_maps(time_dep_unique_ptr, initial_time, grid_coords_dv,
            grid_coords_double, inertial_coords_double, block_maps,
            expected_block_map, dist, gen);
}

template <size_t MeshDim>
void test_with_distorted_frame(
    const std::unique_ptr<TimeDependence<MeshDim>>& time_dep_unique_ptr,
    const double initial_time, const std::string& f_of_t_grid_to_distorted,
    const std::string& f_of_t_distorted_to_inertial) {
  MAKE_GENERATOR(gen);
  CAPTURE(initial_time);
  CAPTURE(f_of_t_grid_to_distorted);
  CAPTURE(f_of_t_distorted_to_inertial);

  test_common(time_dep_unique_ptr);

  // Test coordinate maps
  UniformCustomDistribution<size_t> dist_size_t{1, 10};
  const size_t num_blocks = dist_size_t(gen);
  CAPTURE(num_blocks);

  const auto expected_block_map_grid_to_inertial =
      create_coord_map_grid_to_inertial<MeshDim>(f_of_t_grid_to_distorted,
                                                 f_of_t_distorted_to_inertial);
  const auto block_maps_grid_to_inertial =
      time_dep_unique_ptr->block_maps_grid_to_inertial(num_blocks);
  for (const auto& block_map_unique_ptr : block_maps_grid_to_inertial) {
    const auto* const block_map =
        dynamic_cast<const ConcreteMapCombined<MeshDim>*>(
            block_map_unique_ptr.get());
    REQUIRE(block_map != nullptr);
    CHECK(*block_map == expected_block_map_grid_to_inertial);
  }

  const auto expected_block_map_grid_to_distorted =
      create_coord_map<MeshDim, Frame::Grid, Frame::Distorted>(
          f_of_t_grid_to_distorted);
  const auto block_maps_grid_to_distorted =
      time_dep_unique_ptr->block_maps_grid_to_distorted(num_blocks);
  for (const auto& block_map_unique_ptr : block_maps_grid_to_distorted) {
    const auto* const block_map = dynamic_cast<
        const ConcreteMapSimple<MeshDim, Frame::Grid, Frame::Distorted>*>(
        block_map_unique_ptr.get());
    REQUIRE(block_map != nullptr);
    CHECK(*block_map == expected_block_map_grid_to_distorted);
  }

  const auto expected_block_map_distorted_to_inertial =
      create_coord_map<MeshDim, Frame::Distorted, Frame::Inertial>(
          f_of_t_distorted_to_inertial);
  const auto block_maps_distorted_to_inertial =
      time_dep_unique_ptr->block_maps_distorted_to_inertial(num_blocks);
  for (const auto& block_map_unique_ptr : block_maps_distorted_to_inertial) {
    const auto* const block_map = dynamic_cast<
        const ConcreteMapSimple<MeshDim, Frame::Distorted, Frame::Inertial>*>(
        block_map_unique_ptr.get());
    REQUIRE(block_map != nullptr);
    CHECK(*block_map == expected_block_map_distorted_to_inertial);
  }

  test_f_of_time(time_dep_unique_ptr,
                 {f_of_t_grid_to_distorted, f_of_t_distorted_to_inertial});

  // For a random point at a random time check that the values agree. This is
  // to check that the internals were assigned the correct function of times.
  TIME_DEPENDENCE_GENERATE_COORDS(make_not_null(&gen), MeshDim, -1.0, 1.0);

  test_maps(time_dep_unique_ptr, initial_time, grid_coords_dv,
            grid_coords_double, inertial_coords_double,
            block_maps_grid_to_inertial, expected_block_map_grid_to_inertial,
            dist, gen);

  TIME_DEPENDENCE_GENERATE_DISTORTED_COORDS(make_not_null(&gen), dist, MeshDim);

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
    UniformTranslation<1> ut0{1.0, {{2.0}}};
    UniformTranslation<1> ut1{1.2, {{2.0}}};
    UniformTranslation<1> ut2{1.0, {{3.0}}};
    CHECK(ut0 == ut0);
    CHECK_FALSE(ut0 != ut0);
    CHECK(ut0 != ut1);
    CHECK_FALSE(ut0 == ut1);
    CHECK(ut0 != ut2);
    CHECK_FALSE(ut0 == ut2);
  }
  {
    UniformTranslation<2> ut0{1.0, {{2.0, 4.0}}};
    UniformTranslation<2> ut1{1.2, {{2.0, 4.0}}};
    UniformTranslation<2> ut2{1.0, {{3.0, 4.0}}};
    UniformTranslation<2> ut3{1.0, {{2.0, 5.0}}};
    CHECK(ut0 == ut0);
    CHECK_FALSE(ut0 != ut0);
    CHECK(ut0 != ut1);
    CHECK_FALSE(ut0 == ut1);
    CHECK(ut0 != ut2);
    CHECK_FALSE(ut0 == ut2);
    CHECK(ut0 != ut3);
    CHECK_FALSE(ut0 == ut3);
  }
  {
    UniformTranslation<3> ut0{1.0, {{2.0, 4.0, 6.0}}};
    UniformTranslation<3> ut1{1.2, {{2.0, 4.0, 6.0}}};
    UniformTranslation<3> ut2{1.0, {{3.0, 4.0, 6.0}}};
    UniformTranslation<3> ut3{1.0, {{2.0, 5.0, 6.0}}};
    UniformTranslation<3> ut4{1.0, {{2.0, 4.0, 7.0}}};
    CHECK(ut0 == ut0);
    CHECK_FALSE(ut0 != ut0);
    CHECK(ut0 != ut1);
    CHECK_FALSE(ut0 == ut1);
    CHECK(ut0 != ut2);
    CHECK_FALSE(ut0 == ut2);
    CHECK(ut0 != ut3);
    CHECK_FALSE(ut0 == ut3);
    CHECK(ut0 != ut4);
    CHECK_FALSE(ut0 == ut4);
  }

  // With distorted frame
  {
    UniformTranslation<3> ut0{1.0, {{2.0, 4.0, 6.0}}, {{1.0, 2.0, 3.0}}};
    UniformTranslation<3> ut1{1.0, {{2.0, 4.0, 6.0}}};
    UniformTranslation<3> ut2{1.0, {{3.0, 4.0, 6.0}}, {{1.0, 2.0, 3.0}}};
    UniformTranslation<3> ut3{1.0, {{2.0, 5.0, 6.0}}, {{1.0, 2.0, 3.0}}};
    UniformTranslation<3> ut4{1.0, {{2.0, 4.0, 6.0}}, {{1.0, 3.0, 2.0}}};
    CHECK(ut0 == ut0);
    CHECK_FALSE(ut0 != ut0);
    CHECK(ut0 != ut1);
    CHECK_FALSE(ut0 == ut1);
    CHECK(ut0 != ut2);
    CHECK_FALSE(ut0 == ut2);
    CHECK(ut0 != ut3);
    CHECK_FALSE(ut0 == ut3);
    CHECK(ut0 != ut4);
    CHECK_FALSE(ut0 == ut4);
  }
}

template <size_t Dim>
void test_with_distorted_frame_driver(
    const std::array<double, Dim>& velocity_grid_to_distorted,
    const std::array<double, Dim>& velocity_distorted_to_inertial) {
  const double initial_time = 1.3;

  // This name must match the hard coded one in UniformTranslation
  const std::string f_of_t_grid_to_distorted = "Translation";
  const std::string f_of_t_distorted_to_inertial =
      "TranslationDistortedToInertial";
  const std::unique_ptr<domain::creators::time_dependence::TimeDependence<Dim>>
      time_dep = std::make_unique<UniformTranslation<Dim>>(
          initial_time, velocity_grid_to_distorted,
          velocity_distorted_to_inertial);
  test_with_distorted_frame(time_dep, initial_time, f_of_t_grid_to_distorted,
                            f_of_t_distorted_to_inertial);
  test_with_distorted_frame(time_dep->get_clone(), initial_time,
                            f_of_t_grid_to_distorted,
                            f_of_t_distorted_to_inertial);
}

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependence.UniformTranslation",
                  "[Domain][Unit]") {
  const double initial_time = 1.3;

  {
    // 1d
    const std::array<double, 1> velocity{{2.4}};
    // This name must match the hard coded one in UniformTranslation
    const std::string f_of_t_name = "Translation";
    const std::unique_ptr<domain::creators::time_dependence::TimeDependence<1>>
        time_dep =
            std::make_unique<UniformTranslation<1>>(initial_time, velocity);
    test(time_dep, initial_time, f_of_t_name);
    test(time_dep->get_clone(), initial_time, f_of_t_name);

    test(TestHelpers::test_creation<std::unique_ptr<TimeDependence<1>>>(
             "UniformTranslation:\n"
             "  InitialTime: 1.3\n"
             "  Velocity: [2.4]\n"),
         initial_time, f_of_t_name);
  }

  {
    // 2d
    const std::array<double, 2> velocity{{2.4, 3.1}};
    // This name must match the hard coded one in UniformTranslation
    const std::string f_of_t_name = "Translation";
    const std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
        time_dep =
            std::make_unique<UniformTranslation<2>>(initial_time, velocity);
    test(time_dep, initial_time, f_of_t_name);
    test(time_dep->get_clone(), initial_time, f_of_t_name);

    test(TestHelpers::test_creation<std::unique_ptr<TimeDependence<2>>>(
             "UniformTranslation:\n"
             "  InitialTime: 1.3\n"
             "  Velocity: [2.4, 3.1]\n"),
         initial_time, f_of_t_name);
  }

  {
    // 3d
    const std::array<double, 3> velocity{{2.4, 3.1, -1.2}};
    // This name must match the hard coded one in UniformTranslation
    const std::string f_of_t_name = "Translation";
    const std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dep =
            std::make_unique<UniformTranslation<3>>(initial_time, velocity);
    test(time_dep, initial_time, f_of_t_name);
    test(time_dep->get_clone(), initial_time, f_of_t_name);

    test(TestHelpers::test_creation<std::unique_ptr<TimeDependence<3>>>(
             "UniformTranslation:\n"
             "  InitialTime: 1.3\n"
             "  Velocity: [2.4, 3.1, -1.2]\n"),
         initial_time, f_of_t_name);
  }

  test_with_distorted_frame_driver<1>({{2.4}}, {{3.3}});
  test_with_distorted_frame_driver<2>({{2.4, 3.1}}, {{3.3, 1.6}});
  test_with_distorted_frame_driver<3>({{2.4, 3.1, 1.3}}, {{3.3, 1.6, -1.4}});
  test_equivalence();
}
}  // namespace

}  // namespace domain::creators::time_dependence
