// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <memory>
#include <random>
#include <vector>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/TimeDependence/CompositionUniformTranslation.hpp"
#include "Domain/Creators/TimeDependence/OptionTags.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/Creators/TimeDependence/TestHelpers.hpp"

namespace domain::creators::time_dependence {

namespace {
using Translation = domain::CoordinateMaps::TimeDependent::Translation<1>;

template <typename T0, size_t MeshDim>
void test_impl(
    const gsl::not_null<T0> gen, const double initial_time,
    const std::unique_ptr<TimeDependence<MeshDim>>& time_dep,
    const std::unique_ptr<TimeDependence<MeshDim>>& expected_time_dep,
    const std::vector<std::string>& expected_f_of_t_names) {
  // Test coordinate maps
  UniformCustomDistribution<size_t> dist_size_t{1, 10};
  const size_t num_blocks = dist_size_t(*gen);
  CAPTURE(num_blocks);

  CHECK_FALSE(time_dep->is_none());

  const auto functions_of_time = time_dep->functions_of_time();
  REQUIRE(functions_of_time.size() == expected_f_of_t_names.size());
  for (const auto& f_of_t_name : expected_f_of_t_names) {
    CHECK(functions_of_time.count(f_of_t_name) == 1);
  }

  const auto functions_of_time_for_expected =
      expected_time_dep->functions_of_time();

  // For a random point at a random time check that the values agree. This is to
  // check that the internals were assigned the correct function of times.
  TIME_DEPENDENCE_GENERATE_COORDS(gen, MeshDim, -1.0, 1.0);

  const auto block_maps = time_dep->block_maps(num_blocks);
  const auto expected_block_maps = expected_time_dep->block_maps(num_blocks);
  REQUIRE(block_maps.size() == expected_block_maps.size());

  for (size_t i = 0; i < block_maps.size(); ++i) {
    const double time_offset = dist(*gen) + 1.2;
    CHECK_ITERABLE_APPROX(
        expected_block_maps[i]->operator()(grid_coords_dv,
                                           initial_time + time_offset,
                                           functions_of_time_for_expected),
        block_maps[i]->operator()(grid_coords_dv, initial_time + time_offset,
                                  functions_of_time));
    CHECK_ITERABLE_APPROX(
        expected_block_maps[i]->operator()(grid_coords_double,
                                           initial_time + time_offset,
                                           functions_of_time_for_expected),
        block_maps[i]->operator()(
            grid_coords_double, initial_time + time_offset, functions_of_time));

    CHECK_ITERABLE_APPROX(
        expected_block_maps[i]
            ->inverse(inertial_coords_double, initial_time + time_offset,
                      functions_of_time_for_expected)
            .value(),
        block_maps[i]
            ->inverse(inertial_coords_double, initial_time + time_offset,
                      functions_of_time)
            .value());

    CHECK_ITERABLE_APPROX(
        expected_block_maps[i]->inv_jacobian(grid_coords_dv,
                                             initial_time + time_offset,
                                             functions_of_time_for_expected),
        block_maps[i]->inv_jacobian(grid_coords_dv, initial_time + time_offset,
                                    functions_of_time));
    CHECK_ITERABLE_APPROX(
        expected_block_maps[i]->inv_jacobian(grid_coords_double,
                                             initial_time + time_offset,
                                             functions_of_time_for_expected),
        block_maps[i]->inv_jacobian(
            grid_coords_double, initial_time + time_offset, functions_of_time));

    CHECK_ITERABLE_APPROX(
        expected_block_maps[i]->jacobian(grid_coords_dv,
                                         initial_time + time_offset,
                                         functions_of_time_for_expected),
        block_maps[i]->jacobian(grid_coords_dv, initial_time + time_offset,
                                functions_of_time));
    CHECK_ITERABLE_APPROX(
        expected_block_maps[i]->jacobian(grid_coords_double,
                                         initial_time + time_offset,
                                         functions_of_time_for_expected),
        block_maps[i]->jacobian(grid_coords_double, initial_time + time_offset,
                                functions_of_time));
  }
}

template <size_t Dim, typename T>
void test_composition_uniform_translation(const gsl::not_null<T> gen,
                                          const double initial_time,
                                          const double update_delta_t) {
  INFO("Test composition of two uniform translations");
  using Composition = CompositionUniformTranslation<Dim>;
  UniformCustomDistribution<double> dist_double{-1.0, 1.0};

  const auto velocity0 = make_with_random_values<std::array<double, Dim>>(
      gen, make_not_null(&dist_double));
  const auto velocity1 = make_with_random_values<std::array<double, Dim>>(
      gen, make_not_null(&dist_double));
  const std::string f_of_t_names0 = "Translation";
  const std::string f_of_t_names1 = "Translation1";

  std::unique_ptr<TimeDependence<Dim>> time_dep0 =
      std::make_unique<UniformTranslation<Dim>>(initial_time, update_delta_t,
                                                velocity0, f_of_t_names0);
  std::unique_ptr<TimeDependence<Dim>> time_dep1 =
      std::make_unique<UniformTranslation<Dim>>(initial_time, update_delta_t,
                                                velocity1, f_of_t_names1);

  std::unique_ptr<TimeDependence<Dim>> expected_time_dep =
      std::make_unique<UniformTranslation<Dim>>(
          initial_time, update_delta_t, velocity0 + velocity1, "TranslationX");

  const std::unique_ptr<TimeDependence<Dim>> time_dep{
      std::make_unique<Composition>(std::move(time_dep0),
                                    std::move(time_dep1))};

  test_impl(gen, initial_time, time_dep, expected_time_dep,
            {f_of_t_names0, f_of_t_names1});

  test_impl(gen, initial_time, time_dep->get_clone(), expected_time_dep,
            {f_of_t_names0, f_of_t_names1});
}

template <typename T>
void test_options(const gsl::not_null<T> gen, const double initial_time,
                  const double update_delta_t) {
  INFO("Test create by options");

  const std::array<double, 1> velocity0{{0.5}};
  const std::array<double, 1> velocity1{{1.5}};
  const std::string f_of_t_names0 = "Translation";
  const std::string f_of_t_names1 = "Translation1";

  std::unique_ptr<TimeDependence<1>> expected_time_dep =
      std::make_unique<UniformTranslation<1>>(
          initial_time, update_delta_t, velocity0 + velocity1, "TranslationX");

  const auto created_with_options =
      TestHelpers::test_creation<std::unique_ptr<TimeDependence<1>>>(
          "CompositionUniformTranslation:\n"
          "  UniformTranslation:\n"
          "    UniformTranslation:\n"
          "      InitialTime: 1.3\n"
          "      InitialExpirationDeltaT: 2.5\n"
          "      Velocity: [0.5]\n"
          "      FunctionOfTimeName: Translation\n"
          "  UniformTranslation1:\n"
          "    UniformTranslation:\n"
          "      InitialTime: 1.3\n"
          "      InitialExpirationDeltaT: 2.5\n"
          "      Velocity: [1.5]\n"
          "      FunctionOfTimeName: Translation1\n");

  test_impl(gen, initial_time, created_with_options, expected_time_dep,
            {f_of_t_names0, f_of_t_names1});
}

SPECTRE_TEST_CASE(
    "Unit.Domain.Creators.TimeDependence.CompositionUniformTranslation",
    "[Domain][Unit]") {
  MAKE_GENERATOR(gen);
  const double initial_time = 1.3;
  const double update_delta_t = 2.5;
  test_composition_uniform_translation<1>(make_not_null(&gen), initial_time,
                                          update_delta_t);
  test_composition_uniform_translation<2>(make_not_null(&gen), initial_time,
                                          update_delta_t);
  test_composition_uniform_translation<3>(make_not_null(&gen), initial_time,
                                          update_delta_t);
  test_options(make_not_null(&gen), initial_time, update_delta_t);
}
}  // namespace
}  // namespace domain::creators::time_dependence
