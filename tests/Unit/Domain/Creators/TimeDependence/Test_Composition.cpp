// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/TimeDependence/Composition.hpp"
#include "Domain/Creators/TimeDependence/Composition.tpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/Creators/TimeDependence/TestHelpers.hpp"

namespace domain {
namespace creators {
namespace time_dependence {

namespace {
using Translation = domain::CoordinateMaps::TimeDependent::Translation;

template <typename T0, size_t MeshDim>
void test_impl(
    const gsl::not_null<T0> gen, const double initial_time,
    const std::unique_ptr<
        domain::creators::time_dependence::TimeDependence<MeshDim>>& time_dep,
    const std::unique_ptr<
        domain::creators::time_dependence::TimeDependence<MeshDim>>&
        expected_time_dep,
    const std::vector<std::string>& expected_f_of_t_names) noexcept {
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

template <typename T>
void test_composition_1d(const gsl::not_null<T> gen,
                         const double initial_time) noexcept {
  using Composition1d =
      Composition<TimeDependenceCompositionTag<UniformTranslation<1>>,
                  TimeDependenceCompositionTag<UniformTranslation<1>, 1>>;

  const std::array<double, 1> velocity0{{2.4}};
  const std::array<double, 1> velocity1{{-1.4}};
  const std::array<std::string, 1> f_of_t_names0{{"TranslationInX0"}};
  const std::array<std::string, 1> f_of_t_names1{{"TranslationInX1"}};

  std::unique_ptr<domain::creators::time_dependence::TimeDependence<1>>
      time_dep0 = std::make_unique<UniformTranslation<1>>(
          initial_time, velocity0, f_of_t_names0);
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<1>>
      time_dep1 = std::make_unique<UniformTranslation<1>>(
          initial_time, velocity1, f_of_t_names1);

  const std::unique_ptr<domain::creators::time_dependence::TimeDependence<1>>
      expected_time_dep = std::make_unique<UniformTranslation<1>>(
          initial_time, velocity0 + velocity1,
          std::array<std::string, 1>{{"TranslationX"}});

  const std::unique_ptr<domain::creators::time_dependence::TimeDependence<1>>
      time_dep = std::make_unique<Composition1d>(std::move(time_dep0),
                                                 std::move(time_dep1));

  test_impl(gen, initial_time, time_dep, expected_time_dep,
            {f_of_t_names0[0], f_of_t_names1[0]});

  test_impl(gen, initial_time, time_dep->get_clone(), expected_time_dep,
            {f_of_t_names0[0], f_of_t_names1[0]});
}

template <typename T>
void test_composition_2d(const gsl::not_null<T> gen,
                         const double initial_time) noexcept {
  using Composition2d =
      Composition<TimeDependenceCompositionTag<UniformTranslation<2>>,
                  TimeDependenceCompositionTag<UniformTranslation<2>, 1>>;

  const std::array<double, 2> velocity0{{2.4, -7.3}};
  const std::array<double, 2> velocity1{{-1.4, 3.2}};
  const std::array<std::string, 2> f_of_t_names0{
      {"TranslationInX0", "TranslationInY0"}};
  const std::array<std::string, 2> f_of_t_names1{
      {"TranslationInX1", "TranslationInY1"}};

  std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
      time_dep0 = std::make_unique<UniformTranslation<2>>(
          initial_time, velocity0, f_of_t_names0);
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
      time_dep1 = std::make_unique<UniformTranslation<2>>(
          initial_time, velocity1, f_of_t_names1);

  const std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
      expected_time_dep = std::make_unique<UniformTranslation<2>>(
          initial_time, velocity0 + velocity1,
          std::array<std::string, 2>{{"TranslationX", "TranslationY"}});

  const std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
      time_dep = std::make_unique<Composition2d>(std::move(time_dep0),
                                                 std::move(time_dep1));

  test_impl(
      gen, initial_time, time_dep, expected_time_dep,
      {f_of_t_names0[0], f_of_t_names1[0], f_of_t_names0[1], f_of_t_names1[1]});

  test_impl(
      gen, initial_time, time_dep->get_clone(), expected_time_dep,
      {f_of_t_names0[0], f_of_t_names1[0], f_of_t_names0[1], f_of_t_names1[1]});
}

template <typename T>
void test_composition_3d(const gsl::not_null<T> gen,
                         const double initial_time) noexcept {
  using Composition3d =
      Composition<TimeDependenceCompositionTag<UniformTranslation<3>>,
                  TimeDependenceCompositionTag<UniformTranslation<3>, 1>>;

  const std::array<double, 3> velocity0{{2.4, -7.3, 5.7}};
  const std::array<double, 3> velocity1{{-1.4, 3.2, -1.9}};
  const std::array<std::string, 3> f_of_t_names0{
      {"TranslationInX0", "TranslationInY0", "TranslationInZ0"}};
  const std::array<std::string, 3> f_of_t_names1{
      {"TranslationInX1", "TranslationInY1", "TranslationInZ1"}};

  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dep0 = std::make_unique<UniformTranslation<3>>(
          initial_time, velocity0, f_of_t_names0);
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dep1 = std::make_unique<UniformTranslation<3>>(
          initial_time, velocity1, f_of_t_names1);

  const std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      expected_time_dep = std::make_unique<UniformTranslation<3>>(
          initial_time, velocity0 + velocity1,
          std::array<std::string, 3>{
              {"TranslationX", "TranslationY", "TranslationZ"}});

  const std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dep = std::make_unique<Composition3d>(std::move(time_dep0),
                                                 std::move(time_dep1));

  test_impl(gen, initial_time, time_dep, expected_time_dep,
            {f_of_t_names0[0], f_of_t_names1[0], f_of_t_names0[1],
             f_of_t_names1[1], f_of_t_names0[2], f_of_t_names1[2]});

  test_impl(gen, initial_time, time_dep->get_clone(), expected_time_dep,
            {f_of_t_names0[0], f_of_t_names1[0], f_of_t_names0[1],
             f_of_t_names1[1], f_of_t_names0[2], f_of_t_names1[2]});
}

SPECTRE_TEST_CASE("Unit.Domain.Creators.TimeDependence.Composition",
                  "[Domain][Unit]") {
  MAKE_GENERATOR(gen);
  const double initial_time = 1.3;
  test_composition_1d(make_not_null(&gen), initial_time);
  test_composition_2d(make_not_null(&gen), initial_time);
  test_composition_3d(make_not_null(&gen), initial_time);
}
}  // namespace
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
