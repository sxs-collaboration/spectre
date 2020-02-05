// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "Evolution/Systems/Cce/ScriPlusInterpolationManager.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Utilities/VectorAlgebra.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

namespace Cce {
namespace {

template <typename VectorType>
void test_interpolate_quadratic() {
  MAKE_GENERATOR(generator);
  UniformCustomDistribution<double> value_dist{-5.0, 5.0};
  const double linear_coefficient = value_dist(generator);
  const double quadratic_coefficient = value_dist(generator);
  const size_t vector_size = 5;
  const size_t data_points = 40;

  const VectorType random_vector = make_with_random_values<VectorType>(
      make_not_null(&generator), make_not_null(&value_dist), vector_size);

  UniformCustomDistribution<double> time_dist{-0.5, 0.5};

  ScriPlusInterpolationManager<VectorType, ::Tags::TempScalar<0, VectorType>>
      interpolation_manager{
          4, vector_size,
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(7u, 9u)};

  VectorType comparison_lhs;
  VectorType comparison_rhs;

  Approx interpolation_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e4)
          .scale(1.0);

  // Construct data at a peculiar time set
  //  f(u_bondi) = f_0 *(1.0 + a * u_bondi + b * u_bondi^2);
  //  where u_bondi is recorded with a small random deviation away from the
  //  actual time
  for (size_t i = 0; i < data_points; ++i) {
    // this will give random times that are nonetheless guaranteed to be
    // monotonically increasing
    const DataVector time_vector = make_with_random_values<DataVector>(
        make_not_null(&generator), make_not_null(&time_dist), vector_size);
    interpolation_manager.insert_data(
        time_vector + i,
        random_vector *
            (1.0 + linear_coefficient * (static_cast<double>(i) + time_vector) +
             quadratic_coefficient *
                 square(static_cast<double>(i) + time_vector)));

    // only demand accuracy when the interpolation is reasonable
    if (i > 3 and i < data_points - 5) {
      interpolation_manager.insert_target_time(static_cast<double>(i) +
                                               time_dist(generator));
    }
    while (interpolation_manager.first_time_is_ready_to_interpolate()) {
      const auto interpolation_result =
          interpolation_manager.interpolate_and_pop_first_time();
      comparison_lhs = interpolation_result.second;
      comparison_rhs =
          random_vector *
          (1.0 + linear_coefficient * interpolation_result.first +
           quadratic_coefficient * square(interpolation_result.first));
      CHECK_ITERABLE_CUSTOM_APPROX(comparison_lhs, comparison_rhs,
                                   interpolation_approx);
    }
    // the culling of the data means that the interpolation manager should never
    // have too many more points than it needs to get a good interpolation.
    CHECK(interpolation_manager.number_of_data_points() < 12);
  }
  while (interpolation_manager.first_time_is_ready_to_interpolate()) {
    auto interpolation_result =
        interpolation_manager.interpolate_and_pop_first_time();
    comparison_lhs = interpolation_result.second;
    comparison_rhs =
        random_vector *
        (1.0 + linear_coefficient * interpolation_result.first +
         quadratic_coefficient * square(interpolation_result.first));
    CHECK_ITERABLE_CUSTOM_APPROX(comparison_lhs, comparison_rhs,
                                 interpolation_approx);
  }
  CHECK(interpolation_manager.number_of_target_times() == 0);

  // check the multiplication version
  ScriPlusInterpolationManager<
      VectorType, ::Tags::Multiplies<::Tags::TempScalar<0, VectorType>,
                                     ::Tags::TempScalar<1, VectorType>>>
      multiplication_interpolation_manager{
          4, vector_size,
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(7u, 9u)};

  const VectorType multiplies_random_vector =
      make_with_random_values<VectorType>(
          make_not_null(&generator), make_not_null(&value_dist), vector_size);

  for (size_t i = 0; i < data_points; ++i) {
    // this will give random times that are nonetheless guaranteed to be
    // monotonically increasing
    const DataVector time_vector = make_with_random_values<DataVector>(
        make_not_null(&generator), make_not_null(&time_dist), vector_size);
    multiplication_interpolation_manager.insert_data(
        time_vector + i,
        random_vector *
            (1.0 + linear_coefficient * (static_cast<double>(i) + time_vector) +
             quadratic_coefficient *
                 square(static_cast<double>(i) + time_vector)),
        multiplies_random_vector *
            (1.0 + linear_coefficient * (static_cast<double>(i) + time_vector) +
             quadratic_coefficient *
                 square(static_cast<double>(i) + time_vector)));

    // only demand accuracy when the interpolation is reasonable
    if (i > 3 and i < data_points - 5) {
      multiplication_interpolation_manager.insert_target_time(
          static_cast<double>(i) + time_dist(generator));
    }
    while (multiplication_interpolation_manager
               .first_time_is_ready_to_interpolate()) {
      const auto interpolation_result =
          multiplication_interpolation_manager.interpolate_and_pop_first_time();
      comparison_lhs = interpolation_result.second;
      comparison_rhs =
          random_vector * multiplies_random_vector *
          square(1.0 + linear_coefficient * interpolation_result.first +
                 quadratic_coefficient * square(interpolation_result.first));
      CHECK_ITERABLE_CUSTOM_APPROX(comparison_lhs, comparison_rhs,
                                   interpolation_approx);
    }
    // the culling of the data means that the interpolation manager should never
    // have too many more points than it needs to get a good interpolation.
    CHECK(multiplication_interpolation_manager.number_of_data_points() < 12);
  }
  while (multiplication_interpolation_manager
             .first_time_is_ready_to_interpolate()) {
    const auto interpolation_result =
        multiplication_interpolation_manager.interpolate_and_pop_first_time();
    comparison_lhs = interpolation_result.second;
    comparison_rhs =
        random_vector * multiplies_random_vector *
        square(1.0 + linear_coefficient * interpolation_result.first +
               quadratic_coefficient * square(interpolation_result.first));
    CHECK_ITERABLE_CUSTOM_APPROX(comparison_lhs, comparison_rhs,
                                 interpolation_approx);
  }
  CHECK(multiplication_interpolation_manager.number_of_target_times() == 0);

  // test the time derivative version of the interpolation manager
  ScriPlusInterpolationManager<VectorType,
                               Tags::Du<::Tags::TempScalar<0, VectorType>>>
      derivative_interpolation_manager{
          4, vector_size,
          std::make_unique<intrp::BarycentricRationalSpanInterpolator>(7u, 9u)};

  for (size_t i = 0; i < data_points; ++i) {
    // this will give random times that are nonetheless guaranteed to be
    // monotonically increasing
    const DataVector time_vector = make_with_random_values<DataVector>(
        make_not_null(&generator), make_not_null(&time_dist), vector_size);
    derivative_interpolation_manager.insert_data(
        time_vector + i,
        random_vector *
            (1.0 + linear_coefficient * (static_cast<double>(i) + time_vector) +
             quadratic_coefficient *
                 square(static_cast<double>(i) + time_vector)));

    // only demand accuracy when the interpolation is reasonable
    if (i > 3 and i < data_points - 5) {
      derivative_interpolation_manager.insert_target_time(
          static_cast<double>(i) + time_dist(generator));
    }
    while (
        derivative_interpolation_manager.first_time_is_ready_to_interpolate()) {
      const auto interpolation_result =
          derivative_interpolation_manager.interpolate_and_pop_first_time();
      comparison_lhs = interpolation_result.second;
      comparison_rhs =
          random_vector * (linear_coefficient + 2.0 * quadratic_coefficient *
                                                    interpolation_result.first);
      CHECK_ITERABLE_CUSTOM_APPROX(comparison_lhs, comparison_rhs,
                                   interpolation_approx);
    }
    // the culling of the data means that the interpolation manager should never
    // have too many more points than it needs to get a good interpolation.
    CHECK(derivative_interpolation_manager.number_of_data_points() < 12);
  }
  while (
      derivative_interpolation_manager.first_time_is_ready_to_interpolate()) {
    const auto interpolation_result =
        derivative_interpolation_manager.interpolate_and_pop_first_time();
    comparison_lhs = interpolation_result.second;
    comparison_rhs =
        random_vector * (linear_coefficient + 2.0 * quadratic_coefficient *
                                                  interpolation_result.first);
    CHECK_ITERABLE_CUSTOM_APPROX(comparison_lhs, comparison_rhs,
                                 interpolation_approx);
  }
  CHECK(derivative_interpolation_manager.number_of_target_times() == 0);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.ScriPlusInterpolationManager",
                  "[Unit][Evolution]") {
  test_interpolate_quadratic<DataVector>();
  test_interpolate_quadratic<ComplexDataVector>();
}
}  // namespace
}  // namespace Cce
