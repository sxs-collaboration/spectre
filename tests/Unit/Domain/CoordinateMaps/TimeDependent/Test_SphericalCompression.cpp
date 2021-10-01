// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <pup.h>
#include <random>
#include <sstream>
#include <string>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/TimeDependent/SphericalCompression.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace {
// Generate a point in a random direction at a given radius
auto points_in_random_direction_generator(
    const gsl::not_null<std::mt19937*> generator,
    const std::array<double, 3>& center) {
  std::uniform_real_distribution<> phi_dis(0, 2.0 * M_PI);
  std::uniform_real_distribution<> theta_dis(0, M_PI);
  const double theta{theta_dis(*generator)};
  const double phi{phi_dis(*generator)};

  // Choose a point of unit radius with the randomly selected theta, phi.
  // This test will rescale this point to place it in different regions.
  const double rho_x0{sin(theta) * cos(phi)};
  const double rho_y0{sin(theta) * sin(phi)};
  const double rho_z0{cos(theta)};

  // A helper that returns a point given a radius
  auto point = [rho_x0, rho_y0, rho_z0, center](const double& radius) {
    return std::array<double, 3>{{radius * rho_x0 + center[0],
                                  radius * rho_y0 + center[1],
                                  radius * rho_z0 + center[2]}};
  };
  return point;
}

// Generates the map, time, and a FunctionOfTime
template <bool InteriorMap>
void generate_map_time_and_f_of_time(
    const gsl::not_null<domain::CoordinateMaps::TimeDependent::
                            SphericalCompression<InteriorMap>*>
        map,
    const gsl::not_null<double*> time,
    const gsl::not_null<std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
        functions_of_time,
    const gsl::not_null<double*> min_radius,
    const gsl::not_null<double*> max_radius,
    const gsl::not_null<std::array<double, 3>*> center,
    const gsl::not_null<std::mt19937*> generator,
    const bool bad_max_radius = false, const bool missing_f_of_t = false) {
  std::string f_of_t_name{"ExpansionFactor"};
  // Set up the map
  if (missing_f_of_t) {
    f_of_t_name = "WrongFunctionOfTimeName"s;
  }
  std::uniform_real_distribution<> rad_dis{0.4, 0.7};
  std::uniform_real_distribution<> drad_dis{0.1, 0.2};
  const double rad{rad_dis(*generator)};
  *min_radius = rad - drad_dis(*generator);
  *max_radius = rad + drad_dis(*generator);
  if (bad_max_radius) {
    *max_radius = *min_radius;
  }
  std::uniform_real_distribution<> center_dis{-0.05, 0.05};
  *center = std::array<double, 3>{
      {center_dis(*generator), center_dis(*generator), center_dis(*generator)}};
  domain::CoordinateMaps::TimeDependent::SphericalCompression<InteriorMap>
      random_map{f_of_t_name, *min_radius, *max_radius, *center};
  *map = random_map;

  // Choose a random time for evaluating the FunctionOfTime
  std::uniform_real_distribution<> time_dis{-1.0, 1.0};
  *time = time_dis(*generator);

  // Create a FunctionsOfTime containing a FunctionOfTime that, when evaluated
  // at time, is invertible. Recall that the map is invertible if
  // min_radius - max_radius < lambda(t) / sqrt(4*pi) < min_radius. So
  // lambda(t) = (min_radius - eps * max_radius) * sqrt(4*pi) is guaranteed
  // invertible for 0 < eps < 1. So set the constant term in the piecewise
  // polynomial to a0 = (min_radius - eps * max_radius) * sqrt(4*pi).
  // The remaining coefficients a1, a2, a3 will be chosen as a random
  // small factor of a0, to ensure that these terms don't make the map
  // non-invertible.
  std::uniform_real_distribution<> eps_dis{0.2, 0.8};
  std::uniform_real_distribution<> higher_coef_dis{-0.01, 0.01};
  const double a0{(*min_radius - eps_dis(*generator) * *max_radius) /
                  (0.25 * M_2_SQRTPI)};
  const std::array<DataVector, 4> initial_coefficients{
      {{{a0}},
       {{higher_coef_dis(*generator) * a0}},
       {{higher_coef_dis(*generator) * a0}},
       {{higher_coef_dis(*generator) * a0}}}};
  std::uniform_real_distribution<> dt_dis{0.1, 0.5};
  const double initial_time{*time - dt_dis(*generator)};
  const double expiration_time{*time + dt_dis(*generator)};
  (*functions_of_time)[f_of_t_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time, initial_coefficients, expiration_time);
}

// Generates the map, time, and a FunctionOfTime, but hides internal details
// (min radius, max radius, and center) when not needed
template <bool InteriorMap>
void generate_map_time_and_f_of_time(
    const gsl::not_null<domain::CoordinateMaps::TimeDependent::
                            SphericalCompression<InteriorMap>*>
        map,
    const gsl::not_null<double*> time,
    const gsl::not_null<std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
        functions_of_time,
    const gsl::not_null<std::mt19937*> generator,
    const bool bad_max_radius = false, const bool missing_f_of_t = false) {
  double min_radius{std::numeric_limits<double>::signaling_NaN()};
  double max_radius{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> center{};
  generate_map_time_and_f_of_time<InteriorMap>(
      map, time, functions_of_time, make_not_null(&min_radius),
      make_not_null(&max_radius), make_not_null(&center), generator,
      bad_max_radius, missing_f_of_t);
}

bool random_bool(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_int_distribution<> bool_dis{0, 1};
  return static_cast<bool>(bool_dis(*generator));
}

template <bool InteriorMap>
void test_out_of_bounds(const gsl::not_null<std::mt19937*> generator) {
  domain::CoordinateMaps::TimeDependent::SphericalCompression<InteriorMap>
      map{};
  double time{std::numeric_limits<double>::signaling_NaN()};
  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  double min_radius{std::numeric_limits<double>::signaling_NaN()};
  double max_radius{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> center{};
  generate_map_time_and_f_of_time(
      make_not_null(&map), make_not_null(&time),
      make_not_null(&functions_of_time), make_not_null(&min_radius),
      make_not_null(&max_radius), make_not_null(&center), generator);

  // A helper that returns a point given a radius
  auto point = points_in_random_direction_generator(generator, center);

  std::uniform_real_distribution<> rad_dis(
      std::numeric_limits<double>::epsilon(),
      1.0 - std::numeric_limits<double>::epsilon());
  double radius = std::numeric_limits<double>::signaling_NaN();
  if constexpr (InteriorMap) {
    // Choose a radius that's out of bounds
    radius = min_radius + rad_dis(*generator) * max_radius +
             std::numeric_limits<double>::epsilon();
  } else {
    // Randomly choose on which side the radius is out of bounds
    if (random_bool(generator)) {
      radius = max_radius + rad_dis(*generator) * min_radius +
               std::numeric_limits<double>::epsilon();
    } else {
      radius = rad_dis(*generator) * min_radius -
               std::numeric_limits<double>::epsilon();
    }
  }
  map(point(radius), time, functions_of_time);
  std::stringstream message;
  message << "radius should have been out of bounds, but "
          << "radius == " << radius << ", min_radius == " << min_radius
          << ", max_radius == " << max_radius
          << ", InteriorMap == " << InteriorMap;
  ERROR(message.str());
}

template <bool InteriorMap>
void test_out_of_bounds_inverse(const gsl::not_null<std::mt19937*> generator) {
  domain::CoordinateMaps::TimeDependent::SphericalCompression<InteriorMap>
      map{};
  double time{std::numeric_limits<double>::signaling_NaN()};
  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  double min_radius{std::numeric_limits<double>::signaling_NaN()};
  double max_radius{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> center{};
  generate_map_time_and_f_of_time(
      make_not_null(&map), make_not_null(&time),
      make_not_null(&functions_of_time), make_not_null(&min_radius),
      make_not_null(&max_radius), make_not_null(&center), generator);

  auto point = points_in_random_direction_generator(generator, center);

  // Get lambda_y needed to determine the bounds for the target radius
  const std::string f_of_t_name{"ExpansionFactor"};
  const double lambda_y{functions_of_time.at(f_of_t_name)->func(time)[0][0] *
                        0.25 * M_2_SQRTPI};

  std::uniform_real_distribution<> rad_dis(
      std::numeric_limits<double>::epsilon(),
      1.0 - std::numeric_limits<double>::epsilon());
  double target_radius = std::numeric_limits<double>::signaling_NaN();
  if constexpr (InteriorMap) {
    // Choose a radius that's out of bounds
    target_radius = min_radius - lambda_y + rad_dis(*generator) * max_radius +
                    std::numeric_limits<double>::epsilon();
  } else {
    // Randomly choose on which side the radius is out of bounds
    if (random_bool(generator)) {
      target_radius = max_radius + rad_dis(*generator) * min_radius +
                      std::numeric_limits<double>::epsilon();
    } else {
      target_radius = (min_radius - lambda_y) * (1.0 - rad_dis(*generator)) -
                      std::numeric_limits<double>::epsilon();
    }
  }
  CHECK_FALSE(
      map.inverse(point(target_radius), time, functions_of_time).has_value());
}
}  // namespace

namespace domain {
namespace {
template <bool InteriorMap>
void test_suite(const gsl::not_null<std::mt19937*> generator) {
  INFO("Suite");

  // Create the map, select a time, and create a FunctionsOfTime
  CoordinateMaps::TimeDependent::SphericalCompression<InteriorMap> map{};
  double time{std::numeric_limits<double>::signaling_NaN()};
  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  double min_radius{std::numeric_limits<double>::signaling_NaN()};
  double max_radius{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> center{};
  generate_map_time_and_f_of_time(
      make_not_null(&map), make_not_null(&time),
      make_not_null(&functions_of_time), make_not_null(&min_radius),
      make_not_null(&max_radius), make_not_null(&center), generator);
  CAPTURE(time);
  CAPTURE(min_radius);
  CAPTURE(max_radius);

  // Check map against a suite of functions in
  // tests/Unit/Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp
  std::uniform_real_distribution<> theta_dis(0, M_PI);
  std::uniform_real_distribution<> phi_dis(0, 2.0 * M_PI);

  const double theta = theta_dis(*generator);
  CAPTURE(theta);
  const double phi = phi_dis(*generator);
  CAPTURE(phi);

  // The Jacobian test computes finite-difference derivatives using a distance
  // of 1.e-4, so keep all test points much farther away from the boundaries
  // and the origin than that. Also ensure all points are within some finite
  // maximum radius, which here is taken to be ten times the max radius.
  constexpr double distance_from_boundaries{5.e-3};
  double lower_rad = min_radius + distance_from_boundaries;
  double upper_rad = max_radius - distance_from_boundaries;
  if constexpr (InteriorMap) {
    lower_rad = distance_from_boundaries;
    upper_rad = min_radius - distance_from_boundaries;
  }
  std::uniform_real_distribution<> radius_dis(lower_rad, upper_rad);

  const double radius = radius_dis(*generator);
  CAPTURE(radius);

  // Select a random point. Make sure that the point is not too close to the
  // junctions where the Jacobian becomes discontinuous, to ensure that the
  // jacobian test passes
  const std::array<double, 3> random_point{
      {radius * sin(theta) * cos(phi) + center[0],
       radius * sin(theta) * sin(phi) + center[1],
       radius * cos(theta) + center[2]}};

  const auto test_helper = [&random_point, &time,
                            &functions_of_time](const auto& map_to_test) {
    test_serialization(map_to_test);
    CHECK_FALSE(map_to_test != map_to_test);

    test_coordinate_map_argument_types(map_to_test, random_point, time,
                                       functions_of_time);
    test_frame_velocity(map_to_test, random_point, time, functions_of_time);
    test_jacobian(map_to_test, random_point, time, functions_of_time);
    test_inv_jacobian(map_to_test, random_point, time, functions_of_time);
    test_inverse_map(map_to_test, random_point, time, functions_of_time);
  };
  test_helper(map);

  const auto check_map_equality = [&random_point, &time, &functions_of_time](
                                      const auto& map_one,
                                      const auto& map_two) {
    tnsr::I<double, 3, Frame::BlockLogical> source_point{};
    for (size_t i = 0; i < 3; ++i) {
      source_point.get(i) = gsl::at(random_point, i);
    }
    CHECK_ITERABLE_APPROX(map_one(source_point, time, functions_of_time),
                          map_two(source_point, time, functions_of_time));
    CHECK_ITERABLE_APPROX(
        map_one.jacobian(source_point, time, functions_of_time),
        map_two.jacobian(source_point, time, functions_of_time));
    CHECK_ITERABLE_APPROX(
        map_one.inv_jacobian(source_point, time, functions_of_time),
        map_two.inv_jacobian(source_point, time, functions_of_time));
  };
  const auto map2 = serialize_and_deserialize(map);
  check_map_equality(
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(map),
      domain::make_coordinate_map<Frame::BlockLogical, Frame::Grid>(map2));
  test_helper(map2);
}

template <bool InteriorMap>
void test_map(const gsl::not_null<std::mt19937*> generator) {
  INFO("Map");
  // Create a map, choose a time, and create a FunctionOfTime.
  // Set up the map
  CoordinateMaps::TimeDependent::SphericalCompression<InteriorMap> map{};
  double time{std::numeric_limits<double>::signaling_NaN()};
  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};

  double min_radius{std::numeric_limits<double>::signaling_NaN()};
  double max_radius{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> center{std::numeric_limits<double>::signaling_NaN()};
  generate_map_time_and_f_of_time(
      make_not_null(&map), make_not_null(&time),
      make_not_null(&functions_of_time), make_not_null(&min_radius),
      make_not_null(&max_radius), make_not_null(&center), generator);

  auto point = points_in_random_direction_generator(generator, center);

  // A helper that checks if two points are scaled by the same factor
  // or not
  auto check_point_scale_factors =
      [&center, &map, &time, &functions_of_time](
          const std::array<double, 3>& orig_point_1,
          const std::array<double, 3>& orig_point_2,
          const bool check_if_equal) {
        const std::array<double, 3> mapped_point_1{
            map(orig_point_1, time, functions_of_time)};
        const std::array<double, 3> mapped_point_2{
            map(orig_point_2, time, functions_of_time)};
        Approx custom_approx = Approx::custom().epsilon(1.0e-9).scale(1.0);
        auto scale_factor_component =
            [&center](const std::array<double, 3>& orig_point,
                      const std::array<double, 3>& mapped_point,
                      const size_t i) {
              return (gsl::at(orig_point, i) - gsl::at(mapped_point, i)) /
                     (gsl::at(orig_point, i) - gsl::at(center, i));
            };
        for (size_t i = 0; i < 3; ++i) {
          if (check_if_equal) {
            CHECK(scale_factor_component(orig_point_1, mapped_point_1, i) ==
                  custom_approx(
                      scale_factor_component(orig_point_2, mapped_point_2, i)));
          } else {
            CHECK(scale_factor_component(orig_point_1, mapped_point_1, i) !=
                  custom_approx(
                      scale_factor_component(orig_point_2, mapped_point_2, i)));
          }
        }
      };

  // Set up distributions for choosing radii in the three regions:
  // smaller than min_radius, larger than max_radius, and in between
  constexpr double eps{0.01};

  double lower_radius = min_radius + eps;
  double upper_radius = max_radius - eps;
  if constexpr (InteriorMap) {
    lower_radius = eps;
    upper_radius = min_radius - eps;
  }
  std::uniform_real_distribution<> radius_dis{lower_radius, upper_radius};
  if constexpr (InteriorMap) {
    // In the interior, two random points should change radius by the same
    // factor, and that factor is the same as for a point exactly at
    // the minimum radius
    check_point_scale_factors(point(radius_dis(*generator)),
                              point(radius_dis(*generator)), true);
    check_point_scale_factors(point(radius_dis(*generator)), point(min_radius),
                              true);
  } else {
    // In the middle, two points should only move radially, but by
    // a different amount
    check_point_scale_factors(point(radius_dis(*generator)),
                              point(radius_dis(*generator)), false);
  }

  // Check that a point is invertible with the supplied functions_of_time
  CHECK(map.inverse(map(point(radius_dis(*generator)), time, functions_of_time),
                    time, functions_of_time)
            .has_value());

  // Check that a point is not invertible if the function of time is
  // too big or too small
  std::uniform_real_distribution<> higher_coef_dis{-0.01, 0.01};
  const double a0{(min_radius - max_radius - 0.5) / (0.25 * M_2_SQRTPI)};
  const std::array<DataVector, 4> initial_coefficients{
      {{{a0}},
       {{higher_coef_dis(*generator) * a0}},
       {{higher_coef_dis(*generator) * a0}},
       {{higher_coef_dis(*generator) * a0}}}};
  const double b0{(min_radius + 0.5) / (0.25 * M_2_SQRTPI)};
  const std::array<DataVector, 4> initial_coefficients_b{
      {{{b0}},
       {{higher_coef_dis(*generator) * b0}},
       {{higher_coef_dis(*generator) * b0}},
       {{higher_coef_dis(*generator) * b0}}}};
  std::uniform_real_distribution<> dt_dis{0.1, 0.5};
  const double initial_time{time - dt_dis(*generator)};
  const double expiration_time{time + dt_dis(*generator)};

  const std::string f_of_t_name{"ExpansionFactor"};
  functions_of_time[f_of_t_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time, initial_coefficients, expiration_time);
  CHECK_FALSE(
      map.inverse(map(point(radius_dis(*generator)), time, functions_of_time),
                  time, functions_of_time)
          .has_value());
  functions_of_time[f_of_t_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<3>>(
          initial_time, initial_coefficients_b, expiration_time);
  CHECK_FALSE(
      map.inverse(map(point(radius_dis(*generator)), time, functions_of_time),
                  time, functions_of_time)
          .has_value());
}

template <bool InteriorMap>
void test_is_identity(const gsl::not_null<std::mt19937*> generator) {
  INFO("Is identity");
  CoordinateMaps::TimeDependent::SphericalCompression<InteriorMap> map{};
  double time{std::numeric_limits<double>::signaling_NaN()};
  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  generate_map_time_and_f_of_time(make_not_null(&map), make_not_null(&time),
                                  make_not_null(&functions_of_time), generator);
  CHECK(not map.is_identity());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.SphericalCompression",
                  "[Domain][Unit]") {
  MAKE_GENERATOR(gen);
  test_suite<true>(make_not_null(&gen));
  test_map<true>(make_not_null(&gen));
  test_is_identity<true>(make_not_null(&gen));
  test_out_of_bounds_inverse<true>(make_not_null(&gen));
  test_suite<false>(make_not_null(&gen));
  test_map<false>(make_not_null(&gen));
  test_is_identity<false>(make_not_null(&gen));
  test_out_of_bounds_inverse<false>(make_not_null(&gen));
}

// [[OutputRegex, max_radius must be greater]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.CoordinateMaps.SphericalCompression.MaxGreater",
    "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  MAKE_GENERATOR(gen);
  CoordinateMaps::TimeDependent::SphericalCompression<false> map{};
  double time{std::numeric_limits<double>::signaling_NaN()};
  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  generate_map_time_and_f_of_time(make_not_null(&map), make_not_null(&time),
                                  make_not_null(&functions_of_time),
                                  make_not_null(&gen), true, false);
  test_suite<false>(make_not_null(&gen));
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Could not find function of time]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.CoordinateMaps.SphericalCompression.MissingFofT",
    "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  MAKE_GENERATOR(gen);
  CoordinateMaps::TimeDependent::SphericalCompression<false> map{};
  double time{std::numeric_limits<double>::signaling_NaN()};
  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  double min_radius{std::numeric_limits<double>::signaling_NaN()};
  double max_radius{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> center{};
  generate_map_time_and_f_of_time(
      make_not_null(&map), make_not_null(&time),
      make_not_null(&functions_of_time), make_not_null(&min_radius),
      make_not_null(&max_radius), make_not_null(&center), make_not_null(&gen),
      false, false);
  CoordinateMaps::TimeDependent::SphericalCompression<false> bad_map{};
  double bad_time{std::numeric_limits<double>::signaling_NaN()};
  std::unordered_map<std::string,
                     std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>
      bad_functions_of_time{};
  generate_map_time_and_f_of_time(
      make_not_null(&bad_map), make_not_null(&bad_time),
      make_not_null(&bad_functions_of_time), make_not_null(&gen), false, true);
  const std::array<double, 3> point{
      {0.5 * (max_radius + min_radius) + center[0], center[1], center[2]}};
  map(point, 0.4, bad_functions_of_time);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
}  // namespace domain

// [[OutputRegex, not in expected range]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.Domain.CoordinateMaps.SphericalCompression.MapOutOfBounds",
    "[Domain][Unit]") {
  ERROR_TEST();
  MAKE_GENERATOR(gen);
  if (random_bool(make_not_null(&gen))) {
    test_out_of_bounds<true>(make_not_null(&gen));
  } else {
    test_out_of_bounds<false>(make_not_null(&gen));
  }
  ERROR("Failed to trigger ERROR in an error test");
}
