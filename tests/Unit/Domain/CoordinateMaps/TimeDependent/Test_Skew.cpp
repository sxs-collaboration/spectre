// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Skew.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

namespace domain {
namespace {
constexpr size_t deriv_order = 2;
using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>;
using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;

template <typename T>
std::array<T, 3> sph_to_cart(const T& radius, const T& theta, const T& phi) {
  return std::array<T, 3>{radius * sin(theta) * cos(phi),
                          radius * sin(theta) * sin(phi), radius * cos(theta)};
}

template <typename Generator>
void test(const gsl::not_null<Generator*> generator) {
  const double initial_time = 0.5;
  double t = initial_time + 0.05;
  const double dt = 0.6;
  const double expiration_time = 20.0;

  const std::string function_of_time_name{"Skew"};

  // NOLINTBEGIN
  std::uniform_real_distribution<double> fot_dist{-0.1, 0.1};
  std::uniform_real_distribution<double> outer_radius_dist{50.0, 150.0};
  std::uniform_real_distribution<double> angle_dist{0.0, 2.0 * M_PI};
  // NOLINTEND

  std::unordered_map<std::string, FoftPtr> f_of_t_list{};
  f_of_t_list[function_of_time_name] = std::make_unique<Polynomial>(
      initial_time,
      std::array<DataVector, 3>{
          make_with_random_values<DataVector>(
              generator, make_not_null(&fot_dist), DataVector{2, 0.0}),
          0.1 * make_with_random_values<DataVector>(
                    generator, make_not_null(&fot_dist), DataVector{2, 0.0}),
          DataVector{2, 0.0}},
      expiration_time);

  const double outer_radius = outer_radius_dist(*generator);
  // Subtracting 1e-3 from outer radius ensures that the jacobian test helper
  // only evaluates the map within the outer radius
  std::uniform_real_distribution<double> radius_dist{0.0, outer_radius - 1.e-3};
  const std::array<double, 3> center =
      50.0 * std::array{fot_dist(*generator), fot_dist(*generator),
                        fot_dist(*generator)};
  CAPTURE(outer_radius);
  CAPTURE(center);

  const CoordinateMaps::TimeDependent::Skew skew_map{function_of_time_name,
                                                     center, outer_radius};

  CHECK(skew_map.function_of_time_names().contains(function_of_time_name));

  // test serialized/deserialized map
  const auto skew_map_deserialized = serialize_and_deserialize(skew_map);
  CHECK(skew_map_deserialized.function_of_time_names().contains(
      function_of_time_name));

  const Approx deriv_approx = Approx::custom().epsilon(1.e-9).scale(1.0);
  const Approx inv_approx = Approx::custom().epsilon(5.e-13).scale(1.0);

  while (t < expiration_time) {
    CAPTURE(t);
    const auto func_and_derivs =
        f_of_t_list.at(function_of_time_name)->func_and_2_derivs(t);
    CAPTURE(func_and_derivs);

    const std::array<double, 3> point_xi = [&]() {
      const double radius = radius_dist(*generator);
      const double theta = 0.5 * angle_dist(*generator);
      const double phi = angle_dist(*generator);
      return sph_to_cart(radius, theta, phi);
    }();

    const std::array<DataVector, 3> dv_point_xi = [&]() {
      const DataVector for_size{10, 0.0};
      const auto radii = make_with_random_values<DataVector>(
          generator, make_not_null(&radius_dist), for_size);
      const DataVector thetas =
          0.5 * make_with_random_values<DataVector>(
                    generator, make_not_null(&angle_dist), for_size);
      const auto phis = make_with_random_values<DataVector>(
          generator, make_not_null(&angle_dist), for_size);

      return sph_to_cart(radii, thetas, phis);
    }();

    CAPTURE(dv_point_xi);

    const auto run_checks = [&](const auto& points) {
      test_jacobian(skew_map, points, t, f_of_t_list, deriv_approx);
      test_inv_jacobian(skew_map, points, t, f_of_t_list);
      test_frame_velocity(skew_map, points, t, f_of_t_list, deriv_approx);

      test_jacobian(skew_map_deserialized, points, t, f_of_t_list,
                    deriv_approx);
      test_inv_jacobian(skew_map_deserialized, points, t, f_of_t_list);
      test_frame_velocity(skew_map_deserialized, points, t, f_of_t_list,
                          deriv_approx);
    };

    run_checks(point_xi);
    test_coordinate_map_argument_types(skew_map, point_xi, t, f_of_t_list);
    test_coordinate_map_argument_types(skew_map_deserialized, point_xi, t,
                                       f_of_t_list);
    test_inverse_map(skew_map, point_xi, t, f_of_t_list, inv_approx);
    test_inverse_map(skew_map_deserialized, point_xi, t, f_of_t_list,
                     inv_approx);
    run_checks(dv_point_xi);

    t += dt;
  }

  // Check inequivalence operator
  CHECK_FALSE(skew_map != skew_map);
  CHECK_FALSE(skew_map_deserialized != skew_map_deserialized);

  // Check serialization
  CHECK(skew_map == skew_map_deserialized);
  CHECK_FALSE(skew_map != skew_map_deserialized);
}

void test_specific_points() {
  const std::string function_of_time_name{"Skew"};
  const double time = 0.0;
  std::unordered_map<std::string, FoftPtr> f_of_t_list{};
  // Use pi/4 so math is easy
  f_of_t_list[function_of_time_name] = std::make_unique<Polynomial>(
      time,
      std::array{DataVector{2, M_PI_4}, DataVector{2, 0.0}, DataVector{2, 0.0}},
      std::numeric_limits<double>::infinity());

  // Make it unit so math is easy
  const std::array<double, 3> center{0.0, -1.0 / sqrt(2.0), 1.0 / sqrt(2.0)};
  const double outer_radius = 100.0;

  const CoordinateMaps::TimeDependent::Skew skew_map{function_of_time_name,
                                                     center, outer_radius};

  std::array<double, 3> test_point{};
  {
    INFO("Center");
    test_point = center;
    CAPTURE(test_point);

    const auto mapped_point = skew_map(test_point, time, f_of_t_list);
    // Should be exact
    CHECK(mapped_point == test_point);
    const auto inverse_point =
        skew_map.inverse(mapped_point, time, f_of_t_list);
    CHECK(inverse_point.has_value());
    CHECK_ITERABLE_APPROX(inverse_point.value(), test_point);

    const auto jacobian = skew_map.jacobian(test_point, time, f_of_t_list);
    auto expected_jacobian = identity<3>(0.0);
    // Because the angles are pi/4
    get<0, 1>(expected_jacobian) =
        -0.5 * (1.0 + cos(M_PI / square(outer_radius)));
    get<0, 2>(expected_jacobian) = get<0, 1>(expected_jacobian);
    CHECK_ITERABLE_APPROX(jacobian, expected_jacobian);
  }
  {
    INFO("Half way along y-axis");
    test_point = std::array{0.0, 0.5 * outer_radius, 0.0};
    CAPTURE(test_point);

    const auto mapped_point = skew_map(test_point, time, f_of_t_list);
    const double falloff = 0.25 * (2.0 + sqrt(2.0));
    const double tan_sum = -test_point[1];
    // Should be exact
    CHECK_ITERABLE_APPROX(
        mapped_point,
        (std::array{falloff * tan_sum, test_point[1], test_point[2]}));
    const auto inverse_point =
        skew_map.inverse(mapped_point, time, f_of_t_list);
    CHECK(inverse_point.has_value());
    CHECK_ITERABLE_APPROX(inverse_point.value(), test_point);

    const auto jacobian = skew_map.jacobian(test_point, time, f_of_t_list);
    auto expected_jacobian = identity<3>(0.0);
    get<0, 2>(expected_jacobian) = -falloff;
    get<0, 1>(expected_jacobian) =
        -0.5 * M_PI * tan_sum / (sqrt(2.0) * outer_radius) +
        get<0, 2>(expected_jacobian);
    CHECK_ITERABLE_APPROX(jacobian, expected_jacobian);
  }
  {
    INFO("Outer radius");
    test_point = std::array{0.0, 0.0, outer_radius};
    CAPTURE(test_point);

    const auto mapped_point = skew_map(test_point, time, f_of_t_list);
    CHECK_ITERABLE_APPROX(mapped_point, test_point);
    const auto inverse_point =
        skew_map.inverse(mapped_point, time, f_of_t_list);
    CHECK(inverse_point.has_value());
    CHECK_ITERABLE_APPROX(inverse_point.value(), test_point);
    const auto jacobian = skew_map.jacobian(test_point, time, f_of_t_list);
    CHECK_ITERABLE_APPROX(jacobian, identity<3>(0.0));
  }
  {
    INFO("Outer radius plus eps");
    const double eps = std::numeric_limits<double>::epsilon();
    test_point = std::array{0.0, outer_radius + eps, 0.0};
    CAPTURE(test_point);

    const auto mapped_point = skew_map(test_point, time, f_of_t_list);
    CHECK_ITERABLE_APPROX(mapped_point, test_point);
    const auto inverse_point =
        skew_map.inverse(mapped_point, time, f_of_t_list);
    CHECK(inverse_point.has_value());
    CHECK_ITERABLE_APPROX(inverse_point.value(), test_point);
    const auto jacobian = skew_map.jacobian(test_point, time, f_of_t_list);
    CHECK_ITERABLE_APPROX(jacobian, identity<3>(0.0));
  }
}

void test_errors() {
#ifdef SPECTRE_DEBUG
  const std::string function_of_time_name{"Skew"};
  const double time = 0.0;
  std::unordered_map<std::string, FoftPtr> f_of_t_list{};
  // Use values close to -PI/2 to trigger the error
  f_of_t_list[function_of_time_name] = std::make_unique<Polynomial>(
      time,
      std::array{DataVector{2, -M_PI_2 + 1.e-3}, DataVector{2, 0.0},
                 DataVector{2, 0.0}},
      std::numeric_limits<double>::infinity());

  const double outer_radius = 100.0;

  const CoordinateMaps::TimeDependent::Skew skew_map{
      function_of_time_name, std::array{0.0, 0.0, 0.0}, outer_radius};

  const std::array<double, 3> point{50.0, 20.0, 30.0};

  CHECK_THROWS_WITH((skew_map(point, time, f_of_t_list)),
                    Catch::Matchers::ContainsSubstring("Skew map is singular"));
#endif
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.TimeDependent.Skew",
                  "[Domain][Unit]") {
  MAKE_GENERATOR(generator);
  test(make_not_null(&generator));
  test_specific_points();
  test_errors();
}
}  // namespace domain
