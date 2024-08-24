// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/RegisterDerivedWithCharm.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/Wedge.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {
namespace {
// These don't really need to change so we make them global
constexpr double inner_radius = 0.5;
constexpr double outer_radius = 10.0;

std::array<double, 3> sph_to_cart(const double radius, const double theta,
                                  const double phi) {
  return std::array{radius * sin(theta) * cos(phi),
                    radius * sin(theta) * sin(phi), radius * cos(theta)};
}

Wedge make_wedge(
    const double inner_sphericity, const double outer_sphericity,
    const std::array<double, 3>& inner_center = std::array{0.0, 0.0, 0.0},
    const std::array<double, 3>& outer_center = std::array{0.0, 0.0, 0.0},
    const Wedge::Axis axis = Wedge::Axis::PlusZ) {
  const Wedge wedge{inner_center, inner_radius, inner_sphericity,
                    outer_center, outer_radius, outer_sphericity,
                    axis};
  return serialize_and_deserialize(wedge);
}

// Since this is hard to test for general points without just reimplementing the
// function, we test for specific points where it's easier to calculate
void test_gradient_no_offset() {
  INFO("Test gradient no offset");

  const auto compute_gradient =
      [&](const std::array<double, 3>& point, const double inner_distance,
          const double outer_distance, const double inner_sphericity,
          const double outer_sphericity) -> std::array<double, 3> {
    const double radius = magnitude(point);
    const double distance_difference = outer_distance - inner_distance;
    const std::array<double, 3> grad_base =
        1.0 / (sqrt(3.0) * radius) *
        std::array{point[0], point[1],
                   -(square(radius) - square(point[2])) / point[2]} /
        point[2];
    const std::array<double, 3> inner_grad =
        (1.0 - inner_sphericity) * inner_radius * grad_base;
    const std::array<double, 3> outer_grad =
        (1.0 - outer_sphericity) * outer_radius * grad_base;
    return (outer_grad - point / radius) / distance_difference -
           (outer_distance - radius) * (outer_grad - inner_grad) /
               square(distance_difference);
  };

  const auto test_z_axis = [&](const double inner_sphericity,
                               const double outer_sphericity) {
    INFO("Test z-axis");
    const Wedge wedge = make_wedge(inner_sphericity, outer_sphericity);
    double inner_distance = inner_radius;
    double outer_distance = outer_radius;
    if (inner_sphericity == 0.0) {
      inner_distance /= sqrt(3.0);
    }
    if (outer_sphericity == 0.0) {
      outer_distance /= sqrt(3.0);
    }

    for (const double radius : {inner_distance, outer_distance,
                                0.5 * (inner_distance + outer_distance)}) {
      CAPTURE(radius);
      const std::array point{0.0, 0.0, radius};
      CAPTURE(point);

      CHECK_ITERABLE_APPROX(
          wedge.gradient(point),
          (compute_gradient(point, inner_distance, outer_distance,
                            inner_sphericity, outer_sphericity)));
    }
  };

  const auto test_corners = [&](const double inner_sphericity,
                                const double outer_sphericity) {
    INFO("Test corners");
    const Wedge wedge = make_wedge(inner_sphericity, outer_sphericity);
    // Corners will always be at sphere radius
    for (const double radius :
         {inner_radius, outer_radius, 0.5 * (inner_radius + outer_radius)}) {
      for (const double phi :
           {M_PI_4, 3.0 * M_PI_4, 5.0 * M_PI_4, 7.0 * M_PI_4}) {
        CAPTURE(radius);
        CAPTURE(phi * 180.0 / M_PI);
        const std::array point =
            sph_to_cart(radius, acos(1.0 / sqrt(3.0)), phi);
        CAPTURE(point);

        CHECK_ITERABLE_APPROX(
            wedge.gradient(point),
            (compute_gradient(point, inner_radius, outer_radius,
                              inner_sphericity, outer_sphericity)));
      }
    }
  };

  {
    INFO("Both boundaries spherical");
    test_z_axis(1.0, 1.0);
    test_corners(1.0, 1.0);
  }
  {
    INFO("Inner boundary spherical, outer boundary flat");
    test_z_axis(1.0, 0.0);
    test_corners(1.0, 0.0);
  }
  {
    INFO("Inner boundary flat, outer boundary spherical");
    test_z_axis(0.0, 1.0);
    test_corners(0.0, 1.0);
  }
  {
    INFO("Both boundaries flat");
    test_z_axis(0.0, 0.0);
    test_corners(0.0, 0.0);
  }
}

void test_only_transition_no_offset() {
  INFO("Test only transition no offset");
  const double inner_sphericity = 1.0;
  const double outer_sphericity = 0.0;
  const std::array<double, 3> inner_center{0.0, 0.0, 0.0};
  const std::array<double, 3> outer_center{0.0, 0.0, 0.0};

  const Wedge wedge =
      make_wedge(inner_sphericity, outer_sphericity, inner_center, outer_center,
                 Wedge::Axis::PlusX);

  const double inner_distance = 0.5;
  const double outer_distance = outer_radius / sqrt(3.0);
  const double distance_difference = outer_distance - inner_distance;

  std::array<double, 3> point{4.0, 0.0, 0.0};
  const double function_value = (outer_distance - 4.0) / distance_difference;

  CHECK(wedge(point) == approx(function_value));
  // Parallel::printf("Test gradient\n");
  CHECK_ITERABLE_APPROX(wedge.gradient(point),
                        (std::array{-1.0 / distance_difference, 0.0, 0.0}));

  std::optional<double> orig_rad_over_rad{};
  const auto set_orig_rad_over_rad =
      [&](const std::array<double, 3>& mapped_point,
          const double distorted_radii) {
        orig_rad_over_rad =
            wedge.original_radius_over_radius(mapped_point, distorted_radii);
      };

  // Test actual values
  set_orig_rad_over_rad(point, 0.0);
  CHECK(orig_rad_over_rad.has_value());
  CHECK(orig_rad_over_rad.value() == approx(1.0));
  set_orig_rad_over_rad(point * (1.0 - 0.25 * function_value * 0.5), 0.5);
  CHECK(orig_rad_over_rad.has_value());
  CHECK(orig_rad_over_rad.value() ==
        approx(4.0 / magnitude(point * (1.0 - 0.25 * function_value * 0.5))));
  set_orig_rad_over_rad(point * (1.0 + 0.25 * function_value * 0.5), -0.5);
  CHECK(orig_rad_over_rad.has_value());
  CHECK(orig_rad_over_rad.value() ==
        approx(4.0 / magnitude(point * (1.0 + 0.25 * function_value * 0.5))));
  // Hit some internal checks
  set_orig_rad_over_rad(point * 0.0, 0.0);
  CHECK_FALSE(orig_rad_over_rad.has_value());
  set_orig_rad_over_rad(point, 1.0);
  CHECK_FALSE(orig_rad_over_rad.has_value());
  set_orig_rad_over_rad(point, 15.0);
  CHECK_FALSE(orig_rad_over_rad.has_value());
  set_orig_rad_over_rad(point * 15.0, 0.0);
  CHECK_FALSE(orig_rad_over_rad.has_value());
  // Wedge is in x direction. Check other directions
  const auto check_other_directions =
      [&](const std::array<double, 3>& mapped_point) {
        set_orig_rad_over_rad(mapped_point, 0.0);
        CHECK(orig_rad_over_rad.has_value());
        CHECK(orig_rad_over_rad.value() == 1.0);
      };
  check_other_directions(std::array{-4.0, 0.0, 0.0});
  check_other_directions(std::array{0.0, 0.0, 4.0});
  check_other_directions(std::array{4.0, 0.0, -4.0});
  check_other_directions(std::array{0.0, 4.0, 0.0});
  check_other_directions(std::array{0.0, -4.0, 0.0});
  // At overall inner boundary.
  set_orig_rad_over_rad(std::array{0.0, 0.0, 0.2 * inner_radius}, 0.4);
  CHECK(orig_rad_over_rad.has_value());
  CHECK(orig_rad_over_rad.value() == approx(5.0));
}

void test_only_transition_offset() {
  INFO("Test only transition offset");
  const double inner_sphericity = 1.0;
  const double outer_sphericity = 0.0;
  const std::array<double, 3> inner_center{-1.0, -1.0, 0.0};
  const std::array<double, 3> outer_center{0.0, 0.0, 0.0};

  const Wedge wedge =
      make_wedge(inner_sphericity, outer_sphericity, inner_center, outer_center,
                 Wedge::Axis::PlusX);

  // All these values were calculated by hand given the centers above and this
  // specific point below
  const double half_cube_length = outer_radius / sqrt(3.0);
  const std::array<double, 3> point{4.0, 0.0, 0.0};
  const std::array<double, 3> centered_point = point - inner_center;
  const double radius_from_inner_center = magnitude(centered_point);

  const double inner_distance = 0.5;
  const double outer_distance = (half_cube_length + 1) * sqrt(26.0) / 5.0;
  const double distance_difference = outer_distance - inner_distance;

  const double function_value =
      (outer_distance - radius_from_inner_center) / distance_difference;

  CHECK(wedge(centered_point) == approx(function_value));
  const std::array<double, 3> projection_center = inner_center - outer_center;
  // Because of PlusX
  const std::array<double, 3> outer_gradient =
      centered_point / radius_from_inner_center *
          (half_cube_length - projection_center[0]) / centered_point[0] -
      std::array{
          (half_cube_length - projection_center[0]) / square(centered_point[0]),
          0.0, 0.0} *
          radius_from_inner_center;
  const std::array<double, 3> expected_gradient =
      (outer_gradient - centered_point / radius_from_inner_center) /
          distance_difference -
      ((outer_distance - radius_from_inner_center) * outer_gradient /
       square(distance_difference));
  CHECK_ITERABLE_APPROX(wedge.gradient(centered_point), expected_gradient);

  std::optional<double> orig_rad_over_rad{};
  const auto set_orig_rad_over_rad =
      [&](const std::array<double, 3>& mapped_point,
          const double distorted_radii) {
        orig_rad_over_rad =
            wedge.original_radius_over_radius(mapped_point, distorted_radii);
      };

  // Test actual values
  set_orig_rad_over_rad(centered_point, 0.0);
  CHECK(orig_rad_over_rad.has_value());
  CHECK(orig_rad_over_rad.value() == approx(1.0));

  std::array<double, 3> mapped_point =
      centered_point * (1.0 - function_value / radius_from_inner_center * 0.5);
  set_orig_rad_over_rad(mapped_point, 0.5);
  CHECK(orig_rad_over_rad.has_value());
  CHECK(orig_rad_over_rad.value() ==
        approx(radius_from_inner_center / magnitude(mapped_point)));

  mapped_point =
      centered_point * (1.0 + function_value / radius_from_inner_center * 0.5);
  set_orig_rad_over_rad(mapped_point, -0.5);
  CHECK(orig_rad_over_rad.has_value());
  CHECK(orig_rad_over_rad.value() ==
        approx(radius_from_inner_center / magnitude(mapped_point)));

  // Hit some internal checks
  set_orig_rad_over_rad(centered_point * 0.0, 0.0);
  CHECK_FALSE(orig_rad_over_rad.has_value());
  set_orig_rad_over_rad(centered_point, 1.0);
  CHECK_FALSE(orig_rad_over_rad.has_value());
  set_orig_rad_over_rad(centered_point, 15.0);
  CHECK_FALSE(orig_rad_over_rad.has_value());
  set_orig_rad_over_rad(centered_point * 15.0, 0.0);
  CHECK_FALSE(orig_rad_over_rad.has_value());

  // Wedge is in x direction. Check other directions
  const auto check_other_directions =
      [&](const std::array<double, 3>& new_mapped_point) {
        set_orig_rad_over_rad(new_mapped_point, 0.0);
        CHECK(orig_rad_over_rad.has_value());
        CHECK(orig_rad_over_rad.value() == 1.0);
      };
  check_other_directions(std::array{-4.0, 0.0, 0.0});
  check_other_directions(std::array{0.0, 0.0, 4.0});
  check_other_directions(std::array{4.0, 0.0, -4.0});
  check_other_directions(std::array{0.0, 4.0, 0.0});
  check_other_directions(std::array{0.0, -4.0, 0.0});

  // At overall inner boundary.
  set_orig_rad_over_rad(std::array{0.0, 0.0, 0.2 * inner_radius}, 0.4);
  CHECK(orig_rad_over_rad.has_value());
  CHECK(orig_rad_over_rad.value() == approx(5.0));
}

// This also doesn't need to change
constexpr size_t l_max = 4;

// We add an option to check the jacobians because if a point is on a boundary,
// then the numerical derivative will fail because the operator() isn't defined
// outside the boundary
template <typename T>
void test_points_shape_map(const double inner_sphericity,
                           const double outer_sphericity,
                           const std::array<double, 3>& inner_center,
                           const std::array<double, 3>& outer_center,
                           const Wedge::Axis axis, const double check_time,
                           const std::string& fot_name,
                           const FunctionsOfTimeMap& functions_of_time,
                           const std::array<T, 3>& points,
                           const bool check_jacobians = true) {
  std::unique_ptr<ShapeMapTransitionFunction> wedge = std::make_unique<Wedge>(
      inner_center, inner_radius, inner_sphericity, outer_center, outer_radius,
      outer_sphericity, static_cast<Wedge::Axis>(axis));

  const TimeDependent::Shape shape{inner_center, l_max, l_max, std::move(wedge),
                                   fot_name};

  // test_coordinate_map_argument_types creates its own DataVectors and we only
  // allow the inverse to be called with doubles
  if constexpr (std::is_same_v<T, double>) {
    test_coordinate_map_argument_types(shape, points, check_time,
                                       functions_of_time);
    test_inverse_map(shape, points, check_time, functions_of_time);
  }

  test_frame_velocity(shape, points, check_time, functions_of_time);

  if (check_jacobians) {
    test_jacobian(shape, points, check_time, functions_of_time);
    test_inv_jacobian(shape, points, check_time, functions_of_time);
  }
}

template <typename Generator>
void test_in_shape_map_no_offset(const gsl::not_null<Generator*> generator) {
  INFO("Test using shape map without offset");
  std::uniform_real_distribution<double> coef_dist{-0.01, 0.01};

  const double initial_time = 1.0;
  const double check_time = 2.0;
  const size_t num_coefs = 2 * square(l_max + 1);
  const std::array center{0.1, -0.2, 0.3};
  const std::string fot_name{"TheBean"};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  auto coefs = make_with_random_values<DataVector>(
      generator, make_not_null(&coef_dist), DataVector{num_coefs});
  functions_of_time[fot_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          initial_time, std::array{std::move(coefs)},
          std::numeric_limits<double>::infinity());

  std::uniform_real_distribution<double> angle_dist{0.0, 2.0 * M_PI};
  // This guarantees the radius of the point is within the wedge and that
  // numerical jacobians will succeed
  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<double> radial_dist{
      inner_radius + 1.e-3, outer_radius / sqrt(3.0) - 1.e-3};

  for (const auto& [inner_sphericity, outer_sphericity] :
       cartesian_product(std::array{1.0, 0.0}, std::array{1.0, 0.0})) {
    CAPTURE(inner_sphericity);
    CAPTURE(outer_sphericity);

    // Test 10 points for each sphericity and orientation
    for (size_t i = 0; i < 10; i++) {
      const double radius = radial_dist(*generator);
      const double theta = 0.5 * angle_dist(*generator);
      const double phi = angle_dist(*generator);

      CAPTURE(radius);
      CAPTURE(theta);
      CAPTURE(phi);

      const std::array<double, 3> centered_point =
          sph_to_cart(radius, theta, phi);

      int axis = 0;
      double max = abs(centered_point[0]);
      for (int j = 1; j < 3; j++) {
        if (const double maybe_max = abs(gsl::at(centered_point, j));
            maybe_max > max) {
          axis = j;
          max = maybe_max;
        }
      }

      // Because Wedge::Axis is defined slightly differently
      const int axis_for_wedge =
          (axis + 1) * (gsl::at(centered_point, axis) > 0.0 ? 1 : -1);

      CAPTURE(centered_point);
      CAPTURE(axis_for_wedge);

      test_points_shape_map(inner_sphericity, outer_sphericity, center, center,
                            static_cast<Wedge::Axis>(axis_for_wedge),
                            check_time, fot_name, functions_of_time,
                            centered_point + center);
    }
  }
}

template <typename Generator>
void test_in_shape_map_offset(const gsl::not_null<Generator*> generator) {
  INFO("Test using shape map with offset");
  std::uniform_real_distribution<double> coef_dist{-0.01, 0.01};
  // This guarantees the inner center will be inside the outer surface
  std::uniform_real_distribution<double> center_dist{-2.0, 2.0};
  const double half_cube_length = outer_radius / sqrt(3.0);

  const double initial_time = 1.0;
  const double check_time = 2.0;
  const size_t num_coefs = 2 * square(l_max + 1);
  const std::string fot_name{"TheBean2.0"};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  auto coefs = make_with_random_values<DataVector>(
      generator, make_not_null(&coef_dist), DataVector{num_coefs});
  functions_of_time[fot_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          initial_time, std::array{std::move(coefs)},
          std::numeric_limits<double>::infinity());

  // Worst case scenario is that inner and outer center are exactly opposite of
  // each other, (2,2,2) and (-2,-2,-2) for example. Given that the half cube
  // length L = 10/sqrt(3) = 5.77, we can take the smallest angle of any wedge
  // and use that for all wedges to be safe.
  // NOLINTNEXTLINE(misc-const-correctness)
  std::uniform_real_distribution<double> theta_dist{
      0.0, atan2(half_cube_length - 4.0, half_cube_length + 4.0)};
  std::uniform_real_distribution<double> phi_dist{0.0, 2.0 * M_PI};  // NOLINT
  std::uniform_int_distribution<int> direction_dist{0, 2};           // NOLINT
  // This is a map between the integer representation of axes from the
  // Wedge::Axis class to the index of the `orientation_maps` array
  const std::unordered_map<int, int> axes{{3, 0},  {-3, 1}, {2, 2},
                                          {-2, 3}, {1, 4},  {-1, 5}};
  const auto orientation_maps = orientations_for_sphere_wrappings();

  for (const double outer_sphericity :
       std::array{1.0, 0.0, phi_dist(*generator) / (2.0 * M_PI)}) {
    CAPTURE(outer_sphericity);

    const auto inner_center = make_with_random_values<std::array<double, 3>>(
        generator, make_not_null(&center_dist));
    const auto outer_center = make_with_random_values<std::array<double, 3>>(
        generator, make_not_null(&center_dist));

    CAPTURE(inner_center);
    CAPTURE(outer_center);

    const auto make_point = [&](const std::optional<int>& axis_opt,
                                const std::optional<int>& axis_sign_opt)
        -> std::pair<int, std::array<double, 3>> {
      const int axis = axis_opt.value_or(direction_dist(*generator));
      const int axis_sign =
          axis_sign_opt.value_or(coef_dist(*generator) > 0.0 ? 1 : -1);
      const int axis_for_wedge = axis_sign * (axis + 1);
      CAPTURE(axis_for_wedge);
      // Compute the distance directly along the axis in the direction of this
      // wedge to the cube surface. This is the smallest outer distance that
      // will always work
      const double minimal_outer_distance =
          static_cast<double>(axis_sign) *
              gsl::at(outer_center - inner_center, axis) +
          half_cube_length;

      // This guarantees the radius of the point is within the wedge and that
      // numerical jacobians will succeed
      // NOLINTNEXTLINE(misc-const-correctness)
      std::uniform_real_distribution<double> radial_dist{
          inner_radius + 1.e-3, minimal_outer_distance - 1.e-3};

      // We construct points aligned in the +z direction with all the proper
      // angles, and then rotate it to the correct wedge.
      const double radius = radial_dist(*generator);
      const double theta = theta_dist(*generator);
      const double phi = phi_dist(*generator);
      const std::array<double, 3> unrotated_point =
          sph_to_cart(radius, theta, phi);
      const DiscreteRotation<3> rotation{
          gsl::at(orientation_maps, axes.at(axis_for_wedge))};

      return {axis_for_wedge, rotation(unrotated_point) + inner_center};
    };

    // Test 10 random double points for each sphericity
    for (size_t i = 0; i < 10; i++) {
      INFO("Doubles");
      const auto [axis_for_wedge, point] =
          make_point(std::nullopt, std::nullopt);

      CAPTURE(axis_for_wedge);
      CAPTURE(point);

      test_points_shape_map(1.0, outer_sphericity, inner_center, outer_center,
                            static_cast<Wedge::Axis>(axis_for_wedge),
                            check_time, fot_name, functions_of_time, point);
    }

    // Now do a DataVector
    {
      INFO("DataVector");
      std::array<DataVector, 3> points = make_array<3>(DataVector{10_st});
      const int axis = direction_dist(*generator);
      const int axis_sign = coef_dist(*generator) > 0.0 ? 1 : -1;
      const int axis_for_wedge = axis_sign * (axis + 1);
      for (size_t i = 0; i < points[0].size(); i++) {
        const auto axis_and_point = make_point(axis, axis_sign);
        for (size_t j = 0; j < 3; j++) {
          gsl::at(points, j)[i] = gsl::at(axis_and_point.second, j);
        }
      }

      CAPTURE(axis_sign);
      CAPTURE(points);

      test_points_shape_map(1.0, outer_sphericity, inner_center, outer_center,
                            static_cast<Wedge::Axis>(axis_for_wedge),
                            check_time, fot_name, functions_of_time, points);
    }

    // Test all eight corners of the outer surface, but for each axis as well so
    // we get all the wedges.
    INFO("Testing corners with offset");
    for (const auto& [axis_for_wedge, unused] : axes) {
      (void)unused;
      const int axis = abs(axis_for_wedge) - 1;
      const double axis_sign = axis_for_wedge > 0 ? 1.0 : -1.0;

      CAPTURE(axis_for_wedge);

      for (const double other_length_1 :
           {-half_cube_length, half_cube_length}) {
        for (const double other_length_2 :
             {-half_cube_length, half_cube_length}) {
          std::array<double, 3> point = outer_center;
          gsl::at(point, axis) += axis_sign * half_cube_length;
          gsl::at(point, (axis + 1) % 3) += other_length_1;
          gsl::at(point, (axis + 2) % 3) += other_length_2;
          CAPTURE(point);

          // Don't test jacobians at corners because of numerical derivatives
          test_points_shape_map(
              1.0, outer_sphericity, inner_center, outer_center,
              static_cast<Wedge::Axis>(axis_for_wedge), check_time, fot_name,
              functions_of_time, point, false);
        }
      }
    }
  }
}

void test_errors() {
  CHECK_THROWS_WITH(
      (Wedge{std::array{0.0, 0.0, 0.0}, inner_radius, 0.5,
             std::array{1.0, 1.0, 1.0}, outer_radius, 1.0, Wedge::Axis::PlusZ}),
      Catch::Matchers::ContainsSubstring(
          "The sphericity of the inner surface must be exactly 1.0 if the "
          "centers of the inner and outer object are different."));

  CHECK_THROWS_WITH(
      (Wedge{std::array{outer_radius * 2.0, 0.0, 0.0}, inner_radius, 1.0,
             std::array{0.0, 0.0, 0.0}, outer_radius, 1.0, Wedge::Axis::PlusZ}),
      Catch::Matchers::ContainsSubstring("The inner surface center") and
          Catch::Matchers::ContainsSubstring(
              "must be within the outer surface with center"));

  CHECK_THROWS_WITH(
      (Wedge{std::array{0.0, 0.0, 0.0}, inner_radius, 0.5,
             std::array{0.0, 0.0, 0.0}, outer_radius, 1.0, Wedge::Axis::None}),
      Catch::Matchers::ContainsSubstring("Axis for Wedge shape map transition "
                                         "function cannot be 'None'. Please "
                                         "choose another."));
}

template <typename Generator>
void test_no_offset(const gsl::not_null<Generator*> generator) {
  test_only_transition_no_offset();
  test_gradient_no_offset();
  test_in_shape_map_no_offset(generator);
}

template <typename Generator>
void test_offset(const gsl::not_null<Generator*> generator) {
  test_only_transition_offset();
  // We don't test the gradient explicitly here because the formulas would be
  // too complicated, so we just let it be tested in the shape map test
  test_in_shape_map_offset(generator);
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Shape.Wedge", "[Domain][Unit]") {
  MAKE_GENERATOR(generator);
  domain::CoordinateMaps::ShapeMapTransitionFunctions::
      register_derived_with_charm();
  test_no_offset(make_not_null(&generator));
  test_offset(make_not_null(&generator));
  test_errors();
}
}  // namespace
}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
