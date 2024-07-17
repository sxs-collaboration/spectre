// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>

#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
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
#include "Utilities/StdArrayHelpers.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {
namespace {
std::array<double, 3> sph_to_cart(const double radius, const double theta,
                                  const double phi) {
  return std::array{radius * sin(theta) * cos(phi),
                    radius * sin(theta) * sin(phi), radius * cos(theta)};
}

// Since this is hard to test for general points without just reimplementing the
// function, we test for specific points where it's easier to calculate
void test_gradient() {
  INFO("Test gradient");
  const double inner_radius = 0.5;
  const double outer_radius = 10.0;

  const auto make_wedge = [&](const double inner_sphericity,
                              const double outer_sphericity) {
    return Wedge{inner_radius, outer_radius, inner_sphericity, outer_sphericity,
                 2_st};
  };

  // The extra grad factors should be 0 or 1 for spherical or flat boundaries
  // (this is opposite of their sphericity)
  const auto compute_gradient =
      [&](const double radius, const std::array<double, 3>& point,
          const double inner_distance, const double outer_distance,
          const double extra_inner_grad_factor,
          const double extra_outer_grad_factor) {
        const double distance_difference = outer_distance - inner_distance;
        const std::array<double, 3> grad_base =
            1.0 / (sqrt(3.0) * radius) *
            std::array{point[0], point[1],
                       -(square(radius) - square(point[2])) / point[2]} /
            point[2];
        const std::array<double, 3> inner_grad =
            extra_inner_grad_factor * inner_radius * grad_base;
        const std::array<double, 3> outer_grad =
            extra_outer_grad_factor * outer_radius * grad_base;
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
          (compute_gradient(radius, point, inner_distance, outer_distance,
                            1.0 - inner_sphericity, 1.0 - outer_sphericity)));
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
            (compute_gradient(radius, point, inner_radius, outer_radius,
                              1.0 - inner_sphericity, 1.0 - outer_sphericity)));
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

void test_only_transition() {
  const double inner_radius = 0.5;
  const double outer_radius = 10.0;
  const double inner_sphericity = 1.0;
  const double outer_sphericity = 0.0;

  const Wedge wedge{inner_radius, outer_radius, inner_sphericity,
                    outer_sphericity, 0_st};

  const double inner_distance = 0.5;
  const double outer_distance = outer_radius / sqrt(3.0);
  const double distance_difference = outer_distance - inner_distance;

  std::array<double, 3> point{4.0, 0.0, 0.0};
  const double function_value = (outer_distance - 4.0) / distance_difference;

  CHECK(wedge(point) == approx(function_value));
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

  test_gradient();
}

void test_in_shape_map() {
  INFO("Test using shape map");
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> coef_dist{-0.01, 0.01};

  const double initial_time = 1.0;
  const double check_time = 2.0;
  const size_t l_max = 4;
  const size_t num_coefs = 2 * square(l_max + 1);
  const std::array center{0.1, -0.2, 0.3};
  const std::string fot_name{"TheBean"};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  auto coefs = make_with_random_values<DataVector>(make_not_null(&generator),
                                                   make_not_null(&coef_dist),
                                                   DataVector{num_coefs});
  functions_of_time[fot_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<0>>(
          initial_time, std::array{std::move(coefs)},
          std::numeric_limits<double>::infinity());

  // We'll keep the radii the same since they don't really matter
  const double inner_radius = 0.5;
  const double outer_radius = 10.0;

  std::uniform_real_distribution<double> angle_dist{0.0, 2.0 * M_PI};

  for (const auto& [inner_sphericity, outer_sphericity] :
       cartesian_product(std::array{1.0, 0.0}, std::array{1.0, 0.0})) {
    CAPTURE(inner_sphericity);
    CAPTURE(outer_sphericity);
    // This guarantees the radius of the point is within the wedge
    std::uniform_real_distribution<double> radial_dist{
        inner_radius, outer_radius / sqrt(3.0)};

    std::unique_ptr<ShapeMapTransitionFunction> wedge{};

    // Test 10 points for each sphericity and orientation
    for (size_t i = 0; i < 10; i++) {
      const double radius = radial_dist(generator);
      const double theta = 0.5 * angle_dist(generator);
      const double phi = angle_dist(generator);

      CAPTURE(radius);
      CAPTURE(theta);
      CAPTURE(phi);

      const std::array<double, 3> centered_point =
          sph_to_cart(radius, theta, phi);

      size_t axis = 0;
      double max = abs(centered_point[0]);
      for (size_t j = 1; j < 3; j++) {
        if (const double maybe_max = abs(gsl::at(centered_point, j));
            maybe_max > max) {
          axis = j;
          max = maybe_max;
        }
      }

      wedge = std::make_unique<Wedge>(inner_radius, outer_radius,
                                      inner_sphericity, outer_sphericity, axis);

      TimeDependent::Shape shape{center, l_max, l_max, std::move(wedge),
                                 fot_name};

      const std::array<double, 3> point = centered_point + center;

      test_coordinate_map_argument_types(shape, point, check_time,
                                         functions_of_time);
      test_inverse_map(shape, point, check_time, functions_of_time);
      test_frame_velocity(shape, point, check_time, functions_of_time);
      test_jacobian(shape, point, check_time, functions_of_time);
      test_inv_jacobian(shape, point, check_time, functions_of_time);
    }
  }
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Shape.Wedge", "[Domain][Unit]") {
  test_only_transition();
  test_in_shape_map();
}
}  // namespace
}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
