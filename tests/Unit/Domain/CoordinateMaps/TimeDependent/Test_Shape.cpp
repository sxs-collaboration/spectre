// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "ApparentHorizons/Tags.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/RegisterDerivedWithCharm.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/ShapeMapTransitionFunction.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace domain {
namespace {

const std::complex<double> imag(0, 1);

std::complex<double> Y00(double /*theta*/, double /*phi*/) {
  return 0.5 * sqrt(1. / M_PI);
}

std::complex<double> Y10(double theta, double /*phi*/) {
  return 0.5 * sqrt(3. / M_PI) * cos(theta);
}

std::complex<double> Y11(double theta, double phi) {
  return -0.5 * sqrt(3. / (2. * M_PI)) * sin(theta) *
         std::exp<double>(imag * phi);
}

std::complex<double> Y20(double theta, double /*phi*/) {
  return 0.25 * sqrt(5. / M_PI) * (3. * cos(theta) * cos(theta) - 1.);
}

std::complex<double> Y21(double theta, double phi) {
  return -0.5 * sqrt(15. / (2. * M_PI)) * sin(theta) * cos(theta) *
         std::exp<double>(imag * phi);
}

std::complex<double> Y22(double theta, double phi) {
  return 0.25 * sqrt(15. / (2. * M_PI)) * sin(theta) * sin(theta) *
         std::exp<double>(2. * imag * phi);
}

std::complex<double> Y30(double theta, double /*phi*/) {
  return 0.25 * sqrt(7. / M_PI) * (5 * pow(cos(theta), 3) - 3 * cos(theta));
}

std::complex<double> Y31(double theta, double phi) {
  return -0.125 * sqrt(21. / M_PI) * sin(theta) *
         (5 * cos(theta) * cos(theta) - 1) * std::exp<double>(imag * phi);
}

std::complex<double> Y32(double theta, double phi) {
  return 0.25 * sqrt(105. / (2. * M_PI)) * sin(theta) * sin(theta) *
         cos(theta) * std::exp<double>(2. * imag * phi);
}

std::complex<double> Y33(double theta, double phi) {
  return -0.125 * sqrt(35. / M_PI) * pow(sin(theta), 3) *
         std::exp<double>(3. * imag * phi);
}

// vector of spherical harmonics, e.g. Y21 is harmonics[2][1]
using SphericalHarmonic = std::function<std::complex<double>(double, double)>;
std::vector<std::vector<SphericalHarmonic>> harmonics = {
    {Y00}, {Y10, Y11}, {Y20, Y21, Y22}, {Y30, Y31, Y32, Y33}};

// returns Ylm
auto identity = [](size_t l, size_t m) {
  return [l, m](double theta, double phi) {
    return gsl::at(gsl::at(harmonics, l), m)(theta, phi);
  };
};

// computes the coefficient C_{lm} = sqrt((l+m)(l-m)/(2l-1)(2l+1)) used by
// `dtheta` below
double c_coef(size_t l, size_t m) {
  return sqrt(static_cast<double>((l + m) * (l - m)) /
              ((2. * l - 1.) * (2. * l + 1.)));
}

// returns sin(\theta) \partial_\theta Y_{l m} = l C_{l+1 m} Y_{l+1 m} -
// (l+1) C_{l m} Y_{l m}
auto dtheta = [](size_t l, size_t m) {
  return [l, m](double theta, double phi) {
    const auto term1 = l * c_coef(l + 1, m) *
                       gsl::at(gsl::at(harmonics, l + 1), m)(theta, phi);
    const auto term2 =
        (l > 0 and m < l)
            ? (l + 1.) * c_coef(l, m) *
                  gsl::at(gsl::at(harmonics, l - 1), m)(theta, phi)
            : std::complex(0.);
    return term1 - term2;
  };
};

// returns \partial_\phi Y_{l m} = i m Y_{l m}
auto dphi = [](size_t l, size_t m) {
  return [l, m](double theta, double phi) {
    return std::complex(0., 1.) * static_cast<double>(m) *
           gsl::at(gsl::at(harmonics, l), m)(theta, phi);
  };
};

// Sums up spherical harmonics using the provided functions and coefficients
auto evaluate_harmonic_expansion =
    [](auto function, double theta, double phi,
       const std::vector<std::vector<std::complex<double>>>& coefs,
       const std::optional<double>& lambda_00_coef, size_t l_max,
       size_t m_max) {
      std::complex<double> res = 0.;
      for (size_t l = 0; l <= l_max; ++l) {
        if (l == 0 and lambda_00_coef.has_value()) {
          CHECK(coefs[0][0] == 0.0);
          res += lambda_00_coef.value() * function(l, 0)(theta, phi);
        } else {
          res += gsl::at(gsl::at(coefs, l), 0) * function(l, 0)(theta, phi);
        }
        for (size_t m = 1; m <= std::min(l, m_max); ++m) {
          res += 2. * (std::real(gsl::at(gsl::at(coefs, l), m)) *
                           std::real(function(l, m)(theta, phi)) -
                       std::imag(gsl::at(gsl::at(coefs, l), m)) *
                           std::imag(function(l, m)(theta, phi)));
        }
      }
      ASSERT(std::imag(res) == approx(0.),
             "Expansion has non-zero imaginary component: " << std::imag(res));
      return std::real(res);
    };

using FunctionsOfTimeMap =
    std::unordered_map<std::string,
                       std::unique_ptr<FunctionsOfTime::FunctionOfTime>>;

std::vector<std::vector<std::complex<double>>> generate_random_coefs(
    size_t l_max, size_t m_max, gsl::not_null<std::mt19937*> generator) {
  std::vector<std::vector<std::complex<double>>> coefs{};
  // if the coefficients are made too large, the map has a good chance of
  // mapping through the center, causing the test to fail which is why we damp
  // higher mode coefficients by `8 * l * l + 1`. This damping factor was found
  // empirically by trial and error.
  std::uniform_real_distribution<double> coef_dist{0., 1.};

  for (size_t l = 0; l <= l_max; ++l) {
    // m=0 is real
    std::vector<std::complex<double>> tmp{
        make_with_random_values<double>(generator, coef_dist, 1) /
        double(8 * l * l + 1)};
    for (size_t m = 1; m <= std::min(l, m_max); ++m) {
      tmp.emplace_back(make_with_random_values<std::complex<double>>(
                           generator, coef_dist, 2) /
                       double(8 * l * l + 1));
    }
    coefs.emplace_back(tmp);
  }

  return coefs;
}

double generate_random_00_coef(const gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution<double> coef_dist{0., 1.};
  return coef_dist(*generator);
}

// converts complex coefficients to format expected by spherepack
DataVector convert_coefs_to_spherepack(
    const std::vector<std::vector<std::complex<double>>>& coefs, size_t l_max,
    size_t m_max) {
  SpherepackIterator iter(l_max, m_max);
  auto spherepack_coefs =
      make_with_value<DataVector>(iter.spherepack_array_size(), 0.);

  const double sqrt_2_by_pi = sqrt(2. / M_PI);
  for (size_t l = 0; l <= l_max; ++l) {
    iter.set(l, 0);
    spherepack_coefs[iter()] =
        sqrt_2_by_pi * std::real(gsl::at(gsl::at(coefs, l), 0));
    for (size_t m = 1; m <= std::min(l, m_max); ++m) {
      iter.set(l, m);
      spherepack_coefs[iter()] =
          pow(-1, m) * sqrt_2_by_pi * std::real(gsl::at(gsl::at(coefs, l), m));
      iter.set(l, -m);
      spherepack_coefs[iter()] =
          pow(-1, m) * sqrt_2_by_pi * std::imag(gsl::at(gsl::at(coefs, l), m));
    }
  }
  return spherepack_coefs;
}

// Generates the map, time, and a FunctionOfTime. `const_f_of_t` decides whether
// the function of time will be constant which is needed for the analytical
// comparisons.
template <typename TransitionFunction>
void generate_random_map_time_and_f_of_time(
    const gsl::not_null<CoordinateMaps::TimeDependent::Shape*> map,
    const gsl::not_null<double*> time,
    const gsl::not_null<FunctionsOfTimeMap*> functions_of_time,
    const size_t l_max, const size_t m_max, const std::array<double, 3>& center,
    const TransitionFunction& transition_func, const DataVector& ylm_coefs,
    const std::optional<double>& lambda_00_coef, const bool const_f_of_t,
    gsl::not_null<std::mt19937*> generator) {
  const std::string shape_f_of_t_name{"Shape"};
  const std::optional<std::string> size_f_of_t_name{"Size"};

  *map = CoordinateMaps::TimeDependent::Shape(
      center, l_max, m_max,
      std::make_unique<TransitionFunction>(transition_func), shape_f_of_t_name,
      lambda_00_coef.has_value() ? size_f_of_t_name : std::nullopt);
  // Choose a random time for evaluating the FunctionOfTime
  std::uniform_real_distribution<double> time_dist{-1.0, 1.0};
  *time = time_dist(*generator);

  std::uniform_real_distribution<> dt_dis{0.1, 0.5};
  const double initial_time{*time - dt_dis(*generator)};
  const double expiration_time{*time + dt_dis(*generator)};

  DataVector shape_dtcoefs{ylm_coefs.size(), 0.};
  DataVector shape_ddtcoefs{ylm_coefs.size(), 0.};

  if (not const_f_of_t) {
    const auto dt_random_coefs = generate_random_coefs(l_max, m_max, generator);
    shape_dtcoefs = convert_coefs_to_spherepack(dt_random_coefs, l_max, m_max);
    const auto ddt_random_coefs =
        generate_random_coefs(l_max, m_max, generator);
    shape_ddtcoefs =
        convert_coefs_to_spherepack(ddt_random_coefs, l_max, m_max);
  }

  if (lambda_00_coef.has_value()) {
    DataVector size_dtcoefs{1, 0.};
    DataVector size_ddtcoefs{1, 0.};
    if (not const_f_of_t) {
      size_dtcoefs[0] = generate_random_00_coef(generator);
      size_ddtcoefs[0] = generate_random_00_coef(generator);
    }
    const std::array<DataVector, 3> initial_size_coefficients = {
        DataVector{lambda_00_coef.value()}, size_dtcoefs, size_ddtcoefs};
    (*functions_of_time)[size_f_of_t_name.value()] =
        std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
            initial_time, initial_size_coefficients, expiration_time);

    shape_dtcoefs[0] = 0.0;
    shape_ddtcoefs[0] = 0.0;
  }

  const std::array<DataVector, 3> initial_shape_coefficients = {
      ylm_coefs, shape_dtcoefs, shape_ddtcoefs};

  (*functions_of_time)[shape_f_of_t_name] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, initial_shape_coefficients, expiration_time);
}

template <typename TransitionFunction>
void test_map_helpers(const TransitionFunction& transition_func, size_t l_max,
                      size_t m_max, const bool include_size,
                      gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution dist{-10., 10.};
  const auto center =
      make_with_random_values<std::array<double, 3>>(generator, dist, 3);

  FunctionsOfTimeMap functions_of_time{};
  double time{};
  auto map = CoordinateMaps::TimeDependent::Shape{};
  auto random_coefs = generate_random_coefs(l_max, m_max, generator);
  DataVector spherepack_coefs =
      convert_coefs_to_spherepack(random_coefs, l_max, m_max);
  std::optional<double> random_00_coef{};
  if (include_size) {
    spherepack_coefs[0] = 0.0;
    random_00_coef = generate_random_00_coef(generator);
  }

  generate_random_map_time_and_f_of_time(
      make_not_null(&map), make_not_null(&time),
      make_not_null(&functions_of_time), l_max, m_max, center, transition_func,
      spherepack_coefs, random_00_coef, false, generator);

  const auto random_point =
      make_with_random_values<std::array<double, 3>>(generator, dist, 3);

  // Check map against a suite of functions in
  // tests/Unit/Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp
  test_serialization(map);
  test_copy_semantics(map);
  CHECK_FALSE(map != map);
  test_coordinate_map_argument_types(map, random_point, time,
                                     functions_of_time);
  test_inverse_map(map, random_point, time, functions_of_time);
  test_frame_velocity(map, random_point, time, functions_of_time);
  test_jacobian(map, random_point, time, functions_of_time);
  test_inv_jacobian(map, random_point, time, functions_of_time);
}

// duplicates map but calculates spherical harmonics expansion directly.
std::array<DataVector, 3> calculate_analytical_map(
    std::array<DataVector, 3> target_points, std::array<double, 3> center,
    size_t l_max, size_t m_max,
    const std::vector<std::vector<std::complex<double>>>& coefs,
    const std::optional<double>& lambda_00_coef,
    const CoordinateMaps::ShapeMapTransitionFunctions::
        ShapeMapTransitionFunction& transition_func) {
  const size_t num_points = target_points[0].size();

  const auto centered_coords = target_points - center;

  const DataVector target_thetas =
      atan2(hypot(centered_coords[0], centered_coords[1]), centered_coords[2]);
  const DataVector target_phis = atan2(centered_coords[1], centered_coords[0]);

  DataVector angular_part(num_points);

  for (size_t i = 0; i < num_points; ++i) {
    angular_part[i] = evaluate_harmonic_expansion(
        identity, gsl::at(target_thetas, i), gsl::at(target_phis, i), coefs,
        lambda_00_coef, l_max, m_max);
  }
  const auto spatial_part = transition_func(centered_coords);
  return target_points - centered_coords * angular_part * spatial_part;
}

template <typename TransitionFunction>
void test_analytical_solution(const TransitionFunction& transition_func,
                              size_t l_max, size_t m_max,
                              const bool include_size, size_t num_points,
                              gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution dist{-10., 10.};
  const auto center =
      make_with_random_values<std::array<double, 3>>(generator, dist, 3);
  const auto target_data = make_with_random_values<std::array<DataVector, 3>>(
      generator, dist, num_points);

  auto random_coefs = generate_random_coefs(l_max, m_max, generator);

  FunctionsOfTimeMap functions_of_time{};
  double time{};
  auto map = CoordinateMaps::TimeDependent::Shape{};

  DataVector spherepack_coefs =
      convert_coefs_to_spherepack(random_coefs, l_max, m_max);
  std::optional<double> random_00_coef{};
  if (include_size) {
    random_coefs[0][0] = 0.0;
    spherepack_coefs[0] = 0.0;
    random_00_coef = generate_random_00_coef(generator);
  }

  generate_random_map_time_and_f_of_time(
      make_not_null(&map), make_not_null(&time),
      make_not_null(&functions_of_time), l_max, m_max, center, transition_func,
      spherepack_coefs, random_00_coef, true, generator);
  const auto mapped_result = map(target_data, time, functions_of_time);
  const auto analytical_result =
      calculate_analytical_map(target_data, center, l_max, m_max, random_coefs,
                               random_00_coef, transition_func);
  CHECK_ITERABLE_APPROX(mapped_result, analytical_result);
}

// calculate the Jacobian using spherical harmonics directly
tnsr::Ij<DataVector, 3, Frame::NoFrame> calculate_analytical_jacobian(
    const std::array<DataVector, 3>& target_points,
    const std::array<double, 3>& center, size_t l_max, size_t m_max,
    const std::vector<std::vector<std::complex<double>>>& coefs,
    const std::optional<double>& lambda_00_coef,
    const CoordinateMaps::ShapeMapTransitionFunctions::
        ShapeMapTransitionFunction& transition_func) {
  const size_t num_points = target_points[0].size();
  const auto centered_coords = target_points - center;
  const DataVector target_thetas =
      atan2(hypot(centered_coords[0], centered_coords[1]), centered_coords[2]);
  const DataVector target_phis = atan2(centered_coords[1], centered_coords[0]);
  DataVector angular_part(num_points);
  DataVector theta_gradient(num_points);
  DataVector phi_gradient(num_points);
  for (size_t i = 0; i < num_points; ++i) {
    angular_part.at(i) = evaluate_harmonic_expansion(
        identity, gsl::at(target_thetas, i), gsl::at(target_phis, i), coefs,
        lambda_00_coef, l_max, m_max);
    theta_gradient.at(i) = evaluate_harmonic_expansion(
        dtheta, gsl::at(target_thetas, i), gsl::at(target_phis, i), coefs,
        lambda_00_coef, l_max, m_max);
    phi_gradient.at(i) = evaluate_harmonic_expansion(
        dphi, gsl::at(target_thetas, i), gsl::at(target_phis, i), coefs,
        lambda_00_coef, l_max, m_max);
  }
  // multiply angular gradient by inverse jacobian to get cartesian gradient
  std::array<DataVector, 3> cartesian_gradient{};
  cartesian_gradient[0] =
      (cos(target_thetas) * cos(target_phis) * theta_gradient -
       sin(target_phis) * phi_gradient) /
      sin(target_thetas);
  cartesian_gradient[1] =
      (cos(target_thetas) * sin(target_phis) * theta_gradient +
       cos(target_phis) * phi_gradient) /
      sin(target_thetas);
  cartesian_gradient[2] = -theta_gradient;

  // this part essentially duplicates the code from the map
  const auto spatial_part = transition_func(centered_coords);
  const auto spatial_gradient = transition_func.gradient(centered_coords);
  const auto spatial_part_over_radius =
      transition_func.map_over_radius(centered_coords);
  tnsr::Ij<DataVector, 3, Frame::NoFrame> result(num_points);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      result.get(i, j) =
          -gsl::at(centered_coords, i) *
          (angular_part * gsl::at(spatial_gradient, j) +
           gsl::at(cartesian_gradient, j) * spatial_part_over_radius);
    }
    result.get(i, i) += 1. - angular_part * spatial_part;
  }
  return result;
}

template <typename TransitionFunction>
void test_analytical_jacobian(const TransitionFunction& transition_func,
                              size_t l_max, size_t m_max,
                              const bool include_size, size_t num_points,
                              gsl::not_null<std::mt19937*> generator) {
  std::uniform_real_distribution dist{-10., 10.};
  const auto center =
      make_with_random_values<std::array<double, 3>>(generator, dist, 3);
  auto target_data = make_with_random_values<std::array<DataVector, 3>>(
      generator, dist, num_points);

  // the analytical solution divides by sin(theta), so it might get into
  // trouble at the poles, although it appeared fine without this safety
  // measure in >10^8 tries. Here, the y-coordinate is increased if the point
  // has theta close to zero.
  const auto centered_coords = target_data - center;
  const DataVector sin_thetas = sin(
      atan2(hypot(centered_coords[0], centered_coords[1]), centered_coords[2]));
  for (size_t i = 0; i < sin_thetas.size(); ++i) {
    if (abs(gsl::at(sin_thetas, i)) < 1e-4) {
      target_data[1][i] += 1.;
    }
  }

  auto random_coefs = generate_random_coefs(l_max, m_max, generator);

  FunctionsOfTimeMap functions_of_time{};
  double time{};
  auto map = CoordinateMaps::TimeDependent::Shape{};

  DataVector spherepack_coefs =
      convert_coefs_to_spherepack(random_coefs, l_max, m_max);
  std::optional<double> random_00_coef{};
  if (include_size) {
    random_coefs[0][0] = 0.0;
    spherepack_coefs[0] = 0.0;
    random_00_coef = generate_random_00_coef(generator);
  }

  generate_random_map_time_and_f_of_time(
      make_not_null(&map), make_not_null(&time),
      make_not_null(&functions_of_time), l_max, m_max, center, transition_func,
      spherepack_coefs, random_00_coef, true, generator);
  const auto mapped_jacobian =
      map.jacobian(target_data, time, functions_of_time);
  const auto analytical_jacobian = calculate_analytical_jacobian(
      target_data, center, l_max, m_max, random_coefs, random_00_coef,
      transition_func);
  CHECK_ITERABLE_APPROX(mapped_jacobian, analytical_jacobian);
}

template <typename Generator>
void test_inverse(const gsl::not_null<Generator*> generator) {
  using TransitionFunc =
      CoordinateMaps::ShapeMapTransitionFunctions::SphereTransition;
  const double time = 1.0;
  const TransitionFunc sphere_transition{1.0, 1.5};
  CoordinateMaps::TimeDependent::Shape shape{
      std::array{0.0, 0.0, 0.0}, 10, 10,
      std::make_unique<TransitionFunc>(sphere_transition), "Shape"};

  DataVector coefs{ylm::Spherepack::spectral_size(10, 10), 0.0};
  DataVector dt_coefs{ylm::Spherepack::spectral_size(10, 10), 0.0};
  DataVector d2t_coefs{ylm::Spherepack::spectral_size(10, 10), 0.0};

  const double factor = sqrt(2.0 / M_PI);

  coefs[0] = 0.5 * factor;
  dt_coefs[0] = -0.04 * factor;
  d2t_coefs[0] = 0.003 * factor;

  FunctionsOfTimeMap functions_of_time{};
  functions_of_time["Shape"] =
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          time, std::array{coefs, dt_coefs, d2t_coefs}, time + 3.0);

  std::uniform_real_distribution<double> dist_phi{0.0, 2.0 * M_PI};
  std::uniform_real_distribution<double> dist_theta{0.0, M_PI};

  for (const double radius : std::array{1.0, 1.2, 1.5}) {
    const double theta = dist_theta(*generator);
    const double phi = dist_phi(*generator);

    // Random point on a sphere of that radius
    const double x = radius * sin(theta) * cos(phi);
    const double y = radius * sin(theta) * sin(phi);
    const double z = radius * cos(theta);
    const std::array<double, 3> grid_coords{x, y, z};

    const auto inertial_coords = shape(grid_coords, time, functions_of_time);

    const auto mapped_coords =
        shape.inverse(inertial_coords, time, functions_of_time);
    CHECK(mapped_coords.has_value());
    const double mapped_radius = magnitude(mapped_coords.value());

    CHECK_ITERABLE_APPROX(grid_coords, mapped_coords.value());
    CHECK(radius == approx(mapped_radius));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.TimeDependent.Shape",
                  "[Domain][Unit]") {
  domain::CoordinateMaps::ShapeMapTransitionFunctions::
      register_derived_with_charm();
  const CoordinateMaps::ShapeMapTransitionFunctions::SphereTransition
      sphere_transition{1e-7, 100.};

  MAKE_GENERATOR(generator);

  test_inverse(make_not_null(&generator));

  for (const auto include_size : make_array(false, true)) {
    CAPTURE(include_size);
    {
      INFO("Testing MapHelpers");
      for (size_t l_max = 2; l_max < 12; ++l_max) {
        for (size_t m_max = 2; m_max <= l_max; ++m_max) {
          CAPTURE(l_max, m_max);
          test_map_helpers(sphere_transition, l_max, m_max, include_size,
                           make_not_null(&generator));
        }
      }
    }
    {
      INFO("Testing analytical solution");
      for (size_t l_max = 2; l_max <= 3; ++l_max) {
        for (size_t m_max = 2; m_max <= l_max; ++m_max) {
          CAPTURE(l_max, m_max);
          test_analytical_solution(sphere_transition, l_max, m_max,
                                   include_size, 1000,
                                   make_not_null(&generator));
        }
      }
    }
    {
      INFO("Testing analytical gradient");
      const size_t l_max = 2;
      const size_t m_max = 2;
      test_analytical_jacobian(sphere_transition, l_max, m_max, include_size,
                               1000, make_not_null(&generator));
    }
  }
}
}  // namespace domain
