// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Integration/GslQuadAdaptive.hpp"
#include "Utilities/ErrorHandling/Exceptions.hpp"

namespace {
// [integrated_function]
double gaussian(const double x, const double mean, const double factor) {
  return 2. * factor / sqrt(M_PI) * exp(-square(x - mean));
}
// [integrated_function]

double integrable_singularity(const double x, const double factor) {
  return factor * cos(sqrt(abs(x))) / sqrt(abs(x));
}

SPECTRE_TEST_CASE("Unit.Numerical.Integration.GslQuadAdaptive",
                  "[Unit][NumericalAlgorithms]") {
  const double absolute_tolerance = 1.e-10;
  Approx custom_approx = Approx::custom().epsilon(absolute_tolerance);

  const double error_causing_tolerance =
      1.e-3 * std::numeric_limits<double>::epsilon();

  {
    INFO("StandardGaussKronrod");
    // Construct the integration and give an example
    // [integration_example]
    const integration::GslQuadAdaptive<
        integration::GslIntegralType::StandardGaussKronrod>
        integration{20};
    const double mean = 5.;
    const double factor = 2.;
    const double lower_boundary = -4.;
    const double upper_boundary = 10.;
    const auto result = integration(
        [&mean, &factor](const double x) { return gaussian(x, mean, factor); },
        lower_boundary, upper_boundary, absolute_tolerance, 4);
    // [integration_example]
    CHECK(result == custom_approx(factor * erf(upper_boundary - mean) -
                                  factor * erf(lower_boundary - mean)));
    CHECK_THROWS_AS(
        integration([&mean, &factor](
                        const double x) { return gaussian(x, mean, factor); },
                    lower_boundary, upper_boundary, error_causing_tolerance, 4),
        convergence_error);
  }

  {
    INFO("InfiniteInterval");
    const integration::GslQuadAdaptive<
        integration::GslIntegralType::InfiniteInterval>
        integration{20};
    const double mean = 5.;
    const double factor = 2.;
    const auto result = integration(
        [&mean, &factor](const double x) { return gaussian(x, mean, factor); },
        absolute_tolerance);
    CHECK(result == custom_approx(2. * factor));
    CHECK_THROWS_AS(
        integration([&mean, &factor](
                        const double x) { return gaussian(x, mean, factor); },
                    error_causing_tolerance),
        convergence_error);
  }

  {
    INFO("UpperBoundaryInfinite");
    const integration::GslQuadAdaptive<
        integration::GslIntegralType::UpperBoundaryInfinite>
        integration{20};
    const double mean = 5.;
    const double factor = 2.;
    const double lower_boundary = -4.;
    auto result = integration(
        [&mean, &factor](const double x) { return gaussian(x, mean, factor); },
        lower_boundary, absolute_tolerance);
    CHECK(result == custom_approx(factor * (1. - erf(lower_boundary - mean))));
    CHECK_THROWS_AS(
        integration([&mean, &factor](
                        const double x) { return gaussian(x, mean, factor); },
                    lower_boundary, error_causing_tolerance),
        convergence_error);
  }

  {
    INFO("LowerBoundaryInfinite");
    const integration::GslQuadAdaptive<
        integration::GslIntegralType::LowerBoundaryInfinite>
        integration{20};
    const double mean = 5.;
    const double factor = 2.;
    const double upper_boundary = 10.;
    auto result = integration(
        [&mean, &factor](const double x) { return gaussian(x, mean, factor); },
        upper_boundary, absolute_tolerance);
    CHECK(result == custom_approx(factor * (1. + erf(upper_boundary - mean))));
    CHECK_THROWS_AS(
        integration([&mean, &factor](
                        const double x) { return gaussian(x, mean, factor); },
                    upper_boundary, error_causing_tolerance),
        convergence_error);
  }

  {
    INFO("IntegrableSingularitiesPresent");
    const integration::GslQuadAdaptive<
        integration::GslIntegralType::IntegrableSingularitiesPresent>
        integration{20};
    const double factor = 2.;
    const double upper_boundary = square(0.5 * M_PI);
    auto result = integration(
        [&factor](const double x) { return integrable_singularity(x, factor); },
        0., upper_boundary, absolute_tolerance);
    CHECK(result == custom_approx(2. * factor * (sin(sqrt(upper_boundary)))));
    CHECK_THROWS_AS(integration(
                        [&factor](const double x) {
                          return integrable_singularity(x, factor);
                        },
                        0., upper_boundary, error_causing_tolerance),
                    convergence_error);
  }

  {
    INFO("IntegrableSingularitiesKnown");
    const integration::GslQuadAdaptive<
        integration::GslIntegralType::IntegrableSingularitiesKnown>
        integration{30};
    const double factor = 2.;
    const double upper_boundary = square(0.5 * M_PI);
    const std::vector<double> points{-upper_boundary, 0., upper_boundary};
    auto result = integration(
        [&factor](const double x) { return integrable_singularity(x, factor); },
        points, absolute_tolerance);
    CHECK(result == custom_approx(4. * factor * sin(sqrt(upper_boundary))));
    CHECK_THROWS_AS(integration(
                        [&factor](const double x) {
                          return integrable_singularity(x, factor);
                        },
                        points, error_causing_tolerance),
                    convergence_error);
  }
}
}  // namespace
