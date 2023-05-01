// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <random>

#include "ApparentHorizons/StrahlkorperInDifferentFrame.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/Gsl.hpp"

namespace {

template<bool Aligned, typename DestFrame>
void test_strahlkorper_in_different_frame() {
  const size_t grid_points_each_dimension = 5;

  // Set up a Strahlkorper corresponding to a Schwarzschild hole of
  // mass 1, in the grid frame.
  // Center the Strahlkorper at (0.03,0.02,0.01) so that we test a
  // nonzero center.
  const std::array<double, 3> strahlkorper_grid_center = {0.03, 0.02, 0.01};
  const size_t l_max = 8;
  const Strahlkorper<Frame::Grid> strahlkorper_grid(l_max, 2.0,
                                                    strahlkorper_grid_center);

  // Create a Domain.
  // We choose a spherical shell domain extending from radius 1.9M to
  // 2.9M, so that the Strahlkorper is inside the domain. It gives a
  // narrow domain so that we don't need a large number of grid points
  // to resolve the horizon (which would make the test slower).
  std::vector<double> radial_partitioning{};
  std::vector<domain::CoordinateMaps::Distribution> radial_distribution{
      domain::CoordinateMaps::Distribution::Linear};
  std::unique_ptr<domain::creators::Sphere> domain_creator;
  if constexpr (Aligned) {
    if constexpr (std::is_same_v<DestFrame, ::Frame::Inertial>) {
      domain_creator.reset(new domain::creators::Sphere(
          1.9, 2.9, domain::creators::Sphere::Excision{}, 1_st,
          grid_points_each_dimension, false, std::nullopt, radial_partitioning,
          radial_distribution, ShellWedges::All,
          // Choose time dependence to be centered at the strahlkorper center
          // and spherical.
          std::make_unique<
              domain::creators::time_dependence::SphericalCompression>(
              0.0, 1.0, 12.0, strahlkorper_grid_center, 0.05, 0.1, 0.0)));
    } else {
      domain_creator.reset(new domain::creators::Sphere(
          1.9, 2.9, domain::creators::Sphere::Excision{}, 1_st,
          grid_points_each_dimension, false, std::nullopt, radial_partitioning,
          radial_distribution, ShellWedges::All,
          // Choose time dependence to be centered at the strahlkorper center
          // and spherical.
          std::make_unique<
              domain::creators::time_dependence::SphericalCompression>(
              0.0, 1.0, 12.0, strahlkorper_grid_center, 0.05, 0.1, 0.0, 0.025,
              0.05, 0.0)));
    }
  } else {
    domain_creator.reset(new domain::creators::Sphere(
        1.9, 2.9, domain::creators::Sphere::Excision{}, 1_st,
        grid_points_each_dimension, false, std::nullopt, radial_partitioning,
        radial_distribution, ShellWedges::All,
        std::make_unique<
            domain::creators::time_dependence::UniformTranslation<3>>(
            0.0, std::array<double, 3>({{0.01, 0.02, 0.03}}))));
  }
  Domain<3> domain = domain_creator->create_domain();
  const auto functions_of_time = domain_creator->functions_of_time();

  // Compute strahlkorper in the destination frame.
  const double time = 0.5;
  Strahlkorper<DestFrame> strahlkorper_dest{};
  if constexpr (Aligned) {
    strahlkorper_in_different_frame_aligned(
        make_not_null(&strahlkorper_dest), strahlkorper_grid, domain,
        functions_of_time, time);

  } else {
    strahlkorper_in_different_frame(make_not_null(&strahlkorper_dest),
                                    strahlkorper_grid, domain,
                                    functions_of_time, time);
  }

  // Now compare.
  std::unique_ptr<Strahlkorper<DestFrame>> strahlkorper_expected;
  if constexpr (Aligned) {
    // Here work out by hand the SphericalCompression map.
    const double old_radius = 2.0;
    const double rho_max    = 12.0;
    const double rho_min    = 1.0;
    const double lambda_00  = 0.1;
    const double new_radius =
        old_radius * (1.0 - lambda_00 * (rho_max / old_radius - 1.0) /
                                ((rho_max - rho_min) * sqrt(4.0 * M_PI)));
    strahlkorper_expected.reset(new Strahlkorper<DestFrame>(
        l_max, new_radius, strahlkorper_grid_center));
  } else {
    strahlkorper_expected.reset(new Strahlkorper<DestFrame>(
        l_max, 2.0,
        {{strahlkorper_grid_center[0] + 0.005,
          strahlkorper_grid_center[1] + 0.01,
          strahlkorper_grid_center[2] + 0.015}}));
  }
  CHECK_ITERABLE_APPROX(strahlkorper_expected->physical_center(),
                        strahlkorper_dest.physical_center());
  CHECK_ITERABLE_APPROX(strahlkorper_expected->coefficients(),
                        strahlkorper_dest.coefficients());
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperInDifferentFrame",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  test_strahlkorper_in_different_frame<false, Frame::Inertial>();
  test_strahlkorper_in_different_frame<true, Frame::Inertial>();
  test_strahlkorper_in_different_frame<true, Frame::Distorted>();
}

}  // namespace
