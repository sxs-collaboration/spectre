// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <random>

#include "ApparentHorizons/SpherepackIterator.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/StrahlkorperInDifferentFrame.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

namespace {

void test_strahlkorper_in_different_frame() {
  const size_t grid_points_each_dimension = 5;
  const double expiration_time = 1.0;

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
  domain::creators::Shell domain_creator(
      1.9, 2.9, 1,
      std::array<size_t, 2>{grid_points_each_dimension,
                            grid_points_each_dimension},
      false, 1.0, radial_partitioning, radial_distribution, ShellWedges::All,
      std::make_unique<
          domain::creators::time_dependence::UniformTranslation<3>>(
          0.0, expiration_time, std::array<double, 3>({{0.01, 0.02, 0.03}})));
  Domain<3> domain = domain_creator.create_domain();
  const auto functions_of_time = domain_creator.functions_of_time();

  // Compute strahlkorper in the inertial frame.
  const double time = 0.5;
  Strahlkorper<Frame::Inertial> strahlkorper_inertial{};
  strahlkorper_in_different_frame(make_not_null(&strahlkorper_inertial),
                                  strahlkorper_grid, domain, functions_of_time,
                                  time);

  // Now compare.
  const Strahlkorper<Frame::Inertial> strahlkorper_expected(
      l_max, 2.0,
      {{strahlkorper_grid_center[0] + 0.005, strahlkorper_grid_center[1] + 0.01,
        strahlkorper_grid_center[2] + 0.015}});
  CHECK_ITERABLE_APPROX(strahlkorper_expected.physical_center(),
                        strahlkorper_inertial.physical_center());
  CHECK_ITERABLE_APPROX(strahlkorper_expected.coefficients(),
                        strahlkorper_inertial.coefficients());
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperInDifferentFrame",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  test_strahlkorper_in_different_frame();
}

}  // namespace
