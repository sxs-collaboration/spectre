// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "ApparentHorizons/StrahlkorperCoordsInDifferentFrame.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Shell.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "Utilities/Gsl.hpp"

namespace {

void test_strahlkorper_coords_in_different_frame() {
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
  domain::creators::Shell domain_creator(
      1.9, 2.9, 1,
      std::array<size_t, 2>{grid_points_each_dimension,
                            grid_points_each_dimension},
      false, {{1.0, 2}}, radial_partitioning, radial_distribution,
      ShellWedges::All,
      std::make_unique<
          domain::creators::time_dependence::UniformTranslation<3>>(
          0.0, std::array<double, 3>({{0.01, 0.02, 0.03}})));
  Domain<3> domain = domain_creator.create_domain();
  const auto functions_of_time = domain_creator.functions_of_time();

  // Compute strahlkorper coords in the inertial frame.
  const double time = 0.5;
  tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords{};

  strahlkorper_coords_in_different_frame(make_not_null(&inertial_coords),
                                         strahlkorper_grid, domain,
                                         functions_of_time, time);

  // Now compare with expected result, which is the grid-frame coords of
  // the Strahlkorper translated by (0.005,0.01,0.015).
  const auto grid_coords = StrahlkorperFunctions::cartesian_coords(
      strahlkorper_grid, StrahlkorperFunctions::radius(strahlkorper_grid),
      StrahlkorperFunctions::rhat(
          StrahlkorperFunctions::theta_phi(strahlkorper_grid)));
  CHECK_ITERABLE_APPROX(get<0>(grid_coords) + 0.005, get<0>(inertial_coords));
  CHECK_ITERABLE_APPROX(get<1>(grid_coords) + 0.01, get<1>(inertial_coords));
  CHECK_ITERABLE_APPROX(get<2>(grid_coords) + 0.015, get<2>(inertial_coords));
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperCoordsInDifferentFrame",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  test_strahlkorper_coords_in_different_frame();
}

}  // namespace
