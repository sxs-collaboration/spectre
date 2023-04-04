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
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "Utilities/Gsl.hpp"

namespace {

template<bool IsTimeDependent, typename SrcFrame>
void test_strahlkorper_coords_in_different_frame() {
  const size_t grid_points_each_dimension = 5;

  // Set up a Strahlkorper corresponding to a Schwarzschild hole of
  // mass 1, in the source frame.
  // Center the Strahlkorper at (0.03,0.02,0.01) so that we test a
  // nonzero center.
  const std::array<double, 3> strahlkorper_src_center = {0.03, 0.02, 0.01};
  const size_t l_max = 8;
  const Strahlkorper<SrcFrame> strahlkorper_src(l_max, 2.0,
                                                strahlkorper_src_center);

  // Create a Domain.
  // We choose a spherical shell domain extending from radius 1.9M to
  // 2.9M, so that the Strahlkorper is inside the domain. It gives a
  // narrow domain so that we don't need a large number of grid points
  // to resolve the horizon (which would make the test slower).
  std::vector<double> radial_partitioning{};
  std::vector<domain::CoordinateMaps::Distribution> radial_distribution{
      domain::CoordinateMaps::Distribution::Linear};

  std::unique_ptr<DomainCreator<3>> domain_creator;
  if constexpr (IsTimeDependent) {
    // In computing the time_dependence, make sure that src-to-inertial
    // velocity is (0.01,0.02,0.03) to agree with the analytic checks
    // below.  If src is distorted frame, then grid-to-distorted
    // velocity doesn't matter for the value of the check (but does matter
    // in terms of which points are in which blocks).
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence;
    if constexpr (std::is_same_v<SrcFrame, ::Frame::Grid>) {
      time_dependence = std::make_unique<
          domain::creators::time_dependence::UniformTranslation<3>>(
          0.0, std::array<double, 3>({{0.01, 0.02, 0.03}}));
    } else {
      static_assert(std::is_same_v<SrcFrame, ::Frame::Distorted>,
                    "Src frame must be Distorted if it is not Grid");
      time_dependence = std::make_unique<
          domain::creators::time_dependence::UniformTranslation<3>>(
          0.0, std::array<double, 3>({{-0.02, -0.01, -0.01}}),
          std::array<double, 3>({{0.01, 0.02, 0.03}}));
    }
    domain_creator = std::make_unique<domain::creators::Sphere>(
        1.9, 2.9, domain::creators::Sphere::Excision{}, 1_st,
        grid_points_each_dimension, false, std::nullopt, radial_partitioning,
        radial_distribution, ShellWedges::All, std::move(time_dependence));
  } else {
    domain_creator = std::make_unique<domain::creators::Sphere>(
        1.9, 2.9, domain::creators::Sphere::Excision{}, 1_st,
        grid_points_each_dimension, false, std::nullopt, radial_partitioning,
        radial_distribution, ShellWedges::All);
  }
  Domain<3> domain = domain_creator->create_domain();
  const auto functions_of_time = domain_creator->functions_of_time();

  // Compute strahlkorper coords in the inertial frame.
  const double time = 0.5;
  tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords{};

  strahlkorper_coords_in_different_frame(make_not_null(&inertial_coords),
                                         strahlkorper_src, domain,
                                         functions_of_time, time);

  // Now compare with expected result, which is the src-frame coords of
  // the Strahlkorper translated by (0.005,0.01,0.015).
  const auto src_coords = StrahlkorperFunctions::cartesian_coords(
      strahlkorper_src, StrahlkorperFunctions::radius(strahlkorper_src),
      StrahlkorperFunctions::rhat(
          StrahlkorperFunctions::theta_phi(strahlkorper_src)));
  if constexpr (IsTimeDependent) {
    CHECK_ITERABLE_APPROX(get<0>(src_coords) + 0.005, get<0>(inertial_coords));
    CHECK_ITERABLE_APPROX(get<1>(src_coords) + 0.01, get<1>(inertial_coords));
    CHECK_ITERABLE_APPROX(get<2>(src_coords) + 0.015, get<2>(inertial_coords));
  } else {
    CHECK_ITERABLE_APPROX(get<0>(src_coords), get<0>(inertial_coords));
    CHECK_ITERABLE_APPROX(get<1>(src_coords), get<1>(inertial_coords));
    CHECK_ITERABLE_APPROX(get<2>(src_coords), get<2>(inertial_coords));
  }
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperCoordsInDifferentFrame",
                  "[Unit]") {
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  test_strahlkorper_coords_in_different_frame<true, Frame::Grid>();
  test_strahlkorper_coords_in_different_frame<true, Frame::Distorted>();
  test_strahlkorper_coords_in_different_frame<false, Frame::Distorted>();
}

}  // namespace
