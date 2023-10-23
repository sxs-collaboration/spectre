// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <random>

#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/PolarAngle.hpp"
#include "Domain/CoordinateMaps/SphericalToCartesian.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"

namespace domain {

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.SphericalShell",
                  "[Domain][Unit]") {
  MAKE_GENERATOR(generator);
  {
    INFO("Test PolarAngle map");
    const CoordinateMaps::PolarAngle map{};
    std::uniform_real_distribution<> dist(-1., 1.);
    const auto sample_points = make_with_random_values<std::array<double, 10>>(
        make_not_null(&generator), make_not_null(&dist));
    for (double scalar : sample_points) {
      const std::array<double, 1> random_point{{scalar}};
      test_coordinate_map_argument_types(map, random_point);
      test_inverse_map(map, random_point);
      test_jacobian(map, random_point);
      test_inv_jacobian(map, random_point);
    }
    const std::array<double, 1> endpoint_left{{-1.}};
    const std::array<double, 1> endpoint_right{{1.}};
    CHECK(get<0>(map(endpoint_left)) == approx(0.));
    CHECK(get<0>(map(endpoint_right)) == approx(M_PI));
    CHECK_FALSE(map != map);
    test_serialization(map);
    CHECK(not map.is_identity());
  }
  {
    INFO("Consistency with ylm::Spherepack");
    const size_t l_max = 6;
    const size_t m_max = 3;
    // Theta: Gauss-Legendre points in cos(theta)
    const auto logical_theta =
        Spectral::collocation_points<Spectral::Basis::Legendre,
                                     Spectral::Quadrature::Gauss>(l_max + 1);
    const CoordinateMaps::PolarAngle theta_map{};
    const auto theta =
        get<0>(theta_map(std::array<DataVector, 1>{{logical_theta}}));
    // Phi: equidistant points in [0, 2pi)
    DataVector logical_phi(2 * m_max + 1);
    const double spacing = 2. / logical_phi.size();
    for (size_t i = 0; i < logical_phi.size(); ++i) {
      logical_phi[i] = -1. + i * spacing;
    }
    const CoordinateMaps::Affine phi_map{-1., 1., 0., 2. * M_PI};
    const auto phi = get<0>(phi_map(std::array<DataVector, 1>{{logical_phi}}));
    // Check consistency with ylm::Spherepack
    const ylm::Spherepack ylm{l_max, m_max};
    DataVector ylm_theta(ylm.theta_points().size());
    std::copy(ylm.theta_points().begin(), ylm.theta_points().end(),
              ylm_theta.begin());
    DataVector ylm_phi(ylm.phi_points().size());
    std::copy(ylm.phi_points().begin(), ylm.phi_points().end(),
              ylm_phi.begin());
    CHECK_ITERABLE_APPROX(theta, ylm_theta);
    CHECK_ITERABLE_APPROX(phi, ylm_phi);
  }
  {
    INFO("Test SphericalToCartesian map");
    const CoordinateMaps::SphericalToCartesian<3> map{};
    std::uniform_real_distribution<> dist_r(0.5, 3.);
    std::uniform_real_distribution<> dist_theta(0., M_PI);
    std::uniform_real_distribution<> dist_phi(0., 2. * M_PI - 0.01);
    const size_t num_points = 10;
    const auto sample_r = make_with_random_values<DataVector>(
        make_not_null(&generator), make_not_null(&dist_r), num_points);
    const auto sample_theta = make_with_random_values<DataVector>(
        make_not_null(&generator), make_not_null(&dist_theta), num_points);
    const auto sample_phi = make_with_random_values<DataVector>(
        make_not_null(&generator), make_not_null(&dist_phi), num_points);
    const std::array<DataVector, 3> random_points{
        {sample_theta, sample_phi, sample_r}};
    for (size_t i = 0; i < num_points; ++i) {
      const std::array<double, 3> random_point{
          {sample_theta[i], sample_phi[i], sample_r[i]}};
      test_coordinate_map_argument_types(map, random_point);
      test_inverse_map(map, random_point);
      test_jacobian(map, random_point);
      test_inv_jacobian(map, random_point);
    }
    const std::array<double, 3> x_axis{{M_PI_2, 0., 10.}};
    CHECK_ITERABLE_APPROX(map(x_axis), (std::array<double, 3>{{10., 0., 0.}}));
    const std::array<double, 3> y_axis{{M_PI_2, M_PI_2, 10.}};
    CHECK_ITERABLE_APPROX(map(y_axis), (std::array<double, 3>{{0., 10., 0.}}));
    const std::array<double, 3> z_axis{{0., 0., 10.}};
    CHECK_ITERABLE_APPROX(map(z_axis), (std::array<double, 3>{{0., 0., 10.}}));
    CHECK_FALSE(map != map);
    test_serialization(map);
    CHECK(not map.is_identity());
  }
}

}  // namespace domain
