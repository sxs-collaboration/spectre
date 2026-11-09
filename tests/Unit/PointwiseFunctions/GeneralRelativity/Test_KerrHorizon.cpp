// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/KerrHorizon.hpp"
#include "PointwiseFunctions/SpecialRelativity/LorentzBoostMatrix.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace gr::Solutions {
namespace {

// This wraps the kerr_horizon_radius function with
// types that CheckWithRandomValues can understand.
// We use magnitude and angle of dimensionless spin so that
// we know their bounds.
template <typename DataType>
Scalar<DataType> wrap_kerr_horizon_radius(const Scalar<DataType>& theta,
                                          const Scalar<DataType>& phi,
                                          const double mass,
                                          const double dimless_spin_magnitude,
                                          const double dimless_spin_theta,
                                          const double dimless_spin_phi) {
  return kerr_horizon_radius<DataType>(
      {{get(theta), get(phi)}}, mass,
      {{dimless_spin_magnitude * sin(dimless_spin_theta) *
            cos(dimless_spin_phi),
        dimless_spin_magnitude * sin(dimless_spin_theta) *
            sin(dimless_spin_phi),
        dimless_spin_magnitude * cos(dimless_spin_theta)}});
}

template <typename DataType>
void test_kerr_horizon(const DataType& used_for_size) {
  pypp::check_with_random_values<6>(&wrap_kerr_horizon_radius<DataType>,
                                    "KerrHorizon", "kerr_horizon_radius",
                                    {{{0.0, M_PI},
                                      {0.0, 2.0 * M_PI},
                                      {1.0, 2.0},
                                      {0.0, 1.0},
                                      {0.0, M_PI},
                                      {0.0, 2.0 * M_PI}}},
                                    used_for_size);
}

// The spin is given in the rest frame of the black hole, while the angles and
// the returned radius are in the boosted frame (at boosted time t = 0). Map the
// boosted surface back to the rest frame and check that it is a surface of
// constant Boyer-Lindquist radius there. This works for arbitrary relative
// orientations of the spin and the boost. A zero boost is included so the
// unboosted surface is checked against the same relation.
void test_boosted_surface_has_constant_bl_radius() {
  MAKE_GENERATOR(gen);
  const double mass = 1.3;
  const std::array<double, 3> dimless_spin{{0.2, -0.4, 0.5}};
  const std::array<double, 3> spin = mass * dimless_spin;
  const double spin_squared = dot(spin, spin);
  const std::array<double, 3> nonzero_boost_velocity{{0.3, 0.5, -0.4}};
  const double r_plus =
      mass * (1.0 + sqrt(1.0 - dot(dimless_spin, dimless_spin)));

  // Special directions, in addition to the random ones: the coordinate poles
  // and both ends of the spin and boost axes
  const auto angles_of = [](const std::array<double, 3>& direction) {
    return std::array{acos(direction[2] / magnitude(direction)),
                      atan2(direction[1], direction[0])};
  };
  const std::array<std::array<double, 2>, 6> special_theta_phi{
      {{{0.0, 0.0}},
       {{M_PI, 0.0}},
       angles_of(dimless_spin),
       angles_of(-1.0 * dimless_spin),
       angles_of(nonzero_boost_velocity),
       angles_of(-1.0 * nonzero_boost_velocity)}};

  const size_t num_random_points = 10;
  const size_t num_points = num_random_points + special_theta_phi.size();
  const DataVector used_for_size(num_points);
  // Draw cos(theta) uniformly so that the points cover the sphere uniformly
  std::uniform_real_distribution<double> cos_theta_dist(-1.0, 1.0);
  std::uniform_real_distribution<double> phi_dist(0.0, 2.0 * M_PI);
  DataVector theta = acos(make_with_random_values<DataVector>(
      make_not_null(&gen), make_not_null(&cos_theta_dist), used_for_size));
  auto phi = make_with_random_values<DataVector>(
      make_not_null(&gen), make_not_null(&phi_dist), used_for_size);
  for (size_t i = 0; i < special_theta_phi.size(); ++i) {
    theta[num_random_points + i] = gsl::at(special_theta_phi, i)[0];
    phi[num_random_points + i] = gsl::at(special_theta_phi, i)[1];
  }
  const std::array<DataVector, 2> theta_phi{{theta, phi}};

  // For zero velocity the Lorentz transformation is the identity
  for (const auto& boost_velocity :
       std::array{std::array{0.0, 0.0, 0.0}, nonzero_boost_velocity}) {
    CAPTURE(boost_velocity);
    for (const double bl_radius : std::array{r_plus, 3.7}) {
      CAPTURE(bl_radius);
      const DataVector radius = get(kerr_schild_radius_from_boyer_lindquist(
          bl_radius, theta_phi, mass, dimless_spin, boost_velocity));

      // Boosted-frame Cartesian coordinates of the surface
      tnsr::I<DataVector, 3> x{num_points};
      get<0>(x) = radius * sin(theta) * cos(phi);
      get<1>(x) = radius * sin(theta) * sin(phi);
      get<2>(x) = radius * cos(theta);

      // Lorentz transform the surface points (at boosted time t = 0) to the
      // rest frame of the black hole. This uses the same convention as the
      // boosted KerrSchild solution.
      tnsr::I<DataVector, 3> x_rest{num_points};
      sr::lorentz_boost(make_not_null(&x_rest), x, 0.0, boost_velocity);

      // In the rest frame, a surface of constant Boyer-Lindquist radius r_BL
      // is an oblate spheroid in Kerr-Schild coordinates. With
      // z = a_hat . x_rest the coordinate along the spin axis and
      // rho^2 = |x_rest|^2 - z^2 the squared distance from that axis:
      //   rho^2 / (r_BL^2 + a^2) + z^2 / r_BL^2 = 1.
      // For spin along z this is the familiar
      //   (x^2 + y^2) / (r_BL^2 + a^2) + z^2 / r_BL^2 = 1.
      DataVector rest_radius_squared = square(get<0>(x_rest));
      DataVector spin_dot_x_rest = spin[0] * get<0>(x_rest);
      for (size_t i = 1; i < 3; ++i) {
        rest_radius_squared += square(x_rest.get(i));
        spin_dot_x_rest += gsl::at(spin, i) * x_rest.get(i);
      }
      // z^2 = (a . x_rest)^2 / a^2 (the spin is nonzero in this test)
      const DataVector z_squared = square(spin_dot_x_rest) / spin_squared;
      const DataVector rho_squared = rest_radius_squared - z_squared;
      const DataVector ellipsoid =
          rho_squared / (square(bl_radius) + spin_squared) +
          z_squared / square(bl_radius);
      CHECK_ITERABLE_APPROX(ellipsoid, DataVector(num_points, 1.0));
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.KerrHorizon",
                  "[PointwiseFunctions][Unit]") {
  const pypp::SetupLocalPythonEnvironment local_python_env(
      "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/");
  const DataVector dv(5);
  test_kerr_horizon(dv);
  test_kerr_horizon(0.0);
  test_boosted_surface_has_constant_bl_radius();

  // Test for Schwarzschild (mass=2) at randomly-chosen point
  CHECK(approx(4.0) == get(kerr_horizon_radius<double>({{1.12345, 2.2222}}, 2.0,
                                                       {{0.0, 0.0, 0.0}})));

  // Test for Kerr (mass=2) along pole, for spin not in z direction.
  CHECK(approx(2.0 * (1.0 + sqrt(0.86))) ==
        get(kerr_horizon_radius<double>(
            {{acos(0.3 / sqrt(0.14)), atan2(0.2, 0.1)}}, 2.0,
            {{0.1, 0.2, 0.3}})));

  // Test for Kerr (mass=2) along equator, for spin not in z direction.
  // Angles and radius worked out by hand.
  CHECK(approx(sqrt(square(2.0 * (1.0 + sqrt(0.86))) + (4.0 * 0.14))) ==
        get(kerr_horizon_radius<double>(
            {{acos(0.3 / sqrt(0.14)) + M_PI_2, atan2(0.2, 0.1)}}, 2.0,
            {{0.1, 0.2, 0.3}})));

  // Test for Kerr (mass=2) along pole, for extremal spin not in z direction.
  // one_minus_eps to make sure we don't get FPE in sqrt(M^2-a^2).
  const double one_minus_eps = 1.0 - std::numeric_limits<double>::min();
  const Approx numerical_approx = Approx::custom().epsilon(1.e-8).scale(1.0);
  CHECK(
      numerical_approx(2.0) ==
      get(kerr_horizon_radius<double>(
          {{acos(0.3 / sqrt(0.14)), atan2(0.2, 0.1)}}, 2.0,
          {{one_minus_eps * 0.1 / sqrt(0.14), one_minus_eps * 0.2 / sqrt(0.14),
            one_minus_eps * 0.3 / sqrt(0.14)}})));

  // Test for Kerr (mass=2) along equator,
  // for extremal spin not in z direction.
  CHECK(
      numerical_approx(sqrt(8.0)) ==
      get(kerr_horizon_radius<double>(
          {{acos(0.3 / sqrt(0.14)) + M_PI_2, atan2(0.2, 0.1)}}, 2.0,
          {{one_minus_eps * 0.1 / sqrt(0.14), one_minus_eps * 0.2 / sqrt(0.14),
            one_minus_eps * 0.3 / sqrt(0.14)}})));

  {
    const double mass = 0.45;
    const std::array<double, 3> dimless_spin{{0., 0.3, 0.5}};
    const std::array<double, 3> boost_velocity{{0., 0., 0.}};
    const double r_plus =
        mass * (1. + sqrt(1. - dot(dimless_spin, dimless_spin)));
    const DataVector theta{0., M_PI_4 * 0.34, M_PI_4, M_PI_2 * 0.67, M_PI_2};
    const DataVector phi{M_PI_2 * 0.4, M_PI * 0.55, M_PI_4, M_PI_4 * 1.2, M_PI};
    const std::array<DataVector, 2> theta_phi{{theta, phi}};
    CHECK_ITERABLE_APPROX(
        kerr_schild_radius_from_boyer_lindquist(r_plus, theta_phi, mass,
                                                dimless_spin, boost_velocity),
        kerr_horizon_radius(theta_phi, mass, dimless_spin));
  }
}
}  // namespace gr::Solutions
