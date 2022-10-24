// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/StrahlkorperTestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/YlmTestFunctions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"

namespace Frame {
struct Inertial;
}  // namespace Frame

namespace {
void test_radius_and_derivs() {
  const double y11_amplitude = 1.0;
  const double radius = 2.0;
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const auto strahlkorper =
      create_strahlkorper_y11(y11_amplitude, radius, center);

  // Now construct a Y00 + Im(Y11) surface by hand.
  const auto& theta_points = strahlkorper.ylm_spherepack().theta_points();
  const auto& phi_points = strahlkorper.ylm_spherepack().phi_points();
  const auto n_pts = theta_points.size() * phi_points.size();
  YlmTestFunctions::Y11 y_11;
  DataVector expected_radius{n_pts};
  y_11.func(&expected_radius, 1, 0, theta_points, phi_points);
  for (size_t s = 0; s < n_pts; ++s) {
    expected_radius[s] *= y11_amplitude;
    expected_radius[s] += radius;
  }

  const DataVector strahlkorper_radius{
      get(StrahlkorperFunctions::radius(strahlkorper))};
  CHECK_ITERABLE_APPROX(strahlkorper_radius, expected_radius);

   // Test derivative of radius
  const auto theta_phi = strahlkorper.ylm_spherepack().theta_phi_points();
  tnsr::i<DataVector, 3> expected_dx_radius(n_pts);
  for (size_t s = 0; s < n_pts; ++s) {
    // Analytic solution Mark Scheel computed in Mathematica
    const double theta = theta_phi[0][s];
    const double phi = theta_phi[1][s];
    const double r = expected_radius[s];
    const double sin_phi = sin(phi);
    const double cos_phi = cos(phi);
    const double sin_theta = sin(theta);
    const double cos_theta = cos(theta);
    const double amp = -sqrt(3.0 / 8.0 / M_PI) * y11_amplitude;

    expected_dx_radius.get(0)[s] = -((cos_phi * sin_phi) / r) +
                                   (cos_phi * square(cos_theta) * sin_phi) / r;
    expected_dx_radius.get(1)[s] =
        square(cos_phi) / r + (square(cos_theta) * square(sin_phi)) / r;
    expected_dx_radius.get(2)[s] = -((cos_theta * sin_phi * sin_theta) / r);
    for (auto& a : expected_dx_radius) {
      a[s] *= amp;
    }
  }
  CHECK_ITERABLE_APPROX(
      expected_dx_radius,
      StrahlkorperFunctions::cartesian_derivs_of_scalar(
          StrahlkorperFunctions::radius(strahlkorper), strahlkorper,
          StrahlkorperFunctions::radius(strahlkorper),
          StrahlkorperFunctions::inv_jacobian(
              StrahlkorperFunctions::theta_phi(strahlkorper))));

  // Test second derivatives of radius
  tnsr::ii<DataVector, 3> expected_d2x_radius(n_pts);
  for (size_t s = 0; s < n_pts; ++s) {
    // Messy analytic solution Mark Scheel computed in Mathematica
    const double theta = theta_phi[0][s];
    const double phi = theta_phi[1][s];
    const double r = expected_radius[s];
    const double sin_theta = sin(theta);
    const double cos_theta = cos(theta);
    const double sin_phi = sin(phi);
    const double cos_phi = cos(phi);
    const double cos_2_theta = cos(2. * theta);
    const double amp = -sqrt(3. / 8. / M_PI) * y11_amplitude;
    expected_d2x_radius.get(0, 0)[s] =
        (9. * sin(3. * phi) * sin_theta -
         sin_phi * (7. * sin_theta + 12. * square(cos_phi) * sin(3. * theta))) /
        (16. * square(r));
    expected_d2x_radius.get(0, 1)[s] =
        -((cos_phi * (square(cos_phi) +
                      ((-1. + 3. * cos_2_theta) * square(sin_phi)) / 2.) *
           sin_theta) /
          square(r));
    expected_d2x_radius.get(0, 2)[s] =
        (3. * cos_phi * cos_theta * sin_phi * square(sin_theta)) / square(r);
    expected_d2x_radius.get(1, 1)[s] =
        (-3. * (7. * sin_phi * sin_theta + 3. * sin(3. * phi) * sin_theta +
                4. * pow<3>(sin_phi) * sin(3. * theta))) /
        (16. * square(r));
    expected_d2x_radius.get(1, 2)[s] =
        -((cos_theta * (square(cos_phi) +
                        ((-1. + 3. * cos_2_theta) * square(sin_phi)) / 2.)) /
          square(r));
    expected_d2x_radius.get(2, 2)[s] =
        ((1. + 3. * cos_2_theta) * sin_phi * sin_theta) / (2. * square(r));
    for (auto& a : expected_d2x_radius) {
      a[s] *= amp;
    }
  }
  CHECK_ITERABLE_APPROX(
      expected_d2x_radius,
      StrahlkorperFunctions::cartesian_second_derivs_of_scalar(
          StrahlkorperFunctions::radius(strahlkorper), strahlkorper,
          StrahlkorperFunctions::radius(strahlkorper),
          StrahlkorperFunctions::inv_jacobian(
              StrahlkorperFunctions::theta_phi(strahlkorper)),
          StrahlkorperFunctions::inv_hessian(
              StrahlkorperFunctions::theta_phi(strahlkorper))));

  // Test laplacian
  DataVector expected_laplacian(n_pts);
  for (size_t s = 0; s < n_pts; ++s) {
    y_11.scalar_laplacian(&expected_laplacian, 1, s, {theta_phi[0][s]},
                          {theta_phi[1][s]});
    expected_laplacian[s] *= y11_amplitude;
  }
  CHECK_ITERABLE_APPROX(
      expected_laplacian,
      get(StrahlkorperFunctions::laplacian_of_scalar(
          StrahlkorperFunctions::radius(strahlkorper), strahlkorper,
          StrahlkorperFunctions::theta_phi(strahlkorper))));
}

void test_theta_phi() {
  const double y11_amplitude = 1.0;
  const double radius = 2.0;
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const auto strahlkorper =
      create_strahlkorper_y11(y11_amplitude, radius, center);

  const auto expected_theta_phi =
      strahlkorper.ylm_spherepack().theta_phi_points();
  const auto theta_phi = StrahlkorperFunctions::theta_phi(strahlkorper);
  CHECK_ITERABLE_APPROX(get<0>(theta_phi), expected_theta_phi[0]);
  CHECK_ITERABLE_APPROX(get<1>(theta_phi), expected_theta_phi[1]);
}

void test_rhat_jacobian_hessian() {
  const double y11_amplitude = 1.0;
  const double radius = 2.0;
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const auto strahlkorper =
      create_strahlkorper_y11(y11_amplitude, radius, center);

  const auto theta_phi = strahlkorper.ylm_spherepack().theta_phi_points();
  const auto& theta = theta_phi[0];
  const auto& phi = theta_phi[1];
  const DataVector cos_phi = cos(phi);
  const DataVector sin_phi = sin(phi);
  const DataVector cos_theta = cos(theta);
  const DataVector sin_theta = sin(theta);

  tnsr::i<DataVector, 3> expected_rhat(theta.size());
  get<0>(expected_rhat) = sin_theta*cos_phi;
  get<1>(expected_rhat) = sin_theta*sin_phi;
  get<2>(expected_rhat) = cos_theta;
  CHECK_ITERABLE_APPROX(expected_rhat,
                        StrahlkorperFunctions::rhat(
                            StrahlkorperFunctions::theta_phi(strahlkorper)));

  StrahlkorperTags::aliases::Jacobian<Frame::Inertial> expected_jac(
      theta.size());
  get<0, 0>(expected_jac) = cos_theta * cos_phi;  // 1/R dx/dth
  get<1, 0>(expected_jac) = cos_theta * sin_phi;  // 1/R dy/dth
  get<2, 0>(expected_jac) = -sin(theta);          // 1/R dz/dth
  get<0, 1>(expected_jac) = -sin_phi;             // 1/(R sin(th)) dx/dph
  get<1, 1>(expected_jac) = cos_phi;              // 1/(R sin(th)) dy/dph
  get<2, 1>(expected_jac) = 0.0;                  // 1/(R sin(th)) dz/dph
  CHECK_ITERABLE_APPROX(expected_jac,
                        StrahlkorperFunctions::jacobian(
                            StrahlkorperFunctions::theta_phi(strahlkorper)));

  StrahlkorperTags::aliases::InvJacobian<Frame::Inertial> expected_inv_jac(
      theta.size());
  get<0, 0>(expected_inv_jac) = cos_theta * cos_phi;  // R dth/dx
  get<0, 1>(expected_inv_jac) = cos_theta * sin_phi;  // R dth/dy
  get<0, 2>(expected_inv_jac) = -sin(theta);          // R dth/dz
  get<1, 0>(expected_inv_jac) = -sin_phi;             // R sin(th) dph/dx
  get<1, 1>(expected_inv_jac) = cos_phi;              // R sin(th) dph/dy
  get<1, 2>(expected_inv_jac) = 0.0;                  // R sin(th) dph/dz
  CHECK_ITERABLE_APPROX(expected_inv_jac,
                        StrahlkorperFunctions::inv_jacobian(
                            StrahlkorperFunctions::theta_phi(strahlkorper)));

  StrahlkorperTags::aliases::InvHessian<Frame::Inertial> expected_inv_hess(
      theta.size());
  // Note that here expected_inv_hess is computed in a much more
  // straightforward way than in StrahlkorperFunctions.cpp, where it
  // is computed in a way that avoids allocations.
  const DataVector sin_sq_theta = square(sin_theta);
  const DataVector cos_sq_theta = square(cos_theta);
  const DataVector sin_theta_cos_theta = sin_theta * cos_theta;
  const DataVector sin_sq_phi = square(sin_phi);
  const DataVector cos_sq_phi = square(cos_phi);
  const DataVector sin_phi_cos_phi = sin_phi * cos_phi;
  const DataVector csc_theta = 1.0 / sin_theta;
  const DataVector f1 = 1.0 + 2.0 * sin_sq_theta;
  const DataVector cot_theta = cos_theta * csc_theta;
  // R^2 d^2 th/(dx^2)
  get<0, 0, 0>(expected_inv_hess) = cot_theta * (1.0 - cos_sq_phi * f1);
  // R^2 d^2 th/(dxdy)
  get<0, 0, 1>(expected_inv_hess) = -cot_theta * sin_phi_cos_phi * f1;
  // R^2 d^2 th/(dxdz)
  get<0, 0, 2>(expected_inv_hess) = (sin_sq_theta - cos_sq_theta) * cos_phi;
  // R^2 d^2 th/(dydx)
  get<0, 1, 0>(expected_inv_hess) = get<0, 0, 1>(expected_inv_hess);
  // R^2 d^2 th/(dy^2)
  get<0, 1, 1>(expected_inv_hess) = cot_theta * (1.0 - sin_sq_phi * f1);
  // R^2 d^2 th/(dydz)
  get<0, 1, 2>(expected_inv_hess) = (sin_sq_theta - cos_sq_theta) * sin_phi;
  // R^2 d^2 th/(dzdx)
  get<0, 2, 0>(expected_inv_hess) = get<0, 0, 2>(expected_inv_hess);
  // R^2 d^2 th/(dzdy)
  get<0, 2, 1>(expected_inv_hess) = get<0, 1, 2>(expected_inv_hess);
  // R^2 d^2 th/(dz^2)
  get<0, 2, 2>(expected_inv_hess) = 2.0 * sin_theta_cos_theta;
  // R^2 d/dx (sin(th) dph/dx)
  get<1, 0, 0>(expected_inv_hess) =
      sin_phi_cos_phi * (1.0 + sin_sq_theta) * csc_theta;
  // R^2 d/dx (sin(th) dph/dy)
  get<1, 0, 1>(expected_inv_hess) =
      (sin_sq_phi - sin_sq_theta * cos_sq_phi) * csc_theta;
  // R^2 d/dx (sin(th) dph/dz)
  get<1, 0, 2>(expected_inv_hess) = 0.0;
  // R^2 d/dy (sin(th) dph/dx)
  get<1, 1, 0>(expected_inv_hess) =
      (sin_sq_theta * sin_sq_phi - cos_sq_phi) * csc_theta;
  // R^2 d/dy (sin(th) dph/dy)
  get<1, 1, 1>(expected_inv_hess) = -get<1, 0, 0>(expected_inv_hess);
  // R^2 d/dy (sin(th) dph/dz)
  get<1, 1, 2>(expected_inv_hess) = 0.0;
  // R^2 d/dz (sin(th) dph/dx)
  get<1, 2, 0>(expected_inv_hess) = cos_theta * sin_phi;
  // R^2 d/dz (sin(th) dph/dy)
  get<1, 2, 1>(expected_inv_hess) = -cos_theta * cos_phi;
  // R^2 d/dz (sin(th) dph/dz)
  get<1, 2, 2>(expected_inv_hess) = 0.0;
  CHECK_ITERABLE_APPROX(expected_inv_hess,
                        StrahlkorperFunctions::inv_hessian(
                            StrahlkorperFunctions::theta_phi(strahlkorper)));
}

void test_cartesian_coords() {
  const double y11_amplitude = 1.0;
  const double radius = 2.0;
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const auto strahlkorper =
      create_strahlkorper_y11(y11_amplitude, radius, center);

  const auto theta_phi = strahlkorper.ylm_spherepack().theta_phi_points();
  const auto& theta = theta_phi[0];
  const auto& phi = theta_phi[1];
  const DataVector cos_phi = cos(phi);
  const DataVector sin_phi = sin(phi);
  const DataVector cos_theta = cos(theta);
  const DataVector sin_theta = sin(theta);
  const auto n_pts = theta_phi[0].size();

  const double amp = -sqrt(3.0 / 8.0 / M_PI) * y11_amplitude;
  tnsr::I<DataVector, 3> expected_cartesian_coords(n_pts);
  const DataVector temp = radius + amp * sin_phi * sin_theta;
  expected_cartesian_coords.get(0) = cos_phi * sin_theta * temp + center[0];
  expected_cartesian_coords.get(1) = sin_phi * sin_theta * temp + center[1];
  expected_cartesian_coords.get(2) = cos_theta * temp + center[2];

  CHECK_ITERABLE_APPROX(
      expected_cartesian_coords,
      StrahlkorperFunctions::cartesian_coords<Frame::Inertial>(
          strahlkorper, StrahlkorperFunctions::radius(strahlkorper),
          StrahlkorperFunctions::rhat(
              StrahlkorperFunctions::theta_phi(strahlkorper))));
}

void test_normals() {
  const double y11_amplitude = 1.0;
  const double radius = 2.0;
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const auto strahlkorper =
      create_strahlkorper_y11(y11_amplitude, radius, center);

  const auto theta_phi = strahlkorper.ylm_spherepack().theta_phi_points();
  const auto n_pts = theta_phi[0].size();

  // Test surface_tangents
  StrahlkorperTags::aliases ::Jacobian<Frame::Inertial>
      expected_surface_tangents(n_pts);
  const double amp = -sqrt(3.0 / 8.0 / M_PI) * y11_amplitude;

  const auto& theta = theta_phi[0];
  const auto& phi = theta_phi[1];
  const DataVector cos_phi = cos(phi);
  const DataVector sin_phi = sin(phi);
  const DataVector cos_theta = cos(theta);
  const DataVector sin_theta = sin(theta);

  expected_surface_tangents.get(0, 0) =
      cos_phi * cos_theta * (radius + 2. * amp * sin_phi * sin_theta);
  expected_surface_tangents.get(1, 0) =
      cos_theta * sin_phi * (radius + 2. * amp * sin_phi * sin_theta);
  expected_surface_tangents.get(2, 0) =
      amp * square(cos_theta) * sin_phi -
      sin_theta * (radius + amp * sin_phi * sin_theta);
  expected_surface_tangents.get(0, 1) =
      -radius * sin_phi + amp * sin_theta * (2. * square(cos_phi) - 1);
  expected_surface_tangents.get(1, 1) =
      cos_phi * (radius + 2. * amp * sin_phi * sin_theta);
  expected_surface_tangents.get(2, 1) = amp * cos_phi * cos_theta;
  CHECK_ITERABLE_APPROX(
      expected_surface_tangents,
      StrahlkorperFunctions::tangents(
          strahlkorper, StrahlkorperFunctions::radius(strahlkorper),
          StrahlkorperFunctions::rhat(
              StrahlkorperFunctions::theta_phi(strahlkorper)),
          StrahlkorperFunctions::jacobian(
              StrahlkorperFunctions::theta_phi(strahlkorper))));

  // Test normal_one_form
  tnsr::i<DataVector, 3> expected_normal_one_form(n_pts);
  {
    const auto r = get(StrahlkorperFunctions::radius(strahlkorper));
    const DataVector one_over_r = 1.0 / r;
    const DataVector temp = 1.0 + one_over_r * amp * sin_phi * sin_theta;
    expected_normal_one_form.get(0) = cos_phi * sin_theta * temp;
    expected_normal_one_form.get(1) =
        sin_phi * sin_theta * temp - amp * one_over_r;
    expected_normal_one_form.get(2) = cos_theta * temp;
  }
  CHECK_ITERABLE_APPROX(
      expected_normal_one_form,
      StrahlkorperFunctions::normal_one_form(
          StrahlkorperFunctions::cartesian_derivs_of_scalar(
              StrahlkorperFunctions::radius(strahlkorper), strahlkorper,
              StrahlkorperFunctions::radius(strahlkorper),
              StrahlkorperFunctions::inv_jacobian(
                  StrahlkorperFunctions::theta_phi(strahlkorper))),
          StrahlkorperFunctions::rhat(
              StrahlkorperFunctions::theta_phi(strahlkorper))));
}

void test_fit_ylm_coeffs_same() {
  const double l_max = 2;
  const double m_max = l_max;
  const double radius = 2.0;
  const std::array<double, 3> center = {{0.0, 0.0, 0.0}};
  const double y00 = radius * sqrt(8.0);
  const auto strahlkorper0 =
      Strahlkorper<Frame::Inertial>(l_max, m_max, radius, center);
  const auto strahlkorper1 = Strahlkorper<Frame::Inertial>(
      {y00, 0.0, 1.0, -1.0, -5.0, 12.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.01},
      strahlkorper0);
  const auto strahlkorper2 = Strahlkorper<Frame::Inertial>(
      {y00, 0.0, 2.0, -3.0, -3.0, 17.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.04},
      strahlkorper0);
  const auto strahlkorper3 = Strahlkorper<Frame::Inertial>(
      {y00, 0.0, 3.0, -7.0, -1.0, 17.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.09},
      strahlkorper0);
  const auto strahlkorper4 = Strahlkorper<Frame::Inertial>(
      {y00, 0.0, 4.0, -13.0, 1.0, 12.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.16},
      strahlkorper0);

  const DataVector times{0.0, 0.1, 0.2, 0.3, 0.4};
  std::vector<Strahlkorper<Frame::Inertial>> strahlkorpers{
      strahlkorper0, strahlkorper1, strahlkorper2, strahlkorper3,
      strahlkorper4};
  std::vector<std::array<double, 4>> result =
      StrahlkorperFunctions::fit_ylm_coeffs<Frame::Inertial>(times,
                                                             strahlkorpers);
  const std::vector<std::array<double, 4>> expected_result = {
      {y00, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 10.0, 0.0, 0.0},
      {-1.0 / 70.0, -4.88095238095232631, -35.71428571428607768, -250.0 / 3.0},
      {-0.1, -505.0 / 6.0, 450.0, -1750.0 / 3.0},
      {0.02857142857142869, 154.76190476190464551, -378.57142857142787307,
       500.0 / 3.0},
      {0.0142857142857146, 14.88095238095235473, -64.28571428571413549,
       250.0 / 3.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 1.0, 0.0}};
  Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(result, expected_result, custom_approx);
}

void test_fit_ylm_coeffs_diff() {
  const double l_max = 3;
  const double m_max = 2;
  const double radius = 2.0;
  const std::array<double, 3> center = {{0.0, 0.0, 0.0}};
  const double y00 = radius * sqrt(8.0);
  const auto strahlkorper0 =
      Strahlkorper<Frame::Inertial>(l_max, m_max, radius, center);
  const auto strahlkorper1 = Strahlkorper<Frame::Inertial>(
      {y00, 0.0, 1.0, -1.0, -5.0, 12.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.01},
      strahlkorper0);
  const auto strahlkorper2 = Strahlkorper<Frame::Inertial>(
      {y00, 0.0, 2.0, -3.0, -3.0, 17.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.04},
      strahlkorper0);
  const auto strahlkorper3 = Strahlkorper<Frame::Inertial>(
      {y00, 0.0, 3.0, -7.0, -1.0, 17.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.09},
      strahlkorper0);
  const auto strahlkorper4 = Strahlkorper<Frame::Inertial>(
      {y00, 0.0, 4.0, -13.0, 1.0, 12.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 0.0, 0.0, 0.0,   0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.16},
      strahlkorper0);

  const DataVector times{0.0, 0.1, 0.2, 0.3, 0.4};
  std::vector<Strahlkorper<Frame::Inertial>> strahlkorpers{
      strahlkorper0, strahlkorper1, strahlkorper2, strahlkorper3,
      strahlkorper4};
  std::vector<std::array<double, 4>> result =
      StrahlkorperFunctions::fit_ylm_coeffs<Frame::Inertial>(times,
                                                             strahlkorpers);
  const std::vector<std::array<double, 4>> expected_result = {
      {y00, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 10.0, 0.0, 0.0},
      {-1.0 / 70.0, -4.88095238095232631, -35.71428571428607768, -250.0 / 3.0},
      {-0.1, -505.0 / 6.0, 450.0, -1750.0 / 3.0},
      {0.02857142857142869, 154.76190476190464551, -378.57142857142787307,
       500.0 / 3.0},
      {0.0142857142857146, 14.88095238095235473, -64.28571428571413549,
       250.0 / 3.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 1.0, 0.0}};
  Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(result, expected_result, custom_approx);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperFunctions",
                  "[ApparentHorizons][Unit]") {
  test_theta_phi();
  test_rhat_jacobian_hessian();
  test_cartesian_coords();
  test_radius_and_derivs();
  test_normals();
  test_fit_ylm_coeffs_same();
  test_fit_ylm_coeffs_diff();
}
