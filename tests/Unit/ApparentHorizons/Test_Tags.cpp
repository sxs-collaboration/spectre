// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>

#include "ApparentHorizons/SpherepackIterator.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/Tags.hpp"  // IWYU pragma: keep
#include "ApparentHorizons/YlmSpherepack.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ApparentHorizons/YlmTestFunctions.hpp"

namespace {

// Create a strahlkorper with a Im(Y11) dependence, with
// a given average radius and a given center.
auto create_strahlkorper_y11(const double y11_amplitude, const double radius,
                             const std::array<double, 3>& center) {
  const size_t l_max = 4, m_max = 4;

  Strahlkorper<Frame::Inertial> strahlkorper_sphere(l_max, m_max, radius,
                                                    center);

  auto coefs = strahlkorper_sphere.coefficients();
  SpherepackIterator it(l_max, m_max);
  // Conversion between SPHEREPACK b_lm and real valued harmonic coefficients:
  // b_lm = (-1)^{m+1} sqrt(1/2pi) d_lm
  coefs[it.set(1, -1)()] = y11_amplitude * sqrt(0.5 / M_PI);
  return Strahlkorper<Frame::Inertial>(coefs, strahlkorper_sphere);
}

void test_average_radius() {
  // Create spherical Strahlkorper
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const double r = 3.0;
  Strahlkorper<Frame::Inertial> s(4, 4, r, center);
  CHECK(s.average_radius() == approx(r));
}

void test_radius_and_derivs() {
  const double y11_amplitude = 1.0;
  const double radius = 2.0;
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const auto strahlkorper =
      create_strahlkorper_y11(y11_amplitude, radius, center);

  // Now construct a Y00 + Im(Y11) surface by hand.
  const auto& theta_phi = strahlkorper.ylm_spherepack().theta_phi_points();
  const auto& theta_points = strahlkorper.ylm_spherepack().theta_points();
  const auto& phi_points = strahlkorper.ylm_spherepack().phi_points();
  const auto n_pts = theta_phi[0].size();
  YlmTestFunctions::Y11 y_11;
  DataVector expected_radius(n_pts);
  y_11.func(&expected_radius, 1, 0, theta_points, phi_points);
  for (size_t s = 0; s < n_pts; ++s) {
    expected_radius[s] *= y11_amplitude;
    expected_radius[s] += radius;
  }

  // Create DataBox
  auto box = db::create<
      db::AddSimpleTags<StrahlkorperTags::items_tags<Frame::Inertial>>,
      db::AddComputeTags<
          StrahlkorperTags::compute_items_tags<Frame::Inertial>>>(strahlkorper);

  // Test radius
  const auto& strahlkorper_radius =
      db::get<StrahlkorperTags::Radius<Frame::Inertial>>(box);
  CHECK_ITERABLE_APPROX(strahlkorper_radius, expected_radius);

  // Test derivative of radius
  db::item_type<StrahlkorperTags::DxRadius<Frame::Inertial>> expected_dx_radius(
      n_pts);
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
  const auto& strahlkorper_dx_radius =
      db::get<StrahlkorperTags::DxRadius<Frame::Inertial>>(box);
  CHECK_ITERABLE_APPROX(strahlkorper_dx_radius, expected_dx_radius);

  // Test second derivatives.
  db::item_type<StrahlkorperTags::D2xRadius<Frame::Inertial>>
      expected_d2x_radius(n_pts);
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
  const auto& strahlkorper_d2x_radius =
      db::get<StrahlkorperTags::D2xRadius<Frame::Inertial>>(box);
  CHECK_ITERABLE_APPROX(expected_d2x_radius, strahlkorper_d2x_radius);

  // Test nabla squared
  DataVector expected_laplacian(n_pts);
  for (size_t s = 0; s < n_pts; ++s) {
    y_11.scalar_laplacian(&expected_laplacian, 1, s, {theta_phi[0][s]},
                          {theta_phi[1][s]});
    expected_laplacian[s] *= y11_amplitude;
  }
  const auto& strahlkorper_laplacian =
      db::get<StrahlkorperTags::LaplacianRadius<Frame::Inertial>>(box);
  CHECK_ITERABLE_APPROX(strahlkorper_laplacian, expected_laplacian);
}

void test_normals() {
  const double y11_amplitude = 1.0;
  const double radius = 2.0;
  const std::array<double, 3> center = {{0.1, 0.2, 0.3}};
  const auto strahlkorper =
      create_strahlkorper_y11(y11_amplitude, radius, center);

  const auto& theta_phi = strahlkorper.ylm_spherepack().theta_phi_points();
  const auto n_pts = theta_phi[0].size();

  // Test surface_tangents

  db::item_type<StrahlkorperTags::Tangents<Frame::Inertial>>
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

  // Create DataBox
  auto box = db::create<
      db::AddSimpleTags<StrahlkorperTags::items_tags<Frame::Inertial>>,
      db::AddComputeTags<
          StrahlkorperTags::compute_items_tags<Frame::Inertial>>>(strahlkorper);

  const auto& surface_tangents =
      db::get<StrahlkorperTags::Tangents<Frame::Inertial>>(box);
  CHECK_ITERABLE_APPROX(surface_tangents, expected_surface_tangents);

  // Test surface_cartesian_coordinates
  db::item_type<StrahlkorperTags::CartesianCoords<Frame::Inertial>>
      expected_cart_coords(n_pts);

  {
    const DataVector temp = radius + amp * sin_phi * sin_theta;
    expected_cart_coords.get(0) = cos_phi * sin_theta * temp + center[0];
    expected_cart_coords.get(1) = sin_phi * sin_theta * temp + center[1];
    expected_cart_coords.get(2) = cos_theta * temp + center[2];
  }
  const auto& cart_coords =
      db::get<StrahlkorperTags::CartesianCoords<Frame::Inertial>>(box);
  CHECK_ITERABLE_APPROX(expected_cart_coords, cart_coords);

  // Test surface_normal_one_form
  db::item_type<StrahlkorperTags::NormalOneForm<Frame::Inertial>>
      expected_normal_one_form(n_pts);
  {
    const auto& r = db::get<StrahlkorperTags::Radius<Frame::Inertial>>(box);
    const DataVector one_over_r = 1.0 / r;
    const DataVector temp = 1.0 + one_over_r * amp * sin_phi * sin_theta;
    expected_normal_one_form.get(0) = cos_phi * sin_theta * temp;
    expected_normal_one_form.get(1) =
        sin_phi * sin_theta * temp - amp * one_over_r;
    expected_normal_one_form.get(2) = cos_theta * temp;
  }
  const auto& normal_one_form =
      db::get<StrahlkorperTags::NormalOneForm<Frame::Inertial>>(box);
  CHECK_ITERABLE_APPROX(expected_normal_one_form, normal_one_form);

  // Test surface_normal_magnitude.
  tnsr::II<DataVector, 3, Frame::Inertial> invg(n_pts);
  invg.get(0, 0) = 1.0;
  invg.get(1, 0) = 0.1;
  invg.get(2, 0) = 0.2;
  invg.get(1, 1) = 2.0;
  invg.get(1, 2) = 0.3;
  invg.get(2, 2) = 3.0;

  const auto expected_normal_mag = [&]() -> DataVector {
    const auto& r = db::get<StrahlkorperTags::Radius<Frame::Inertial>>(box);

    // Nasty expression Mark Scheel computed in Mathematica.
    const DataVector normsquared =
        (-0.3 * cos_theta * (r + amp * sin_phi * sin_theta) *
             (1. * amp * square(cos_phi) +
              1. * amp * square(cos_theta) * square(sin_phi) -
              1. * r * sin_phi * sin_theta +
              cos_phi * (2.0 / 3.0) *
                  (amp * (-1. + square(cos_theta)) * sin_phi - r * sin_theta) +
              cos_theta * (-10. * r - 10. * amp * sin_phi * sin_theta)) +
         0.1 * cos_phi * (amp * (-1. + 1. * square(cos_theta)) * sin_phi -
                          1. * r * sin_theta) *
             (1. * amp * square(cos_phi) +
              1. * amp * square(cos_theta) * square(sin_phi) -
              1. * r * sin_phi * sin_theta +
              cos_phi * (amp * (-10. + 10. * square(cos_theta)) * sin_phi -
                         10. * r * sin_theta) +
              cos_theta * (-2. * r - 2. * amp * sin_phi * sin_theta)) +
         2. * (amp * square(cos_phi) +
               sin_phi *
                   (amp * square(cos_theta) * sin_phi - 1. * r * sin_theta)) *
             (amp * square(cos_phi) +
              amp * square(cos_theta) * square(sin_phi) -
              1. * r * sin_phi * sin_theta +
              cos_phi * 0.05 *
                  (amp * (-1. + square(cos_theta)) * sin_phi - r * sin_theta) +
              cos_theta * (-0.15 * r - 0.15 * amp * sin_phi * sin_theta))) /
        square(r);
    return sqrt(normsquared);
  }();
  const auto& normal_mag = magnitude(normal_one_form, invg);
  CHECK_ITERABLE_APPROX(expected_normal_mag, get(normal_mag));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizons.StrahlkorperDataBox",
                  "[ApparentHorizons][Unit]") {
  test_average_radius();
  test_radius_and_derivs();
  test_normals();
  CHECK(ah::Tags::FastFlow::name() == "FastFlow");
}
