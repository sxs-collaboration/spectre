// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/StrahlkorperTestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/YlmTestFunctions.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
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
  const auto theta_phi = strahlkorper.ylm_spherepack().theta_phi_points();
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
          tmpl::push_back<
              StrahlkorperTags::compute_items_tags<Frame::Inertial>>,
          StrahlkorperTags::PhysicalCenterCompute<Frame::Inertial>>>(
      strahlkorper);

  // Test radius
  const auto& strahlkorper_radius =
      get(db::get<StrahlkorperTags::Radius<Frame::Inertial>>(box));
  CHECK_ITERABLE_APPROX(strahlkorper_radius, expected_radius);

  // Test physical center tag
  const auto& strahlkorper_physical_center =
      db::get<StrahlkorperTags::PhysicalCenter<Frame::Inertial>>(box);
  CHECK(strahlkorper_physical_center == strahlkorper.physical_center());

  // Test derivative of radius
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
  const auto& strahlkorper_dx_radius =
      db::get<StrahlkorperTags::DxRadius<Frame::Inertial>>(box);
  CHECK_ITERABLE_APPROX(strahlkorper_dx_radius, expected_dx_radius);

  // Test second derivatives.
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
    expected_d2x_radius.get(0, 1)[s] = -(
        (cos_phi *
         (square(cos_phi) + ((-1. + 3. * cos_2_theta) * square(sin_phi)) / 2.) *
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
  CHECK_ITERABLE_APPROX(get(strahlkorper_laplacian), expected_laplacian);
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

  // Create DataBox
  auto box = db::create<
      db::AddSimpleTags<StrahlkorperTags::items_tags<Frame::Inertial>>,
      db::AddComputeTags<
          StrahlkorperTags::compute_items_tags<Frame::Inertial>>>(strahlkorper);

  const auto& surface_tangents =
      db::get<StrahlkorperTags::Tangents<Frame::Inertial>>(box);
  CHECK_ITERABLE_APPROX(surface_tangents, expected_surface_tangents);

  // Test surface_cartesian_coordinates
  tnsr::I<DataVector, 3> expected_cart_coords(n_pts);

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
  tnsr::i<DataVector, 3> expected_normal_one_form(n_pts);
  {
    const auto& r =
        get(db::get<StrahlkorperTags::Radius<Frame::Inertial>>(box));
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
}

struct SomeType {};
struct SomeTag : db::SimpleTag {
  using type = SomeType;
};
struct DummyScalar : db::SimpleTag {
  using type = Scalar<double>;
};

void test_stf_tensor_tag() {
  static_assert(std::is_same_v<
                Stf::Tags::StfTensor<DummyScalar, 0, 3, Frame::Inertial>::type,
                Scalar<double>>);
  static_assert(std::is_same_v<
                Stf::Tags::StfTensor<DummyScalar, 1, 3, Frame::Inertial>::type,
                tnsr::i<double, 3, Frame::Inertial>>);
  static_assert(std::is_same_v<
                Stf::Tags::StfTensor<DummyScalar, 2, 3, Frame::Inertial>::type,
                tnsr::ii<double, 3, Frame::Inertial>>);
  static_assert(std::is_same_v<
                Stf::Tags::StfTensor<DummyScalar, 3, 3, Frame::Inertial>::type,
                tnsr::iii<double, 3, Frame::Inertial>>);
  TestHelpers::db::test_simple_tag<
      Stf::Tags::StfTensor<DummyScalar, 0, 3, Frame::Inertial>>(
      "StfTensor(DummyScalar,0)");
  TestHelpers::db::test_simple_tag<
      Stf::Tags::StfTensor<DummyScalar, 1, 3, Frame::Inertial>>(
      "StfTensor(DummyScalar,1)");
  TestHelpers::db::test_simple_tag<
      Stf::Tags::StfTensor<DummyScalar, 2, 3, Frame::Inertial>>(
      "StfTensor(DummyScalar,2)");
  TestHelpers::db::test_simple_tag<
      Stf::Tags::StfTensor<DummyScalar, 3, 3, Frame::Inertial>>(
      "StfTensor(DummyScalar,3)");
}

}  // namespace

SPECTRE_TEST_CASE("Unit.SphericalHarmonics.Tags", "[ApparentHorizons][Unit]") {
  test_average_radius();
  test_radius_and_derivs();
  test_normals();
  test_stf_tensor_tag();
  TestHelpers::db::test_simple_tag<
      StrahlkorperTags::Strahlkorper<Frame::Inertial>>("Strahlkorper");
  TestHelpers::db::test_simple_tag<StrahlkorperTags::ThetaPhi<Frame::Inertial>>(
      "ThetaPhi");
  TestHelpers::db::test_simple_tag<StrahlkorperTags::Rhat<Frame::Inertial>>(
      "Rhat");
  TestHelpers::db::test_simple_tag<StrahlkorperTags::Jacobian<Frame::Inertial>>(
      "Jacobian");
  TestHelpers::db::test_simple_tag<
      StrahlkorperTags::InvJacobian<Frame::Inertial>>("InvJacobian");
  TestHelpers::db::test_simple_tag<
      StrahlkorperTags::InvHessian<Frame::Inertial>>("InvHessian");
  TestHelpers::db::test_simple_tag<StrahlkorperTags::Radius<Frame::Inertial>>(
      "Radius");
  TestHelpers::db::test_simple_tag<
      StrahlkorperTags::CartesianCoords<Frame::Inertial>>("CartesianCoords");
  TestHelpers::db::test_simple_tag<StrahlkorperTags::DxRadius<Frame::Inertial>>(
      "DxRadius");
  TestHelpers::db::test_simple_tag<
      StrahlkorperTags::D2xRadius<Frame::Inertial>>("D2xRadius");
  TestHelpers::db::test_simple_tag<
      StrahlkorperTags::LaplacianRadius<Frame::Inertial>>("LaplacianRadius");
  TestHelpers::db::test_simple_tag<
      StrahlkorperTags::NormalOneForm<Frame::Inertial>>("NormalOneForm");
  TestHelpers::db::test_simple_tag<StrahlkorperTags::Tangents<Frame::Inertial>>(
      "Tangents");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::ThetaPhiCompute<Frame::Inertial>>("ThetaPhi");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::RhatCompute<Frame::Inertial>>("Rhat");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::JacobianCompute<Frame::Inertial>>("Jacobian");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::InvJacobianCompute<Frame::Inertial>>("InvJacobian");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::InvHessianCompute<Frame::Inertial>>("InvHessian");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::RadiusCompute<Frame::Inertial>>("Radius");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::PhysicalCenterCompute<Frame::Inertial>>(
      "PhysicalCenter");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::CartesianCoordsCompute<Frame::Inertial>>(
      "CartesianCoords");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::DxRadiusCompute<Frame::Inertial>>("DxRadius");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::D2xRadiusCompute<Frame::Inertial>>("D2xRadius");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::LaplacianRadiusCompute<Frame::Inertial>>(
      "LaplacianRadius");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::NormalOneFormCompute<Frame::Inertial>>("NormalOneForm");
  TestHelpers::db::test_compute_tag<
      StrahlkorperTags::TangentsCompute<Frame::Inertial>>("Tangents");
}
