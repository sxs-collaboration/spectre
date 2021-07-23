// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <pup.h>
#include <random>
#include <string>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/Factory.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/ScalarWave/BoundaryCorrections/UpwindPenalty.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

namespace {
template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> gen, const size_t num_pts) {
  PUPable_reg(CurvedScalarWave::BoundaryCorrections::UpwindPenalty<Dim>);

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      CurvedScalarWave::System<Dim>>(
      gen,
      "Evolution.Systems.CurvedScalarWave.BoundaryCorrections.UpwindPenalty",
      {{"dg_package_data_v_psi", "dg_package_data_v_zero",
        "dg_package_data_v_plus", "dg_package_data_v_minus",
        "dg_package_data_gamma2", "dg_package_data_interface_unit_normal",
        "dg_package_data_char_speeds"}},
      {{"dg_boundary_terms_pi", "dg_boundary_terms_phi",
        "dg_boundary_terms_psi"}},
      CurvedScalarWave::BoundaryCorrections::UpwindPenalty<Dim>{},
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::Gauss},
      {}, {});

  const auto upwind_penalty = TestHelpers::test_creation<std::unique_ptr<
      CurvedScalarWave::BoundaryCorrections::BoundaryCorrection<Dim>>>(
      "UpwindPenalty:");

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      CurvedScalarWave::System<Dim>>(
      gen,
      "Evolution.Systems.CurvedScalarWave.BoundaryCorrections.UpwindPenalty",
      {{"dg_package_data_v_psi", "dg_package_data_v_zero",
        "dg_package_data_v_plus", "dg_package_data_v_minus",
        "dg_package_data_gamma2", "dg_package_data_interface_unit_normal",
        "dg_package_data_char_speeds"}},
      {{"dg_boundary_terms_pi", "dg_boundary_terms_phi",
        "dg_boundary_terms_psi"}},
      dynamic_cast<
          const CurvedScalarWave::BoundaryCorrections::UpwindPenalty<Dim>&>(
          *upwind_penalty),
      Mesh<Dim - 1>{num_pts, Spectral::Basis::Legendre,
                    Spectral::Quadrature::Gauss},
      {}, {});
}

template <size_t Dim>
void test_flat_spacetime(const gsl::not_null<std::mt19937*> gen) {
  INFO("test consistency of the curved-scalar-wave upwind flux")
  std::uniform_real_distribution<> uniform_m11_dist(0.1, 1.);

  const size_t num_pts = 3;
  const DataVector value{1., 2., 3.};
  const tnsr::I<DataVector, Dim> x{value};
  const double t = 1.3;

  gr::Solutions::Minkowski<Dim> minkowski{};
  ScalarWave::BoundaryCorrections::UpwindPenalty<Dim> sw_flux_computer{};
  CurvedScalarWave::BoundaryCorrections::UpwindPenalty<Dim> csw_flux_computer{};

  // variables for CSW
  auto csw_v_psi_int = Scalar<DataVector>(num_pts);
  auto csw_v_psi_ext = Scalar<DataVector>(num_pts);
  auto csw_v_zero_int = tnsr::i<DataVector, Dim>(num_pts);
  auto csw_v_zero_ext = tnsr::i<DataVector, Dim>(num_pts);
  auto csw_v_plus_int = Scalar<DataVector>(num_pts);
  auto csw_v_plus_ext = Scalar<DataVector>(num_pts);
  auto csw_v_minus_int = Scalar<DataVector>(num_pts);
  auto csw_v_minus_ext = Scalar<DataVector>(num_pts);
  auto csw_gamma2_int = Scalar<DataVector>(num_pts);
  auto csw_gamma2_ext = Scalar<DataVector>(num_pts);
  auto csw_normal_int = tnsr::i<DataVector, Dim>(num_pts);
  auto csw_normal_ext = tnsr::i<DataVector, Dim>(num_pts);
  auto csw_char_speeds_int = tnsr::a<DataVector, 3>(num_pts);
  auto csw_char_speeds_ext = tnsr::a<DataVector, 3>(num_pts);
  auto csw_pi_bcorr = Scalar<DataVector>(num_pts);
  auto csw_phi_bcorr = tnsr::i<DataVector, Dim>(num_pts);
  auto csw_psi_bcorr = Scalar<DataVector>(num_pts);

  // variables for SW
  auto sw_char_speed_v_psi_int = Scalar<DataVector>(num_pts);
  auto sw_char_speed_v_psi_ext = Scalar<DataVector>(num_pts);
  auto sw_char_speed_v_zero_int = tnsr::i<DataVector, Dim>(num_pts);
  auto sw_char_speed_v_zero_ext = tnsr::i<DataVector, Dim>(num_pts);
  auto sw_char_speed_v_plus_int = Scalar<DataVector>(num_pts);
  auto sw_char_speed_v_plus_ext = Scalar<DataVector>(num_pts);
  auto sw_char_speed_v_minus_int = Scalar<DataVector>(num_pts);
  auto sw_char_speed_v_minus_ext = Scalar<DataVector>(num_pts);
  auto sw_char_speed_n_times_v_plus_int = tnsr::i<DataVector, Dim>(num_pts);
  auto sw_char_speed_n_times_v_plus_ext = tnsr::i<DataVector, Dim>(num_pts);
  auto sw_char_speed_n_times_v_minus_int = tnsr::i<DataVector, Dim>(num_pts);
  auto sw_char_speed_n_times_v_minus_ext = tnsr::i<DataVector, Dim>(num_pts);
  auto sw_char_speed_gamma2_v_psi_int = Scalar<DataVector>(num_pts);
  auto sw_char_speed_gamma2_v_psi_ext = Scalar<DataVector>(num_pts);
  auto sw_char_speeds_int = tnsr::i<DataVector, 3>(num_pts);
  auto sw_char_speeds_ext = tnsr::i<DataVector, 3>(num_pts);
  auto sw_pi_bcorr = Scalar<DataVector>(num_pts);
  auto sw_phi_bcorr = tnsr::i<DataVector, Dim>(num_pts);
  auto sw_psi_bcorr = Scalar<DataVector>(num_pts);

  // For CSW & SW
  // create pi
  const auto pi = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&uniform_m11_dist), x);
  // create phi
  const auto phi = make_with_random_values<tnsr::i<DataVector, Dim>>(
      gen, make_not_null(&uniform_m11_dist), x);
  // create psi
  const auto psi = make_with_random_values<Scalar<DataVector>>(
      gen, make_not_null(&uniform_m11_dist), x);
  // create gamma2
  const Scalar<DataVector> gamma2(num_pts, 0.);
  // create normal covector
  auto normal_int = make_with_random_values<tnsr::i<DataVector, Dim>>(
      gen, make_not_null(&uniform_m11_dist), x);
  tnsr::i<DataVector, Dim> normal_ext{num_pts};
  {
    const auto mag_normal = magnitude(normal_int);
    for (size_t i = 0; i < Dim; ++i) {
      normal_int.get(i) /= get(mag_normal);
      normal_ext.get(i) = -normal_int.get(i);
    }
  }
  // create normal vector
  auto normal_vector_int = make_with_random_values<tnsr::I<DataVector, Dim>>(
      gen, make_not_null(&uniform_m11_dist), x);
  tnsr::I<DataVector, Dim> normal_vector_ext{num_pts};
  {
    const auto mag_normal = magnitude(normal_vector_int);
    for (size_t i = 0; i < Dim; ++i) {
      normal_vector_int.get(i) /= get(mag_normal);
      normal_vector_ext.get(i) = -normal_vector_int.get(i);
    }
  }
  // create mesh velocity
  auto mesh_velocity = make_with_random_values<tnsr::I<DataVector, Dim>>(
      gen, make_not_null(&uniform_m11_dist), x);

  const auto mag_mesh_v = magnitude(mesh_velocity);
  for (size_t i = 0; i < Dim; ++i) {
    mesh_velocity.get(i) /= (2. * get(mag_mesh_v));
  }
  // create normal dot mesh velocity
  const auto normal_dot_mesh_velociy_int =
      dot_product(normal_int, mesh_velocity);
  const auto normal_dot_mesh_velociy_ext =
      dot_product(normal_ext, mesh_velocity);

  // For CSW
  // gamma1
  const Scalar<DataVector> gamma1(num_pts, 0.);
  // lapse
  const auto lapse = get<gr::Tags::Lapse<DataVector>>(
      minkowski.variables(x, t, tmpl::list<gr::Tags::Lapse<DataVector>>{}));
  // shift
  const auto shift = get<gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>(
      minkowski.variables(
          x, t,
          tmpl::list<gr::Tags::Shift<Dim, Frame::Inertial, DataVector>>{}));
  // inverse spatial metric
  const auto inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>>(
          minkowski.variables(
              x, t,
              tmpl::list<gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial,
                                                        DataVector>>{}));

  // ----- package data : CSW
  csw_flux_computer.dg_package_data(
      make_not_null(&csw_v_psi_int), make_not_null(&csw_v_zero_int),
      make_not_null(&csw_v_plus_int), make_not_null(&csw_v_minus_int),
      make_not_null(&csw_gamma2_int), make_not_null(&csw_normal_int),
      make_not_null(&csw_char_speeds_int), pi, phi, psi, lapse, shift,
      inverse_spatial_metric, gamma1, gamma2, normal_int, normal_vector_int,
      mesh_velocity, normal_dot_mesh_velociy_int);
  csw_flux_computer.dg_package_data(
      make_not_null(&csw_v_psi_ext), make_not_null(&csw_v_zero_ext),
      make_not_null(&csw_v_plus_ext), make_not_null(&csw_v_minus_ext),
      make_not_null(&csw_gamma2_ext), make_not_null(&csw_normal_ext),
      make_not_null(&csw_char_speeds_ext), pi, phi, psi, lapse, shift,
      inverse_spatial_metric, gamma1, gamma2, normal_ext, normal_vector_ext,
      mesh_velocity, normal_dot_mesh_velociy_ext);
  csw_flux_computer.dg_boundary_terms(
      make_not_null(&csw_pi_bcorr), make_not_null(&csw_phi_bcorr),
      make_not_null(&csw_psi_bcorr), csw_v_psi_int, csw_v_zero_int,
      csw_v_plus_int, csw_v_minus_int, csw_gamma2_int, csw_normal_int,
      csw_char_speeds_int, csw_v_psi_ext, csw_v_zero_ext, csw_v_plus_ext,
      csw_v_minus_ext, csw_gamma2_ext, csw_normal_ext, csw_char_speeds_ext,
      dg::Formulation::StrongInertial);

  // ----- package data : SW
  sw_flux_computer.dg_package_data(
      make_not_null(&sw_char_speed_v_psi_int),
      make_not_null(&sw_char_speed_v_zero_int),
      make_not_null(&sw_char_speed_v_plus_int),
      make_not_null(&sw_char_speed_v_minus_int),
      make_not_null(&sw_char_speed_n_times_v_plus_int),
      make_not_null(&sw_char_speed_n_times_v_minus_int),
      make_not_null(&sw_char_speed_gamma2_v_psi_int),
      make_not_null(&sw_char_speeds_int), pi, phi, psi, gamma2, normal_int,
      mesh_velocity, normal_dot_mesh_velociy_int);
  sw_flux_computer.dg_package_data(
      make_not_null(&sw_char_speed_v_psi_ext),
      make_not_null(&sw_char_speed_v_zero_ext),
      make_not_null(&sw_char_speed_v_plus_ext),
      make_not_null(&sw_char_speed_v_minus_ext),
      make_not_null(&sw_char_speed_n_times_v_plus_ext),
      make_not_null(&sw_char_speed_n_times_v_minus_ext),
      make_not_null(&sw_char_speed_gamma2_v_psi_ext),
      make_not_null(&sw_char_speeds_ext), pi, phi, psi, gamma2, normal_ext,
      mesh_velocity, normal_dot_mesh_velociy_ext);
  sw_flux_computer.dg_boundary_terms(
      make_not_null(&sw_pi_bcorr), make_not_null(&sw_phi_bcorr),
      make_not_null(&sw_psi_bcorr), sw_char_speed_v_psi_int,
      sw_char_speed_v_zero_int, sw_char_speed_v_plus_int,
      sw_char_speed_v_minus_int, sw_char_speed_n_times_v_plus_int,
      sw_char_speed_n_times_v_minus_int, sw_char_speed_gamma2_v_psi_int,
      sw_char_speeds_int, sw_char_speed_v_psi_ext, sw_char_speed_v_zero_ext,
      sw_char_speed_v_plus_ext, sw_char_speed_v_minus_ext,
      sw_char_speed_n_times_v_plus_ext, sw_char_speed_n_times_v_minus_ext,
      sw_char_speed_gamma2_v_psi_ext, sw_char_speeds_ext,
      dg::Formulation::StrongInertial);

  CHECK_ITERABLE_APPROX(sw_pi_bcorr, csw_pi_bcorr);
  CHECK_ITERABLE_APPROX(sw_phi_bcorr, csw_phi_bcorr);
  CHECK_ITERABLE_APPROX(sw_psi_bcorr, csw_psi_bcorr);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.CurvedScalarWave.BoundaryCorrections.UpwindPenalty",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};
  MAKE_GENERATOR(gen);

  test<1>(make_not_null(&gen), 1);
  test<2>(make_not_null(&gen), 5);
  test<3>(make_not_null(&gen), 5);

  test_flat_spacetime<1>(make_not_null(&gen));
  test_flat_spacetime<2>(make_not_null(&gen));
  test_flat_spacetime<3>(make_not_null(&gen));
}
