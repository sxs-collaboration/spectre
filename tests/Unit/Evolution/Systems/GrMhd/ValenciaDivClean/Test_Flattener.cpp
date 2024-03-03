// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Flattener.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes {
class KastaunEtAl;
}  // namespace grmhd::ValenciaDivClean::PrimitiveRecoverySchemes

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Flattener", "[Unit][GrMhd]") {
  const Mesh<3> mesh(2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  const size_t num_points = mesh.number_of_grid_points();

  const EquationsOfState::Equilibrium3D ideal_fluid{
      EquationsOfState::IdealFluid<true>{4.0 / 3.0}};
  const Scalar<DataVector> tilde_phi(num_points, 0.);
  tnsr::I<DataVector, 3> tilde_b;
  get<0>(tilde_b) = DataVector(num_points, 0.02);
  get<1>(tilde_b) = DataVector(num_points, 0.02);
  get<2>(tilde_b) = DataVector{{0.2, 0.3, 0.1, 0.1, 0.1, 0.01, 0.4, 0.5}};

  const Scalar<DataVector> sqrt_det_spatial_metric(DataVector{
      {sqrt(1.1), sqrt(1.1), sqrt(1.1), 1., 1., sqrt(2.184), 1., 1.}});
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric(num_points, 0.);
  tnsr::II<DataVector, 3, Frame::Inertial> inverse_spatial_metric(num_points,
                                                                  0.);
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = cbrt(get(sqrt_det_spatial_metric));
    inverse_spatial_metric.get(i, i) = 1. / spatial_metric.get(i, i);
  }

  const Scalar<DataVector> det_logical_to_inertial_jacobian(
      DataVector{{0.2, 0.1, 0.3, 0.2, 0.5, 0.6, 0.4, 0.5}});

  const double volume_of_cell =
      definite_integral(get(det_logical_to_inertial_jacobian), mesh);

  const auto flattener = serialize_and_deserialize(
      grmhd::ValenciaDivClean::Flattener<tmpl::list<
          grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>>{
          true, true, false, true});
  CHECK(flattener ==
        grmhd::ValenciaDivClean::Flattener<tmpl::list<
            grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>>{
            true, true, false, true});
  CHECK(flattener !=
        grmhd::ValenciaDivClean::Flattener<tmpl::list<
            grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>>{
            false, true, false, true});
  CHECK(flattener !=
        grmhd::ValenciaDivClean::Flattener<tmpl::list<
            grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>>{
            true, true, true, true});
  CHECK(flattener !=
        grmhd::ValenciaDivClean::Flattener<tmpl::list<
            grmhd::ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>>{
            true, true, false, false});

  const double cutoff_d_for_inversion = 0.0;
  const double density_when_skipping_inversion = 0.0;
  const double kastaun_max_lorentz = 1.0e4;
  const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions
      primitive_from_conservative_options(cutoff_d_for_inversion,
                                          density_when_skipping_inversion,
                                          kastaun_max_lorentz);

  {
    INFO("Case: NoOp");
    Scalar<DataVector> tilde_d(
        DataVector{{1.4, 1.3, 1.2, 1.6, 1.8, 1.7, 1.3, 1.4}});
    Scalar<DataVector> tilde_ye(
        DataVector{{0.14, 0.13, 0.12, 0.16, 0.18, 0.17, 0.13, 0.14}});
    Scalar<DataVector> tilde_tau(
        DataVector{{3.1, 3.3, 4.3, 3.5, 2.9, 2.8, 2.6, 2.7}});
    tnsr::i<DataVector, 3> tilde_s;
    get<0>(tilde_s) = DataVector(num_points, 0.);
    get<1>(tilde_s) = DataVector(num_points, 0.);
    get<2>(tilde_s) =
        DataVector{{-0.3, -0.2, -0.4, -0.3, -0.8, -0.2, -0.3, -0.1}};

    const auto expected_tilde_d = tilde_d;
    const auto expected_tilde_ye = tilde_ye;
    const auto expected_tilde_tau = tilde_tau;
    const auto expected_tilde_s = tilde_s;

    Variables<hydro::grmhd_tags<DataVector>> prims(num_points, 0.);

    flattener(make_not_null(&tilde_d), make_not_null(&tilde_ye),
              make_not_null(&tilde_tau), make_not_null(&tilde_s),
              make_not_null(&prims), tilde_b, tilde_phi,
              sqrt_det_spatial_metric, spatial_metric, inverse_spatial_metric,
              mesh, det_logical_to_inertial_jacobian, ideal_fluid,
              primitive_from_conservative_options);

    CHECK_ITERABLE_APPROX(tilde_d, expected_tilde_d);
    CHECK_ITERABLE_APPROX(tilde_ye, expected_tilde_ye);
    CHECK_ITERABLE_APPROX(tilde_tau, expected_tilde_tau);
    CHECK_ITERABLE_APPROX(tilde_s, expected_tilde_s);
  }

  {
    INFO("Case: ScaledSolution because of negative TildeD");
    constexpr double safety = 0.95;
    Scalar<DataVector> tilde_d(
        DataVector{{1.4, 1.3, 1.2, 1.6, -1.6, 1.7, 1.3, 1.4}});
    Scalar<DataVector> tilde_ye(
        DataVector{{0.14, 0.13, 0.12, 0.16, -0.16, 0.17, 0.13, 0.14}});
    Scalar<DataVector> tilde_tau(
        DataVector{{2.1, 2.3, 4.3, 3.5, 2.9, 1.8, 2.6, 2.9}});
    tnsr::i<DataVector, 3> tilde_s;
    get<0>(tilde_s) = DataVector(num_points, 0.);
    get<1>(tilde_s) = DataVector(num_points, 0.);
    get<2>(tilde_s) =
        DataVector{{-0.3, -0.2, -0.4, -0.3, -0.8, -0.2, -0.3, -0.1}};
    const double expected_mean_tilde_d =
        definite_integral(get(det_logical_to_inertial_jacobian) * get(tilde_d),
                          mesh) /
        volume_of_cell;
    const double expected_mean_tilde_tau =
        definite_integral(
            get(det_logical_to_inertial_jacobian) * get(tilde_tau), mesh) /
        volume_of_cell;
    std::array<double, 3> expected_mean_tilde_s{};
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(expected_mean_tilde_s, i) =
          definite_integral(
              get(det_logical_to_inertial_jacobian) * tilde_s.get(i), mesh) /
          volume_of_cell;
    }

    const double rescale_factor =
        safety * expected_mean_tilde_d /
        (expected_mean_tilde_d -
         (get(tilde_d)[4] * get(sqrt_det_spatial_metric)[4]));
    const Scalar<DataVector> expected_tilde_d(
        DataVector(expected_mean_tilde_d +
                   rescale_factor * (get(tilde_d) - expected_mean_tilde_d)));
    const Scalar<DataVector> expected_tilde_tau(DataVector(
        expected_mean_tilde_tau +
        rescale_factor * (get(tilde_tau) - expected_mean_tilde_tau)));
    auto expected_tilde_s = tilde_s;
    for (size_t i = 0; i < 3; ++i) {
      expected_tilde_s.get(i) =
          gsl::at(expected_mean_tilde_s, i) +
          rescale_factor * (tilde_s.get(i) - gsl::at(expected_mean_tilde_s, i));
    }

    Variables<hydro::grmhd_tags<DataVector>> prims(num_points, 0.);

    flattener(make_not_null(&tilde_d), make_not_null(&tilde_ye),
              make_not_null(&tilde_tau), make_not_null(&tilde_s),
              make_not_null(&prims), tilde_b, tilde_phi,
              sqrt_det_spatial_metric, spatial_metric, inverse_spatial_metric,
              mesh, det_logical_to_inertial_jacobian, ideal_fluid,
              primitive_from_conservative_options);

    // check 1) action, 2) positive tilde_d, 3) all fields changed
    CHECK(min(get(tilde_d)) > 0.);
    CHECK(min(get(tilde_ye)) > 0.);
    CHECK(definite_integral(
              get(det_logical_to_inertial_jacobian) * get(tilde_d), mesh) /
              volume_of_cell ==
          expected_mean_tilde_d);
    CHECK(definite_integral(
              get(det_logical_to_inertial_jacobian) * get(tilde_tau), mesh) /
              volume_of_cell ==
          expected_mean_tilde_tau);
    for (size_t i = 0; i < 3; ++i) {
      CAPTURE(i);
      CHECK(gsl::at(expected_mean_tilde_s, i) ==
            definite_integral(
                get(det_logical_to_inertial_jacobian) * tilde_s.get(i), mesh) /
                volume_of_cell);
    }

    CHECK_ITERABLE_APPROX(tilde_d, expected_tilde_d);
    CHECK_ITERABLE_APPROX(tilde_tau, expected_tilde_tau);
    CHECK_ITERABLE_APPROX(tilde_s, expected_tilde_s);
  }

  {
    INFO("Case: SetSolutionToMean because of too-small TildeTau");
    Scalar<DataVector> tilde_d(
        DataVector{{1.4, 1.3, 1.2, 1.6, 1.8, 1.7, 1.3, 1.4}});
    Scalar<DataVector> tilde_ye(
        DataVector{{0.14, 0.13, 0.12, 0.16, 0.18, 0.17, 0.13, 0.14}});
    Scalar<DataVector> tilde_tau(
        DataVector{{3.1, 3.3, 4.3, 3.5, 2.9, 2.8, 2.6, 9.e-3}});
    tnsr::i<DataVector, 3> tilde_s;
    get<0>(tilde_s) = DataVector(num_points, 0.);
    get<1>(tilde_s) = DataVector(num_points, 0.);
    get<2>(tilde_s) =
        DataVector{{-0.3, -0.2, -0.4, -0.3, -0.8, -0.2, -0.3, -0.1}};

    Variables<hydro::grmhd_tags<DataVector>> prims(num_points, 0.);

    const double expected_mean_tilde_d =
        definite_integral(get(det_logical_to_inertial_jacobian) * get(tilde_d),
                          mesh) /
        volume_of_cell;
    const double expected_mean_tilde_tau =
        definite_integral(
            get(det_logical_to_inertial_jacobian) * get(tilde_tau), mesh) /
        volume_of_cell;
    std::array<double, 3> expected_mean_tilde_s{};
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(expected_mean_tilde_s, i) =
          definite_integral(
              get(det_logical_to_inertial_jacobian) * tilde_s.get(i), mesh) /
          volume_of_cell;
    }

    constexpr double safety = 0.99;
    const double scale_factor =
        safety *
        (0.5 *
             get(dot_product(tilde_b, tilde_b,
                             spatial_metric))[num_points - 1] /
             get(sqrt_det_spatial_metric)[num_points - 1] -
         expected_mean_tilde_tau) /
        (get(tilde_tau)[num_points - 1] - expected_mean_tilde_tau);

    const Scalar<DataVector> expected_tilde_tau{
        DataVector{expected_mean_tilde_tau +
                   scale_factor * (get(tilde_tau) - expected_mean_tilde_tau)}};
    const Scalar<DataVector> expected_tilde_d = tilde_d;
    const tnsr::i<DataVector, 3, Frame::Inertial> expected_tilde_s = tilde_s;

    flattener(make_not_null(&tilde_d), make_not_null(&tilde_ye),
              make_not_null(&tilde_tau), make_not_null(&tilde_s),
              make_not_null(&prims), tilde_b, tilde_phi,
              sqrt_det_spatial_metric, spatial_metric, inverse_spatial_metric,
              mesh, det_logical_to_inertial_jacobian, ideal_fluid,
              primitive_from_conservative_options);

    CHECK(definite_integral(
              get(det_logical_to_inertial_jacobian) * get(tilde_d), mesh) /
              volume_of_cell ==
          expected_mean_tilde_d);
    CHECK(definite_integral(
              get(det_logical_to_inertial_jacobian) * get(tilde_tau), mesh) /
              volume_of_cell ==
          approx(expected_mean_tilde_tau));
    for (size_t i = 0; i < 3; ++i) {
      CAPTURE(i);
      CHECK(gsl::at(expected_mean_tilde_s, i) ==
            definite_integral(
                get(det_logical_to_inertial_jacobian) * tilde_s.get(i), mesh) /
                volume_of_cell);
    }

    CHECK_ITERABLE_APPROX(tilde_d, expected_tilde_d);
    CHECK_ITERABLE_APPROX(tilde_tau, expected_tilde_tau);
    CHECK_ITERABLE_APPROX(tilde_s, expected_tilde_s);

    // Check that the scaling made the max be less than one:
    CHECK(1. > max(0.5 * get(dot_product(tilde_b, tilde_b, spatial_metric)) /
                   get(sqrt_det_spatial_metric) / get(tilde_tau)));
  }
}
