// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Limiters/Flattener.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {

void test_flattener() noexcept {
  const Mesh<3> mesh(2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);

  tnsr::I<DataVector, 3> tilde_b;
  get<0>(tilde_b) = DataVector(8, 0.2);
  get<1>(tilde_b) = DataVector(8, 0.2);
  get<2>(tilde_b) = DataVector{{0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.4, 0.5}};

  tnsr::ii<DataVector, 3> spatial_metric(DataVector(8, 0.));
  get<0, 0>(spatial_metric) = DataVector{{1., 1., 1.1, 1., 1., 1.2, 1., 1.}};
  get<1, 1>(spatial_metric) = DataVector{{1., 1.1, 1., 1., 1., 1.3, 1., 1.}};
  get<2, 2>(spatial_metric) = DataVector{{1.1, 1., 1., 1., 1., 1.4, 1., 1.}};
  const Scalar<DataVector> sqrt_det_spatial_metric(DataVector{
      {sqrt(1.1), sqrt(1.1), sqrt(1.1), 1., 1., sqrt(2.184), 1., 1.}});

  const Scalar<DataVector> det_logical_to_inertial_jacobian(
      DataVector{{0.2, 0.1, 0.3, 0.2, 0.5, 0.6, 0.4, 0.5}});

  {
    INFO("Case: NoOp");
    Scalar<DataVector> tilde_d(
        DataVector{{1.4, 1.3, 1.2, 1.6, 1.8, 1.7, 1.3, 1.4}});
    Scalar<DataVector> tilde_tau(
        DataVector{{2.1, 2.3, 4.3, 3.5, 2.9, 1.8, 2.6, 2.9}});
    tnsr::i<DataVector, 3> tilde_s;
    get<0>(tilde_s) = DataVector(8, 0.);
    get<1>(tilde_s) = DataVector(8, 0.);
    get<2>(tilde_s) =
        DataVector{{-0.3, -0.2, -0.4, -0.3, -0.8, -0.2, -0.3, -0.1}};

    const auto expected_tilde_d = tilde_d;
    const auto expected_tilde_tau = tilde_tau;
    const auto expected_tilde_s = tilde_s;

    const auto flattener_action =
        grmhd::ValenciaDivClean::Limiters::flatten_solution(
            make_not_null(&tilde_d), make_not_null(&tilde_tau),
            make_not_null(&tilde_s), tilde_b, sqrt_det_spatial_metric,
            spatial_metric, mesh, det_logical_to_inertial_jacobian);

    CHECK(flattener_action ==
          grmhd::ValenciaDivClean::Limiters::FlattenerAction::NoOp);
    CHECK_ITERABLE_APPROX(tilde_d, expected_tilde_d);
    CHECK_ITERABLE_APPROX(tilde_tau, expected_tilde_tau);
    CHECK_ITERABLE_APPROX(tilde_s, expected_tilde_s);
  }

  {
    INFO("Case: ScaledSolution because of negative TildeD");
    Scalar<DataVector> tilde_d(
        DataVector{{1.4, 1.3, 1.2, 1.6, -1.6, 1.7, 1.3, 1.4}});
    Scalar<DataVector> tilde_tau(
        DataVector{{2.1, 2.3, 4.3, 3.5, 2.9, 1.8, 2.6, 2.9}});
    tnsr::i<DataVector, 3> tilde_s;
    get<0>(tilde_s) = DataVector(8, 0.);
    get<1>(tilde_s) = DataVector(8, 0.);
    get<2>(tilde_s) =
        DataVector{{-0.3, -0.2, -0.4, -0.3, -0.8, -0.2, -0.3, -0.1}};

    const auto original_mass_density_cons = tilde_d;
    const auto original_energy_density = tilde_tau;
    const auto original_momentum_density = tilde_s;

    const auto flattener_action =
        grmhd::ValenciaDivClean::Limiters::flatten_solution(
            make_not_null(&tilde_d), make_not_null(&tilde_tau),
            make_not_null(&tilde_s), tilde_b, sqrt_det_spatial_metric,
            spatial_metric, mesh, det_logical_to_inertial_jacobian);

    // check 1) action, 2) positive tilde_d, 3) all fields changed
    CHECK(flattener_action ==
          grmhd::ValenciaDivClean::Limiters::FlattenerAction::ScaledSolution);
    CHECK(min(get(tilde_d)) > 0.);
    CHECK_FALSE(tilde_d == original_mass_density_cons);
    CHECK_FALSE(tilde_tau == original_energy_density);
    CHECK_FALSE(tilde_s == original_momentum_density);
  }

  {
    INFO("Case: SetSolutionToMean because of too-small TildeTau");
    Scalar<DataVector> tilde_d(
        DataVector{{1.4, 1.3, 1.2, 1.6, 1.8, 1.7, 1.3, 1.4}});
    Scalar<DataVector> tilde_tau(
        DataVector{{2.1, 2.3, 4.3, 3.5, 2.9, 1.8, 0.1, 2.9}});
    tnsr::i<DataVector, 3> tilde_s;
    get<0>(tilde_s) = DataVector(8, 0.);
    get<1>(tilde_s) = DataVector(8, 0.);
    get<2>(tilde_s) =
        DataVector{{-0.3, -0.2, -0.4, -0.3, -0.8, -0.2, -0.3, -0.1}};

    const auto flattener_action =
        grmhd::ValenciaDivClean::Limiters::flatten_solution(
            make_not_null(&tilde_d), make_not_null(&tilde_tau),
            make_not_null(&tilde_s), tilde_b, sqrt_det_spatial_metric,
            spatial_metric, mesh, det_logical_to_inertial_jacobian);

    CHECK(
        flattener_action ==
        grmhd::ValenciaDivClean::Limiters::FlattenerAction::SetSolutionToMean);
    CHECK_ITERABLE_APPROX(tilde_d, make_with_value<Scalar<DataVector>>(
                                       get(tilde_d), get(tilde_d)[0]));
    CHECK_ITERABLE_APPROX(tilde_tau, make_with_value<Scalar<DataVector>>(
                                         get(tilde_tau), get(tilde_tau)[0]));
    CHECK_ITERABLE_APPROX(get<0>(tilde_s), DataVector(8, get<0>(tilde_s)[0]));
    CHECK_ITERABLE_APPROX(get<1>(tilde_s), DataVector(8, get<1>(tilde_s)[0]));
    CHECK_ITERABLE_APPROX(get<2>(tilde_s), DataVector(8, get<2>(tilde_s)[0]));
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Limiters.Flattener",
                  "[Unit][Evolution]") {
  test_flattener();
}
