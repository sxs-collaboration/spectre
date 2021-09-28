// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <random>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MetricIdentityJacobian.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <size_t Dim>
auto make_map() noexcept {
  if constexpr (Dim == 1) {
    return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
        Affine{-1.0, 1.0, -0.3, 0.7});
  } else if constexpr (Dim == 2) {
    return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
        Affine2D{{-1.0, 1.0, -1.0, -0.99}, {-1.0, 1.0, -1.0, -0.99}},
        domain::CoordinateMaps::Wedge<2>{1.0, 2.0, 0.0, 1.0, {}, false});
  } else {
    return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
        Affine3D{{-1.0, 1.0, -1.0, -0.99},
                 {-1.0, 1.0, -1.0, -0.99},
                 {-1.0, 1.0, -1.0, 1.0}},
        domain::CoordinateMaps::ProductOf2Maps<domain::CoordinateMaps::Wedge<2>,
                                               Affine>{
            {1.0, 2.0, 0.0, 1.0, {}, false}, {0.0, 1.0, 0.0, 1.0}});
  }
}

template <size_t Dim>
void test(const Mesh<Dim>& mesh) {
  CAPTURE(Dim);
  CAPTURE(mesh);
  std::uniform_real_distribution<double> dist(-1.0, 2.3);
  MAKE_GENERATOR(gen);
  {
    tnsr::I<DataVector, Dim, Frame::Inertial> inertial_coords{
        mesh.number_of_grid_points()};
    Jacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial> jacobian{
        mesh.number_of_grid_points()};

    fill_with_random_values(make_not_null(&inertial_coords),
                            make_not_null(&gen), make_not_null(&dist));
    fill_with_random_values(make_not_null(&jacobian), make_not_null(&gen),
                            make_not_null(&dist));

    InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
        result{};

    dg::metric_identity_det_jac_times_inv_jac(make_not_null(&result), mesh,
                                              inertial_coords, jacobian);

    // Check metric identities are satisfied by taking the numerical divergence
    tnsr::i<DataVector, Dim, Frame::Inertial> divergence_terms{
        mesh.number_of_grid_points(), 0.0};
    DataVector buffer(mesh.number_of_grid_points());
    std::array<Mesh<1>, Dim> meshes_1d{};
    for (size_t i = 0; i < Dim; ++i) {
      gsl::at(meshes_1d, i) = mesh.slice_through(i);
    }
    const Matrix identity{};

    for (size_t i_hat = 0; i_hat < Dim; ++i_hat) {  // logical index
      auto diff_matrices = make_array<Dim>(std::cref(identity));
      gsl::at(diff_matrices, i_hat) =
          Spectral::differentiation_matrix(gsl::at(meshes_1d, i_hat));
      for (size_t i = 0; i < Dim; ++i) {  // inertial index
        apply_matrices(make_not_null(&buffer), diff_matrices,
                       result.get(i_hat, i), mesh.extents());
        divergence_terms.get(i) += buffer;
      }
    }

    Approx local_approx = Approx::custom().epsilon(1.0e-12).scale(1.);
    const DataVector expected{mesh.number_of_grid_points(), 0.0};
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_CUSTOM_APPROX(divergence_terms.get(i), expected,
                                   local_approx);
    }
  }

  // Now check with a map that the terms of the inverse Jacobian (times the
  // Jacobian determinent) are computed correctly. This is not expected to be
  // super accurate unless the maps are very well resolved.
  auto map = make_map<Dim>();
  const InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      analytic_inverse_jacobian = map.inv_jacobian(logical_coordinates(mesh));
  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      analytic_det_jac_times_inverse_jacobian = analytic_inverse_jacobian;
  const Scalar<DataVector> analytic_det_jac{
      1.0 / get(determinant(analytic_inverse_jacobian))};
  for (auto& t : analytic_det_jac_times_inverse_jacobian) {
    t *= get(analytic_det_jac);
  }
  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      det_jac_times_inverse_jacobian{};

  dg::metric_identity_det_jac_times_inv_jac(
      make_not_null(&det_jac_times_inverse_jacobian), mesh,
      map(logical_coordinates(mesh)), map.jacobian(logical_coordinates(mesh)));

  Approx coarse_local_approx = Approx::custom().epsilon(1.0e-9).scale(1.);
  for (size_t i = 0; i < Dim; ++i) {
    CAPTURE(i);
    for (size_t j = 0; j < Dim; ++j) {
      CAPTURE(j);
      CHECK_ITERABLE_CUSTOM_APPROX(
          det_jac_times_inverse_jacobian.get(i, j),
          analytic_det_jac_times_inverse_jacobian.get(i, j),
          coarse_local_approx);
    }
  }

  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      det_jac_times_inverse_jacobian2{};
  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      inverse_jacobian{};
  Jacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial> jacobian =
      map.jacobian(logical_coordinates(mesh));
  const auto expected_jacobian = jacobian;
  Scalar<DataVector> det_jacobian{};

  dg::metric_identity_jacobian_quantities(
      make_not_null(&det_jac_times_inverse_jacobian2),
      make_not_null(&inverse_jacobian), make_not_null(&jacobian),
      make_not_null(&det_jacobian), mesh, map(logical_coordinates(mesh)));

  Approx identity_matrix_local_approx =
      Approx::custom().epsilon(1.0e-12).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(get(det_jacobian), get(analytic_det_jac),
                               coarse_local_approx);
  for (size_t i = 0; i < Dim; ++i) {
    CAPTURE(i);
    for (size_t j = 0; j < Dim; ++j) {
      CAPTURE(j);
      CHECK_ITERABLE_APPROX(det_jac_times_inverse_jacobian.get(i, j),
                            det_jac_times_inverse_jacobian2.get(i, j));
      CHECK_ITERABLE_APPROX(
          DataVector{get(det_jacobian) * inverse_jacobian.get(i, j)},
          det_jac_times_inverse_jacobian2.get(i, j));
      CHECK_ITERABLE_CUSTOM_APPROX(
          jacobian.get(i, j), expected_jacobian.get(i, j), coarse_local_approx);
      DataVector jac_inv_jac_contracted =
          jacobian.get(i, 0) * inverse_jacobian.get(0, j);
      for (size_t k = 1; k < Dim; ++k) {
        jac_inv_jac_contracted +=
            jacobian.get(i, k) * inverse_jacobian.get(k, j);
      }
      const DataVector expected{mesh.number_of_grid_points(),
                                i == j ? 1.0 : 0.0};
      CHECK_ITERABLE_CUSTOM_APPROX(expected, jac_inv_jac_contracted,
                                   identity_matrix_local_approx);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DG.MetricIdentityJacobian",
                  "[Unit][NumericalAlgorithms]") {
  test(Mesh<1>{5, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto});
  test(Mesh<1>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss});

  test(Mesh<2>{5, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto});
  test(Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss});

  test(Mesh<3>{5, Spectral::Basis::Legendre,
               Spectral::Quadrature::GaussLobatto});
  test(Mesh<3>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss});
}
