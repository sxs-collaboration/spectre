// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/Systems/Ccz4/RicciScalarPlusDivergenceZ4Constraint.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <size_t Dim, typename DataType>
void test_compute_ricci_scalar_plus_divergence_z4_constraint(
    const DataType& used_for_size) {
  pypp::check_with_random_values<1>(
      static_cast<Scalar<DataType> (*)(
          const Scalar<DataType>&,
          const tnsr::II<DataType, Dim, Frame::Inertial>&,
          const tnsr::ii<DataType, Dim, Frame::Inertial>&,
          const tnsr::ij<DataType, Dim, Frame::Inertial>&)>(
          &Ccz4::ricci_scalar_plus_divergence_z4_constraint<
              Dim, Frame::Inertial, DataType>),
      "RicciScalarPlusDivergenceZ4Constraint",
      "ricci_scalar_plus_divergence_z4_constraint", {{{-1., 1.}}},
      used_for_size);
}

// Test that when \f$\nabla_i Z_j == 0\f$, \f$R + 2 \nabla_k Z^k == R\f$. Uses
// KerrSchild for reference Ricci tensor solution.
template <typename Solution>
void test_divergence_spatial_z4_constraint_vanishes(
    const Solution& solution, size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) {
  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1., 1., lower_bound[0], upper_bound[0]},
              Affine{-1., 1., lower_bound[1], upper_bound[1]},
              Affine{-1., 1., lower_bound[2], upper_bound[2]},
          });
  const size_t num_points_3d = grid_size_each_dimension *
                               grid_size_each_dimension *
                               grid_size_each_dimension;
  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();
  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<SpatialDim>>(vars);
  const auto det_and_inverse_spatial_metric =
      determinant_and_inverse(spatial_metric);
  const auto det_spatial_metric = det_and_inverse_spatial_metric.first;
  const auto inverse_spatial_metric = det_and_inverse_spatial_metric.second;
  const auto& d_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<SpatialDim>,
                      tmpl::size_t<SpatialDim>, Frame::Inertial>>(vars);

  // Compute arguments for KerrSchild Ricci scalar and
  // `Ccz4::ricci_scalar_plus_divergence_z4_constraint`
  const DataVector used_for_size =
      DataVector(num_points_3d, std::numeric_limits<double>::signaling_NaN());
  Scalar<DataVector> conformal_factor_squared{};
  get(conformal_factor_squared) = pow(get(det_spatial_metric), -1. / 3.);

  tnsr::II<DataVector, SpatialDim, Frame::Inertial>
      inverse_conformal_spatial_metric{};
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = i; j < SpatialDim; j++) {
      inverse_conformal_spatial_metric.get(i, j) =
          inverse_spatial_metric.get(i, j) / get(conformal_factor_squared);
    }
  }

  const auto christoffel_second_kind =
      gr::christoffel_second_kind(d_spatial_metric, inverse_spatial_metric);

  using christoffel_second_kind_tag =
      gr::Tags::SpatialChristoffelSecondKind<SpatialDim, Frame::Inertial,
                                             DataVector>;
  Variables<tmpl::list<christoffel_second_kind_tag>>
      christoffel_second_kind_var(num_points_3d);
  get<christoffel_second_kind_tag>(christoffel_second_kind_var) =
      christoffel_second_kind;
  const auto d_christoffel_second_kind_var =
      partial_derivatives<tmpl::list<christoffel_second_kind_tag>>(
          christoffel_second_kind_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& d_christoffel_second_kind =
      get<Tags::deriv<christoffel_second_kind_tag, tmpl::size_t<SpatialDim>,
                      Frame::Inertial>>(d_christoffel_second_kind_var);

  const auto spatial_ricci_tensor =
      gr::ricci_tensor(christoffel_second_kind, d_christoffel_second_kind);

  auto expected_ricci_scalar =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = 0; j < SpatialDim; j++) {
      get(expected_ricci_scalar) +=
          inverse_spatial_metric.get(i, j) * spatial_ricci_tensor.get(i, j);
    }
  }

  // Let \f$\nabla_i Z_j = 0\f$
  const auto grad_spatial_z4_constraint =
      make_with_value<tnsr::ij<DataVector, SpatialDim, Frame::Inertial>>(
          used_for_size, 0.0);

  const auto actual_result = Ccz4::ricci_scalar_plus_divergence_z4_constraint(
      conformal_factor_squared, inverse_conformal_spatial_metric,
      spatial_ricci_tensor, grad_spatial_z4_constraint);

  // Check that the result is the Ricci scalar
  CHECK_ITERABLE_APPROX(actual_result, expected_ricci_scalar);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.Ccz4.RicciScalarPlusDivergenceZ4Constraint",
    "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env("Evolution/Systems/Ccz4/");

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DOUBLES_AND_DATAVECTORS(
      test_compute_ricci_scalar_plus_divergence_z4_constraint, (1, 2, 3));

  const double mass = 2.;
  const std::array<double, 3> spin{{0.3, 0.5, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};

  test_divergence_spatial_z4_constraint_vanishes(solution, grid_size,
                                                 lower_bound, upper_bound);
}
