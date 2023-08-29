// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <climits>
#include <cstddef>
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
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"
#include "Evolution/Systems/Ccz4/Ricci.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <typename Solution>
void test_compute_spatial_ricci_tensor(
    const Solution& solution, size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) {
  // Setup grid
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{grid_size_each_dimension,
                        SpatialDiscretization::Basis::Legendre,
                        SpatialDiscretization::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1., 1., lower_bound[0], upper_bound[0]},
              Affine{-1., 1., lower_bound[1], upper_bound[1]},
              Affine{-1., 1., lower_bound[2], upper_bound[2]},
          });
  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();
  // Evaluate analytic solution
  const auto vars =
      solution.variables(x, t, typename Solution::template tags<DataVector>{});
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, SpatialDim>>(vars);
  const auto det_spatial_metric = determinant_and_inverse(spatial_metric).first;
  const auto& d_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<DataVector, SpatialDim>,
                      tmpl::size_t<SpatialDim>, Frame::Inertial>>(vars);
  const auto d_det_spatial_metric =
      get<gr::Tags::DerivDetSpatialMetric<DataVector, SpatialDim>>(
          solution.variables(
              x, t,
              tmpl::list<
                  gr::Tags::DerivDetSpatialMetric<DataVector, SpatialDim>>{}));

  // Compute arguments for `spatial_ricci_tensor` function to test
  const auto conformal_factor = pow(get(det_spatial_metric), -1. / 6.);

  tnsr::ii<DataVector, SpatialDim, Frame::Inertial> conformal_spatial_metric{};
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = i; j < SpatialDim; j++) {
      conformal_spatial_metric.get(i, j) =
          square(conformal_factor) * spatial_metric.get(i, j);
    }
  }

  tnsr::ijj<DataVector, SpatialDim, Frame::Inertial>
      d_conformal_spatial_metric{};
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        d_conformal_spatial_metric.get(k, i, j) =
            pow<2>(conformal_factor) * d_spatial_metric.get(k, i, j) -
            pow<8>(conformal_factor) * d_det_spatial_metric.get(k) *
                spatial_metric.get(i, j) / 3.;
      }
    }
  }

  const auto inverse_conformal_spatial_metric =
      determinant_and_inverse(conformal_spatial_metric).second;

  tnsr::ijj<DataVector, SpatialDim, Frame::Inertial> field_d{};
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        field_d.get(k, i, j) = 0.5 * d_conformal_spatial_metric.get(k, i, j);
      }
    }
  }

  auto field_d_up = gr::deriv_inverse_spatial_metric(
      inverse_conformal_spatial_metric, field_d);
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        field_d_up.get(k, i, j) *= -1.0;
      }
    }
  }

  const auto contracted_field_d_up =
      ::tenex::evaluate<ti::L>(field_d_up(ti::m, ti::M, ti::L));

  const auto d_field_d =
      partial_derivative(field_d, mesh, coord_map.inv_jacobian(x_logical));
  const auto d_conformal_christoffel_second_kind =
      Ccz4::deriv_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d, d_field_d, field_d_up);

  tnsr::i<DataVector, SpatialDim, Frame::Inertial> field_p{};
  for (size_t i = 0; i < SpatialDim; i++) {
    field_p.get(i) =
        -d_det_spatial_metric.get(i) / (6. * get(det_spatial_metric));
  }

  const auto d_field_p =
      partial_derivative(field_p, mesh, coord_map.inv_jacobian(x_logical));
  const auto conformal_christoffel_second_kind =
      Ccz4::conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                              field_d);

  const auto christoffel_second_kind = Ccz4::christoffel_second_kind(
      conformal_spatial_metric, inverse_conformal_spatial_metric, field_p,
      conformal_christoffel_second_kind);

  const auto contracted_christoffel_second_kind =
      tenex::evaluate<ti::l>(christoffel_second_kind(ti::M, ti::l, ti::m));
  const auto d_christoffel_second_kind = partial_derivative(
      christoffel_second_kind, mesh, coord_map.inv_jacobian(x_logical));
  const auto contracted_d_conformal_christoffel_difference =
      ::tenex::evaluate<ti::i, ti::j>(
          d_conformal_christoffel_second_kind(ti::m, ti::M, ti::i, ti::j) -
          d_conformal_christoffel_second_kind(ti::j, ti::M, ti::i, ti::m));

  // Compute expected and actual ricci tensors using above computed arguments
  const auto expected_py_ricci_tensor{
      pypp::call<tnsr::ii<DataVector, SpatialDim, Frame::Inertial>>(
          "Ricci", "spatial_ricci_tensor", christoffel_second_kind,
          d_conformal_christoffel_second_kind, conformal_spatial_metric,
          inverse_conformal_spatial_metric, field_d, field_d_up, field_p,
          d_field_p)};

  const auto expected_cpp_ricci_tensor =
      gr::ricci_tensor(christoffel_second_kind, d_christoffel_second_kind);

  const auto actual_cpp_ricci_tensor = Ccz4::spatial_ricci_tensor(
      christoffel_second_kind, contracted_christoffel_second_kind,
      contracted_d_conformal_christoffel_difference, conformal_spatial_metric,
      inverse_conformal_spatial_metric, field_d, field_d_up,
      contracted_field_d_up, field_p, d_field_p);

  CHECK_ITERABLE_APPROX(expected_py_ricci_tensor, actual_cpp_ricci_tensor);
  // A custom epsilon is used here because the Legendre polynomials don't fit
  // the derivative of 1 / r well. This was looked at for various box sizes and
  // number of 1D grid points.
  Approx approx = Approx::custom().epsilon(1e-11).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(expected_cpp_ricci_tensor,
                               actual_cpp_ricci_tensor, approx);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Ricci", "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env("Evolution/Systems/Ccz4/");

  const double mass = 2.;
  const std::array<double, 3> spin{{0.3, 0.5, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};

  test_compute_spatial_ricci_tensor(solution, grid_size, lower_bound,
                                    upper_bound);
}
