// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Evolution/Systems/Ccz4/ATilde.hpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"
#include "Evolution/Systems/Ccz4/TimeDerivative.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

// Equations referenced below can be found in \cite Dumbser2017okk

// Compute g(\alpha)
Scalar<DataVector> get_slicing_condition(
    const Ccz4::SlicingConditionType slicing_condition_type,
    const Scalar<DataVector>& lapse) {
  Scalar<DataVector> slicing_condition(get(lapse));
  if (slicing_condition_type == Ccz4::SlicingConditionType::Harmonic) {
    // harmonic slicing condition: g(\alpha) = 1.0
    get(slicing_condition) = 1.0;
  } else if (slicing_condition_type == Ccz4::SlicingConditionType::Log) {
    // 1 + log slicing condition: g(\alpha) = 2 / \alpha
    get(slicing_condition) = 2.0 / get(lapse);
  } else {
    ERROR("Unknown Ccz4::SlicingConditionType");
  }
  return slicing_condition;
}

// Compute the trace of the extrinsic curvature K = K_{ij} * \gamma^ij
template <size_t SpatialDim, typename FrameType>
Scalar<DataVector> get_trace_extrinsic_curvature(
    const tnsr::ii<DataVector, SpatialDim, FrameType>& extrinsic_curvature,
    const tnsr::II<DataVector, SpatialDim, FrameType>& inverse_spatial_metric) {
  Scalar<DataVector> trace_extrinsic_curvature(get<0, 0>(extrinsic_curvature));
  get(trace_extrinsic_curvature) = 0.0;
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = 0; j < SpatialDim; j++) {
      get(trace_extrinsic_curvature) +=
          extrinsic_curvature.get(i, j) * inverse_spatial_metric.get(i, j);
    }
  }
  return trace_extrinsic_curvature;
}

// Compute A_i from eq 6
template <size_t SpatialDim, typename FrameType>
tnsr::i<DataVector, SpatialDim, FrameType> get_field_a(
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, SpatialDim, FrameType>& d_lapse) {
  tnsr::i<DataVector, SpatialDim, FrameType> field_a(get(lapse));
  for (size_t i = 0; i < SpatialDim; i++) {
    field_a.get(i) = d_lapse.get(i) / get(lapse);
  }
  return field_a;
}

// Compute \partial A_i from eq 6
//
// eq 6:
//   A_i = \partial_i \alpha / \alpha
// then, the derivative is:
//   \partial_i A_j =
//       ((\partial_i (\partial_j \alpha)) \alpha -
//         \partial_j \alpha \partial_i \alpha) / \alpha^2
template <size_t SpatialDim, typename FrameType>
tnsr::ij<DataVector, SpatialDim, FrameType> get_d_field_a(
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, SpatialDim, FrameType>& d_lapse,
    const tnsr::ij<DataVector, SpatialDim, FrameType>& d_d_lapse) {
  tnsr::ij<DataVector, SpatialDim, FrameType> d_field_a(get(lapse));
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = 0; j < SpatialDim; j++) {
      d_field_a.get(i, j) =
          (d_d_lapse.get(i, j) * get(lapse) - d_lapse.get(j) * d_lapse.get(i)) /
          square(get(lapse));
    }
  }
  return d_field_a;
}

// Compute the conformal spatial metric
//
// \tilde{\gamma}_{ij} = \phi^2 \gamma_{ij}
template <size_t SpatialDim, typename FrameType>
tnsr::ii<DataVector, SpatialDim, FrameType> get_conformal_spatial_metric(
    const Scalar<DataVector>& conformal_factor_squared,
    const tnsr::ii<DataVector, SpatialDim, FrameType>& spatial_metric) {
  tnsr::ii<DataVector, SpatialDim, FrameType> conformal_spatial_metric(
      get(conformal_factor_squared));
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = i; j < SpatialDim; j++) {
      conformal_spatial_metric.get(i, j) =
          get(conformal_factor_squared) * spatial_metric.get(i, j);
    }
  }
  return conformal_spatial_metric;
}

// Compute the spatial derivative of the conformal spatial metric
//
// If \tilde{\gamma}_{ij} is the conformal metric and \phi is the
// conformal factor, \tilde{\gamma}_{ij} = \phi^2 \gamma_{ij}.
// Therefore, the derivative of the conformal metric is:
//   \partial_k \tilde{\gamma}_{ij} =
//       \phi^2 \partial_k \gamma_{ij} +
//       \partial_k \phi^2 \gamma_{ij}
//
// Since \phi = (det(\gamma_{ij}))^{-1/6}:
//   \partial_k \phi^2
//        = \partial_k ((det(\gamma_{ij}))^{-1/6})^2
//        = \partial_k (det(\gamma_{ij}))^{-1/3}
//        = -(det(\gamma_{ij}))^{-4/3}  \partial_k (det(\gamma_{ij}))/ 3
//        = -\phi^4 \partial_k (det(\gamma_{ij})) / 3
//
// Therefore:
//   \partial_k \tilde{\gamma}_{ij} =
//       \phi^2 \partial_k \gamma_{ij} -
//       \phi^4 \partial_k (det(\gamma_{ij})) \gamma_{ij} / 3
template <size_t SpatialDim, typename FrameType>
tnsr::ijj<DataVector, SpatialDim, FrameType> get_d_conformal_spatial_metric(
    const Scalar<DataVector>& conformal_factor_squared,
    const tnsr::ii<DataVector, SpatialDim, FrameType>& spatial_metric,
    const tnsr::ijj<DataVector, SpatialDim, FrameType>& d_spatial_metric,
    const tnsr::i<DataVector, SpatialDim, FrameType>& d_det_spatial_metric) {
  tnsr::ijj<DataVector, SpatialDim, FrameType> d_conformal_spatial_metric(
      get(conformal_factor_squared));
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        d_conformal_spatial_metric.get(k, i, j) =
            get(conformal_factor_squared) * d_spatial_metric.get(k, i, j) -
            pow<4>(get(conformal_factor_squared)) *
                d_det_spatial_metric.get(k) * spatial_metric.get(i, j) / 3.;
      }
    }
  }
  return d_conformal_spatial_metric;
}

// Compute D_{kij} from eq 6
template <size_t SpatialDim, typename FrameType>
tnsr::ijj<DataVector, SpatialDim, FrameType> get_field_d(
    const tnsr::ijj<DataVector, SpatialDim, FrameType>&
        d_conformal_spatial_metric) {
  tnsr::ijj<DataVector, SpatialDim, FrameType> field_d(
      get<0, 0, 0>(d_conformal_spatial_metric));
  for (size_t i = 0; i < field_d.size(); i++) {
    field_d[i] = 0.5 * d_conformal_spatial_metric[i];
  }
  return field_d;
}

// Compute D_{k}^{ij} from eq 14
template <size_t SpatialDim, typename FrameType>
tnsr::iJJ<DataVector, SpatialDim, FrameType> get_field_d_up(
    const tnsr::II<DataVector, SpatialDim, FrameType>&
        inverse_conformal_spatial_metric,
    const tnsr::ijj<DataVector, SpatialDim, FrameType>& field_d) {
  tnsr::iJJ<DataVector, SpatialDim, FrameType> field_d_up =
      gr::deriv_inverse_spatial_metric(inverse_conformal_spatial_metric,
                                       field_d);
  for (size_t i = 0; i < field_d.size(); i++) {
    field_d_up[i] *= -1.0;
  }
  return field_d_up;
}

// Compute P_i from eq 6
template <size_t SpatialDim, typename FrameType>
tnsr::i<DataVector, SpatialDim, FrameType> get_field_p(
    const Scalar<DataVector>& det_spatial_metric,
    const tnsr::i<DataVector, SpatialDim, FrameType>& d_det_spatial_metric) {
  tnsr::i<DataVector, SpatialDim, FrameType> field_p(get(det_spatial_metric));
  for (size_t i = 0; i < SpatialDim; i++) {
    field_p.get(i) =
        -d_det_spatial_metric.get(i) / (6. * get(det_spatial_metric));
  }
  return field_p;
}

// \brief Test first order CCZ4 with different binary settings against Minkowski
//
// \details Tests that all time derivatives are 0. The evolution equations are
// eq 12a - 12m in \cite Dumbser2017okk.
//
// \param evolve_shift whether or not to evolve the shift
// \param slicing_condition_type which slicing condition to use, e.g.
// harmonic, 1 + log
void test_minkowski(const Ccz4::EvolveShift evolve_shift,
                    const Ccz4::SlicingConditionType slicing_condition_type) {
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;

  // Setup solution
  gr::Solutions::Minkowski<SpatialDim> solution{};

  // Setup grid
  const size_t num_points_1d = 4;
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
  Mesh<SpatialDim> mesh{num_points_1d, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, FrameType>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });
  const size_t num_points_3d = num_points_1d * num_points_1d * num_points_1d;
  const DataVector used_for_size =
      DataVector(num_points_3d, std::numeric_limits<double>::signaling_NaN());
  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Evaluate solution
  const auto minkowski_vars = solution.variables(
      x, t, typename gr::Solutions::Minkowski<SpatialDim>::tags<DataVector>{});

  // Get ingredients for computing arguments to Ccz4::TimeDerivative
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, SpatialDim, FrameType>>(
          minkowski_vars);
  const auto det_spatial_metric = determinant(spatial_metric);
  const auto d_det_spatial_metric =
      make_with_value<tnsr::i<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, SpatialDim, FrameType>>(
          minkowski_vars);
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(minkowski_vars);
  const auto& shift =
      get<gr::Tags::Shift<DataVector, SpatialDim, FrameType>>(minkowski_vars);
  const auto& d_shift =
      get<Tags::deriv<gr::Tags::Shift<DataVector, SpatialDim, FrameType>,
                      tmpl::size_t<SpatialDim>, FrameType>>(minkowski_vars);

  // Params
  const double c = 1.0;
  const double cleaning_speed = 1.6;  // e
  const double f = 0.75;
  const double kappa_1 = 0.1;
  const double kappa_2 = 0.3;
  const double kappa_3 = 0.4;
  const double mu = 0.7;
  const double one_over_relaxation_time = 10.0;         // \tau^{-1}
  const Scalar<DataVector> slicing_condition =          // g(\alpha)
      get_slicing_condition(slicing_condition_type, lapse);

  // Choose free variables \Theta, K_0, b^i, and \eta
  const auto theta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto d_theta =
      make_with_value<tnsr::i<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);

  const auto extrinsic_curvature =
      make_with_value<tnsr::ii<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  const auto trace_extrinsic_curvature =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto d_trace_extrinsic_curvature =
      make_with_value<tnsr::i<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);

  // K_0 == K
  const auto& k_0 = trace_extrinsic_curvature;
  const auto& d_k_0 = d_trace_extrinsic_curvature;

  const auto b = make_with_value<tnsr::I<DataVector, SpatialDim, FrameType>>(
      used_for_size, 0.0);
  const auto d_b = make_with_value<tnsr::iJ<DataVector, SpatialDim, FrameType>>(
      used_for_size, 0.0);

  const auto eta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);

  // Compute arguments for Ccz4::TimeDerivative
  Scalar<DataVector> ln_lapse{};
  get(ln_lapse) = log(get(lapse));

  // eq 6
  const auto field_a =
      make_with_value<tnsr::i<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);
  const auto d_field_a =
      make_with_value<tnsr::ij<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  const auto& field_b = d_shift;
  const auto d_field_b =
      make_with_value<tnsr::ijK<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  // since spatial_metric = conformal_spatial_metric,
  // conformal factor == 1
  const auto conformal_factor_squared =
      make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
  const auto ln_conformal_factor =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);

  // eq 3
  const auto a_tilde =
      Ccz4::a_tilde(conformal_factor_squared, spatial_metric,
                    extrinsic_curvature, trace_extrinsic_curvature);
  const auto d_a_tilde =
      make_with_value<tnsr::ijj<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  const auto& conformal_spatial_metric = spatial_metric;
  const auto d_conformal_spatial_metric =
      make_with_value<tnsr::ijj<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  const auto& inverse_conformal_spatial_metric = inverse_spatial_metric;

  // eq 6
  const auto field_d =
      make_with_value<tnsr::ijj<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);
  const auto d_field_d =
      make_with_value<tnsr::ijkk<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  // eq 14
  const auto field_d_up =
      make_with_value<tnsr::iJJ<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  // eq 6
  const auto field_p =
      make_with_value<tnsr::i<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);
  const auto d_field_p =
      make_with_value<tnsr::ij<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  // eq 15
  const auto conformal_christoffel_second_kind =
      Ccz4::conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                              field_d);
  // eq 16
  const auto d_conformal_christoffel_second_kind =
      make_with_value<tnsr::iJkk<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  // eq 23
  const auto contracted_conformal_christoffel_second_kind =
      Ccz4::contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, conformal_christoffel_second_kind);
  // eq 24
  const auto d_contracted_conformal_christoffel_second_kind =
      make_with_value<tnsr::iJ<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);

  // If Z4 constraint is 0, then \hat{\gamma} == \tilde{\gamma}
  const auto& gamma_hat = contracted_conformal_christoffel_second_kind;
  const auto& d_gamma_hat = d_contracted_conformal_christoffel_second_kind;

  // LHS time derivatives of evolved variables: eq 12a - 12m
  tnsr::ii<DataVector, SpatialDim> dt_conformal_spatial_metric_actual(
      used_for_size);
  Scalar<DataVector> dt_ln_lapse_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_shift_actual(used_for_size);
  Scalar<DataVector> dt_ln_conformal_factor_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> dt_a_tilde_actual(used_for_size);
  Scalar<DataVector> dt_trace_extrinsic_curvature_actual(used_for_size);
  Scalar<DataVector> dt_theta_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_gamma_hat_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_b_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> dt_field_a_actual(used_for_size);
  tnsr::iJ<DataVector, SpatialDim> dt_field_b_actual(used_for_size);
  tnsr::ijj<DataVector, SpatialDim> dt_field_d_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> dt_field_p_actual(used_for_size);
  // quantities we need for computing eq 12 - 27
  Scalar<DataVector> conformal_factor_squared_actual(used_for_size);
  Scalar<DataVector> det_conformal_spatial_metric_actual(used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_conformal_spatial_metric_actual(
      used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_spatial_metric_actual(used_for_size);
  Scalar<DataVector> lapse_actual(used_for_size);
  Scalar<DataVector> slicing_condition_actual(used_for_size);
  Scalar<DataVector> d_slicing_condition_actual(used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_a_tilde_actual(used_for_size);
  // quantities we need for computing eq 12 - 27
  tnsr::ij<DataVector, SpatialDim> a_tilde_times_field_b_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim>
      a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde_actual(
          used_for_size);
  Scalar<DataVector> contracted_field_b_actual(used_for_size);
  tnsr::ijK<DataVector, SpatialDim> symmetrized_d_field_b_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> contracted_symmetrized_d_field_b_actual(
      used_for_size);
  tnsr::ijk<DataVector, SpatialDim> field_b_times_field_d_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> field_d_up_times_a_tilde_actual(
      used_for_size);
  tnsr::I<DataVector, SpatialDim> contracted_field_d_up_actual(used_for_size);
  Scalar<DataVector> half_conformal_factor_squared_actual(used_for_size);
  tnsr::ij<DataVector, SpatialDim> conformal_metric_times_field_b_actual(
      used_for_size);
  tnsr::ijk<DataVector, SpatialDim>
      conformal_metric_times_symmetrized_d_field_b_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> conformal_metric_times_trace_a_tilde_actual(
      used_for_size);
  tnsr::i<DataVector, SpatialDim> inv_conformal_metric_times_d_a_tilde_actual(
      used_for_size);
  tnsr::I<DataVector, SpatialDim>
      gamma_hat_minus_contracted_conformal_christoffel_actual(used_for_size);
  tnsr::iJ<DataVector, SpatialDim>
      d_gamma_hat_minus_contracted_conformal_christoffel_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> contracted_christoffel_second_kind_actual(
      used_for_size);
  tnsr::ij<DataVector, SpatialDim>
      contracted_d_conformal_christoffel_difference_actual(used_for_size);
  Scalar<DataVector> k_minus_2_theta_c_actual(used_for_size);
  Scalar<DataVector> k_minus_k0_minus_2_theta_c_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> lapse_times_a_tilde_actual(used_for_size);
  tnsr::ijj<DataVector, SpatialDim> lapse_times_d_a_tilde_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> lapse_times_field_a_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> lapse_times_conformal_spatial_metric_actual(
      used_for_size);
  Scalar<DataVector> lapse_times_slicing_condition_actual(used_for_size);
  Scalar<DataVector>
      lapse_times_ricci_scalar_plus_divergence_z4_constraint_actual(
          used_for_size);
  tnsr::I<DataVector, SpatialDim> shift_times_deriv_gamma_hat_actual(
      used_for_size);
  tnsr::ii<DataVector, SpatialDim> inv_tau_times_conformal_metric_actual(
      used_for_size);
  // expressions and identities needed for evolution equations: eq 13 - 27
  Scalar<DataVector> trace_a_tilde_actual(used_for_size);
  tnsr::iJJ<DataVector, SpatialDim> field_d_up_actual(used_for_size);
  tnsr::Ijj<DataVector, SpatialDim> conformal_christoffel_second_kind_actual(
      used_for_size);
  tnsr::iJkk<DataVector, SpatialDim> d_conformal_christoffel_second_kind_actual(
      used_for_size);
  tnsr::Ijj<DataVector, SpatialDim> christoffel_second_kind_actual(
      used_for_size);
  tnsr::ii<DataVector, SpatialDim> spatial_ricci_tensor_actual(used_for_size);
  tnsr::ij<DataVector, SpatialDim> grad_grad_lapse_actual(used_for_size);
  Scalar<DataVector> divergence_lapse_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim>
      contracted_conformal_christoffel_second_kind_actual(used_for_size);
  tnsr::iJ<DataVector, SpatialDim>
      d_contracted_conformal_christoffel_second_kind_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> spatial_z4_constraint_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> upper_spatial_z4_constraint_actual(
      used_for_size);
  tnsr::ij<DataVector, SpatialDim> grad_spatial_z4_constraint_actual(
      used_for_size);
  Scalar<DataVector> ricci_scalar_plus_divergence_z4_constraint_actual(
      used_for_size);

  ::Ccz4::TimeDerivative<SpatialDim>::apply(
      make_not_null(&dt_conformal_spatial_metric_actual),
      make_not_null(&dt_ln_lapse_actual), make_not_null(&dt_shift_actual),
      make_not_null(&dt_ln_conformal_factor_actual),
      make_not_null(&dt_a_tilde_actual),
      make_not_null(&dt_trace_extrinsic_curvature_actual),
      make_not_null(&dt_theta_actual), make_not_null(&dt_gamma_hat_actual),
      make_not_null(&dt_b_actual), make_not_null(&dt_field_a_actual),
      make_not_null(&dt_field_b_actual), make_not_null(&dt_field_d_actual),
      make_not_null(&dt_field_p_actual),
      make_not_null(&conformal_factor_squared_actual),
      make_not_null(&det_conformal_spatial_metric_actual),
      make_not_null(&inv_conformal_spatial_metric_actual),
      make_not_null(&inv_spatial_metric_actual), make_not_null(&lapse_actual),
      make_not_null(&slicing_condition_actual),
      make_not_null(&d_slicing_condition_actual),
      make_not_null(&inv_a_tilde_actual),
      make_not_null(&a_tilde_times_field_b_actual),
      make_not_null(
          &a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde_actual),
      make_not_null(&contracted_field_b_actual),
      make_not_null(&symmetrized_d_field_b_actual),
      make_not_null(&contracted_symmetrized_d_field_b_actual),
      make_not_null(&field_b_times_field_d_actual),
      make_not_null(&field_d_up_times_a_tilde_actual),
      make_not_null(&contracted_field_d_up_actual),
      make_not_null(&half_conformal_factor_squared_actual),
      make_not_null(&conformal_metric_times_field_b_actual),
      make_not_null(&conformal_metric_times_symmetrized_d_field_b_actual),
      make_not_null(&conformal_metric_times_trace_a_tilde_actual),
      make_not_null(&inv_conformal_metric_times_d_a_tilde_actual),
      make_not_null(&gamma_hat_minus_contracted_conformal_christoffel_actual),
      make_not_null(&d_gamma_hat_minus_contracted_conformal_christoffel_actual),
      make_not_null(&contracted_christoffel_second_kind_actual),
      make_not_null(&contracted_d_conformal_christoffel_difference_actual),
      make_not_null(&k_minus_2_theta_c_actual),
      make_not_null(&k_minus_k0_minus_2_theta_c_actual),
      make_not_null(&lapse_times_a_tilde_actual),
      make_not_null(&lapse_times_d_a_tilde_actual),
      make_not_null(&lapse_times_field_a_actual),
      make_not_null(&lapse_times_conformal_spatial_metric_actual),
      make_not_null(&lapse_times_slicing_condition_actual),
      make_not_null(
          &lapse_times_ricci_scalar_plus_divergence_z4_constraint_actual),
      make_not_null(&shift_times_deriv_gamma_hat_actual),
      make_not_null(&inv_tau_times_conformal_metric_actual),
      make_not_null(&trace_a_tilde_actual), make_not_null(&field_d_up_actual),
      make_not_null(&conformal_christoffel_second_kind_actual),
      make_not_null(&d_conformal_christoffel_second_kind_actual),
      make_not_null(&christoffel_second_kind_actual),
      make_not_null(&spatial_ricci_tensor_actual),
      make_not_null(&grad_grad_lapse_actual),
      make_not_null(&divergence_lapse_actual),
      make_not_null(&contracted_conformal_christoffel_second_kind_actual),
      make_not_null(&d_contracted_conformal_christoffel_second_kind_actual),
      make_not_null(&spatial_z4_constraint_actual),
      make_not_null(&upper_spatial_z4_constraint_actual),
      make_not_null(&grad_spatial_z4_constraint_actual),
      make_not_null(&ricci_scalar_plus_divergence_z4_constraint_actual), c,
      cleaning_speed, eta, f, k_0, d_k_0, kappa_1, kappa_2, kappa_3, mu,
      one_over_relaxation_time, evolve_shift, slicing_condition_type,
      conformal_spatial_metric, ln_lapse, shift, ln_conformal_factor, a_tilde,
      trace_extrinsic_curvature, theta, gamma_hat, b, field_a, field_b, field_d,
      field_p, d_a_tilde, d_trace_extrinsic_curvature, d_theta, d_gamma_hat,
      d_b, d_field_a, d_field_b, d_field_d, d_field_p);

  const auto zero = DataVector(used_for_size.size(), 0.0);

  // Check that all time derivatives are 0
  for (auto& component : dt_conformal_spatial_metric_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_ln_lapse_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_shift_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_ln_conformal_factor_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_a_tilde_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_trace_extrinsic_curvature_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_theta_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_gamma_hat_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_b_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_field_a_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_field_b_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_field_d_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  for (auto& component : dt_field_p_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
}

// Compute K_0 for KerrSchild
//
// Solve eq 4g for K_0, where \partial_t \alpha = 0:
//   \partial_t \alpha =
//       -\alpha^2 g(\alpha) (K - K_0 - 2 \Theta) +
//       \beta^k \partial_k \alpha
//   K_0 = -((\beta^k \partial_k \alpha) / (\alpha^2 * g(\alpha)) -
//           K + 2 \Theta);
template <size_t SpatialDim, typename FrameType>
Scalar<DataVector> get_k_0_kerr(
    const tnsr::I<DataVector, SpatialDim, FrameType>& shift,
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, SpatialDim, FrameType>& d_lapse,
    const Scalar<DataVector>& slicing_condition,
    const Scalar<DataVector>& theta,
    const Scalar<DataVector>& trace_extrinsic_curvature) {
  Scalar<DataVector> k_0(get(lapse));
  get(k_0) = get<0>(shift) * get<0>(d_lapse);
  for (size_t k = 1; k < SpatialDim; k++) {
    get(k_0) += shift.get(k) * d_lapse.get(k);
  }
  get(k_0) = -((get(k_0) / (square(get(lapse)) * get(slicing_condition))) -
               get(trace_extrinsic_curvature) + 2.0 * get(theta));
  return k_0;
}

// Compute b^i for KerrSchild
//
// Solve eq 12c for b^i, where \partial_t \beta^i = 0:
//   \partial_t \beta^i = s f b + s \beta^k \partial_k \beta^i
template <size_t SpatialDim, typename FrameType>
tnsr::I<DataVector, SpatialDim, FrameType> get_b_kerr(
    const Ccz4::EvolveShift evolve_shift,
    const tnsr::I<DataVector, SpatialDim, FrameType>& shift,
    const tnsr::iJ<DataVector, SpatialDim, FrameType>& d_shift,
    const double f) {
  tnsr::I<DataVector, SpatialDim, FrameType> b(get<0>(shift));
  if (not static_cast<bool>(evolve_shift)) {
    // s == 0
    // pick b = 0
    for (auto& component : b) {
      component = 0.0;
    }
  } else {
    // s == 1
    // b = -(\beta^k \partial_k \beta^i) / f
    for (size_t i = 0; i < SpatialDim; i++) {
      b.get(i) = -shift.get(0) * d_shift.get(0, i);
      for (size_t k = 1; k < SpatialDim; k++) {
        b.get(i) -= shift.get(k) * d_shift.get(k, i);
      }
      b.get(i) /= f;
    }
  }
  return b;
}

// Compute expected value for LHS of eq 12h
//
// \partial_t \hat{\gamma}^i will not be 0 for KerrSchild if
// evolve_shift == false
template <size_t SpatialDim, typename FrameType>
tnsr::I<DataVector, SpatialDim, FrameType> get_dt_gamma_hat_kerr_expected(
    const Ccz4::EvolveShift evolve_shift,
    const tnsr::II<DataVector, SpatialDim, FrameType>&
        inverse_conformal_spatial_metric,
    const tnsr::ijK<DataVector, SpatialDim, FrameType>& d_field_b) {
  tnsr::I<DataVector, SpatialDim, FrameType> dt_gamma_hat_kerr_expected(
      get<0, 0>(inverse_conformal_spatial_metric));
  if (static_cast<bool>(evolve_shift)) {
    // s == 1
    for (auto& component : dt_gamma_hat_kerr_expected) {
      component = 0.0;
    }
  } else {
    // s == 0
    //
    // When s == 1, \partial_t \hat{\gamma}^i == 0, so when s == 0,
    // dt_gamma_hat = terms in eq 12h with s as a factor
    // (red terms that cancel out are ignored)
    for (size_t i = 0; i < SpatialDim; i++) {
      dt_gamma_hat_kerr_expected.get(i) = 0.0;
      for (size_t k = 0; k < SpatialDim; k++) {
        for (size_t l = 0; l < SpatialDim; l++) {
          dt_gamma_hat_kerr_expected.get(i) -=
              (0.5 * inverse_conformal_spatial_metric.get(k, l) *
                   (d_field_b.get(k, l, i) + d_field_b.get(l, k, i)) +
               0.5 * inverse_conformal_spatial_metric.get(i, k) *
                   (d_field_b.get(k, l, l) + d_field_b.get(l, k, l)) / 3.0);
        }
      }
    }
  }
  return dt_gamma_hat_kerr_expected;
}

// Compute expected value for LHS of eq 12i
//
// \partial_t b will not be 0 for KerrSchild if evolve_shift == true
template <size_t SpatialDim, typename FrameType>
tnsr::I<DataVector, SpatialDim, FrameType> get_dt_b_kerr_expected(
    const Ccz4::EvolveShift evolve_shift, const Scalar<DataVector>& eta,
    const tnsr::I<DataVector, SpatialDim, FrameType>& shift,
    const tnsr::iJ<DataVector, SpatialDim, FrameType>& d_gamma_hat,
    const tnsr::I<DataVector, SpatialDim, FrameType>& b,
    const tnsr::iJ<DataVector, SpatialDim, FrameType>& d_b) {
  tnsr::I<DataVector, SpatialDim, FrameType> dt_b_kerr_expected(get(eta));
  if (static_cast<bool>(evolve_shift)) {
    // s == 1
    for (size_t i = 0; i < SpatialDim; i++) {
      dt_b_kerr_expected.get(i) = -get(eta) * b.get(i) +
                                  shift.get(0) * d_b.get(0, i) -
                                  shift.get(0) * d_gamma_hat.get(0, i);
      for (size_t k = 1; k < SpatialDim; k++) {
        dt_b_kerr_expected.get(i) +=
            shift.get(k) * d_b.get(k, i) - shift.get(k) * d_gamma_hat.get(k, i);
      }
    }
  } else {
    // s == 0
    for (auto& component : dt_b_kerr_expected) {
      component = 0.0;
    }
  }
  return dt_b_kerr_expected;
}

// Compute expected value for LHS of eq 12l
//
// \partial_t D_{kij} will not be 0 for KerrSchild if evolve_shift == false
template <size_t SpatialDim, typename FrameType>
tnsr::ijj<DataVector, SpatialDim, FrameType> get_dt_field_d_kerr_expected(
    const Ccz4::EvolveShift evolve_shift,
    const tnsr::ii<DataVector, SpatialDim, FrameType>& conformal_spatial_metric,
    const tnsr::ijK<DataVector, SpatialDim, FrameType>& d_field_b) {
  tnsr::ijj<DataVector, SpatialDim, FrameType> dt_field_d_kerr_expected(
      get<0, 0>(conformal_spatial_metric));
  if (static_cast<bool>(evolve_shift)) {
    // s == 1
    for (auto& component : dt_field_d_kerr_expected) {
      component = 0.0;
    }
  } else {
    // s == 0
    //
    // When s == 1, \partial_t D_{kij} == 0, so when s == 0,
    // \partial_t D_{kij} = terms in eq 12l with s as a factor
    for (size_t k = 0; k < SpatialDim; k++) {
      for (size_t j = 0; j < SpatialDim; j++) {
        for (size_t i = 0; i < SpatialDim; i++) {
          dt_field_d_kerr_expected.get(k, i, j) = 0.0;
          for (size_t m = 0; m < SpatialDim; m++) {
            dt_field_d_kerr_expected.get(k, i, j) +=
                0.5 * conformal_spatial_metric.get(i, j) *
                    (d_field_b.get(k, m, m) + d_field_b.get(m, k, m)) / 3.0 -
                0.25 * conformal_spatial_metric.get(m, i) *
                    (d_field_b.get(k, j, m) + d_field_b.get(j, k, m)) -
                0.25 * conformal_spatial_metric.get(m, j) *
                    (d_field_b.get(k, i, m) + d_field_b.get(i, k, m));
          }
        }
      }
    }
  }
  return dt_field_d_kerr_expected;
}

// Compute expected value for LHS of eq 12m
//
// \partial_t P_i will not be 0 for KerrSchild if evolve_shift == false
template <size_t SpatialDim, typename FrameType>
tnsr::i<DataVector, SpatialDim, FrameType> get_dt_field_p_kerr_expected(
    const Ccz4::EvolveShift evolve_shift,
    const tnsr::ijK<DataVector, SpatialDim, FrameType>& d_field_b) {
  tnsr::i<DataVector, SpatialDim, FrameType> dt_field_p_kerr_expected(
      get<0, 0, 0>(d_field_b));
  if (static_cast<bool>(evolve_shift)) {
    // s == 1
    for (auto& component : dt_field_p_kerr_expected) {
      component = 0.0;
    }
  } else {
    // s == 0
    //
    // When s == 1, \partial_t P_i == 0, so when s == 0,
    // \partial_t P_i = terms in eq 12m with s as a factor
    // (red terms that cancel out are ignored)
    for (size_t k = 0; k < SpatialDim; k++) {
      dt_field_p_kerr_expected.get(k) = 0.0;
      for (size_t i = 0; i < SpatialDim; i++) {
        dt_field_p_kerr_expected.get(k) +=
            d_field_b.get(k, i, i) + d_field_b.get(i, k, i);
      }
      dt_field_p_kerr_expected.get(k) /= 6.0;  // *= 0.5 / 3
    }
  }
  return dt_field_p_kerr_expected;
}

// \brief Test first order CCZ4 with different binary settings against
// KerrSchild
//
// \details Tests that most time derivatives are 0. Depending on whether or not
// the shift is evolved, some evolution variables will have non-zero time
// derivatives. The evolution equations are eq 12a - 12m in
// \cite Dumbser2017okk. More concretely, depending on whether s == 1 or s == 0,
// the time derivatives in eq 12c, 12h, 12i, 12k, 12l, and 12m may or may not be
// expected to be 0. For cases when the time derivative of an evolution
// variable is expected to be non-zero, the test checks for the expected
// non-zero value by computing excess/missing terms due to the value of s.
//
// \param evolve_shift whether or not to evolve the shift
// \param slicing_condition_type which slicing condition to use, e.g.
// harmonic, 1 + log
void test_kerrschild(const Ccz4::EvolveShift evolve_shift,
                     const Ccz4::SlicingConditionType slicing_condition_type) {
  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;

  // Setup solution
  const double mass = 2.;
  const std::array<double, 3> spin{{0.3, 0.5, 0.2}};
  const std::array<double, 3> center{{0.2, 0.3, 0.4}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  // Setup grid
  const size_t num_points_1d = 6;
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
  Mesh<SpatialDim> mesh{num_points_1d, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, FrameType>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });
  const size_t num_points_3d = num_points_1d * num_points_1d * num_points_1d;
  const DataVector used_for_size =
      DataVector(num_points_3d, std::numeric_limits<double>::signaling_NaN());
  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();
  // Evaliuate analytic solution
  const auto kerrschild_vars = solution.variables(
      x, t, typename gr::Solutions::KerrSchild::tags<DataVector>{});

  // Get ingredients for computing arguments to Ccz4::TimeDerivative
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, SpatialDim, FrameType>>(
          kerrschild_vars);
  const auto& d_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<DataVector, SpatialDim>,
                      tmpl::size_t<SpatialDim>, FrameType>>(kerrschild_vars);
  const auto det_spatial_metric = determinant(spatial_metric);
  const auto d_det_spatial_metric =
      get<gr::Tags::DerivDetSpatialMetric<DataVector, SpatialDim, FrameType>>(
          solution.variables(
              x, t,
              tmpl::list<gr::Tags::DerivDetSpatialMetric<DataVector, SpatialDim,
                                                         FrameType>>{}));
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<DataVector, SpatialDim>>>(
          kerrschild_vars);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, SpatialDim, FrameType>>(
          kerrschild_vars);
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(kerrschild_vars);
  const auto& d_lapse =
      get<Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<SpatialDim>,
                      FrameType>>(kerrschild_vars);
  const auto& shift =
      get<gr::Tags::Shift<DataVector, SpatialDim, FrameType>>(kerrschild_vars);
  const auto& d_shift =
      get<Tags::deriv<gr::Tags::Shift<DataVector, SpatialDim, FrameType>,
                      tmpl::size_t<SpatialDim>, FrameType>>(kerrschild_vars);

  // Params
  const double c = 1.0;
  const double cleaning_speed = 1.6;  // e in the paper
  const double f = 0.75;
  const double kappa_1 = 0.1;
  const double kappa_2 = 0.3;
  const double kappa_3 = 0.4;
  const double mu = 0.7;
  const double one_over_relaxation_time = 10.0;         // \tau^{-1}
  const Scalar<DataVector> slicing_condition =          // g(\alpha)
      get_slicing_condition(slicing_condition_type, lapse);

  // Choose free variables \Theta, K_0, b^i, and \eta
  const auto theta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);
  const auto d_theta =
      make_with_value<tnsr::i<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);

  const auto extrinsic_curvature =
      gr::extrinsic_curvature(lapse, shift, d_shift, spatial_metric,
                              dt_spatial_metric, d_spatial_metric);

  // K = K_{ij} * \gamma^ij
  const Scalar<DataVector> trace_extrinsic_curvature =
      get_trace_extrinsic_curvature(extrinsic_curvature,
                                    inverse_spatial_metric);
  const auto d_trace_extrinsic_curvature = partial_derivative(
      trace_extrinsic_curvature, mesh, coord_map.inv_jacobian(x_logical));

  // Solve eq 4g for K_0, where \partial_t \alpha = 0
  const Scalar<DataVector> k_0 =
      get_k_0_kerr(shift, lapse, d_lapse, slicing_condition, theta,
                   trace_extrinsic_curvature);
  const auto d_k_0 =
      partial_derivative(k_0, mesh, coord_map.inv_jacobian(x_logical));

  // Solve eq 12c for b^i, where \partial_t \beta^i = 0
  const tnsr::I<DataVector, SpatialDim, FrameType> b =
      get_b_kerr(evolve_shift, shift, d_shift, f);
  const auto d_b =
      partial_derivative(b, mesh, coord_map.inv_jacobian(x_logical));

  const auto eta = make_with_value<Scalar<DataVector>>(used_for_size, 0.0);

  // Compute arguments for Ccz4::TimeDerivative
  Scalar<DataVector> ln_lapse{};
  get(ln_lapse) = log(get(lapse));

  // eq 6
  const tnsr::i<DataVector, SpatialDim, FrameType> field_a =
      get_field_a(lapse, d_lapse);

  const auto d_d_lapse =
      partial_derivative(d_lapse, mesh, coord_map.inv_jacobian(x_logical));
  // \partial_i A_j , where A_i is defined in eq 6
  const tnsr::ij<DataVector, SpatialDim, FrameType> d_field_a =
      get_d_field_a(lapse, d_lapse, d_d_lapse);

  // eq 6
  const auto& field_b = d_shift;
  const auto d_field_b =
      partial_derivative(field_b, mesh, coord_map.inv_jacobian(x_logical));

  const auto conformal_factor = pow(get(det_spatial_metric), -1. / 6.);
  Scalar<DataVector> conformal_factor_squared{};
  get(conformal_factor_squared) = square(conformal_factor);
  Scalar<DataVector> ln_conformal_factor{};
  get(ln_conformal_factor) = log(conformal_factor);

  // eq 3
  const auto a_tilde =
      Ccz4::a_tilde(conformal_factor_squared, spatial_metric,
                    extrinsic_curvature, trace_extrinsic_curvature);
  const auto d_a_tilde =
      partial_derivative(a_tilde, mesh, coord_map.inv_jacobian(x_logical));

  // \tilde{\gamma}_{ij} = \phi^2 \gamma_{ij}
  const tnsr::ii<DataVector, SpatialDim, FrameType> conformal_spatial_metric =
      get_conformal_spatial_metric(conformal_factor_squared, spatial_metric);
  // \partial_k \tilde{\gamma}_{ij}
  const tnsr::ijj<DataVector, SpatialDim, FrameType>
      d_conformal_spatial_metric = get_d_conformal_spatial_metric(
          conformal_factor_squared, spatial_metric, d_spatial_metric,
          d_det_spatial_metric);

  const auto inverse_conformal_spatial_metric =
      determinant_and_inverse(conformal_spatial_metric).second;

  // eq 6
  const tnsr::ijj<DataVector, SpatialDim, FrameType> field_d =
      get_field_d(d_conformal_spatial_metric);
  const auto d_field_d =
      partial_derivative(field_d, mesh, coord_map.inv_jacobian(x_logical));

  // eq 14
  const tnsr::iJJ<DataVector, SpatialDim, FrameType> field_d_up =
      get_field_d_up(inverse_conformal_spatial_metric, field_d);

  // eq 6
  const tnsr::i<DataVector, SpatialDim, FrameType> field_p =
      get_field_p(det_spatial_metric, d_det_spatial_metric);
  const auto d_field_p =
      partial_derivative(field_p, mesh, coord_map.inv_jacobian(x_logical));

  // eq 15
  const auto conformal_christoffel_second_kind =
      Ccz4::conformal_christoffel_second_kind(inverse_conformal_spatial_metric,
                                              field_d);
  // eq 16
  const auto d_conformal_christoffel_second_kind =
      Ccz4::deriv_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d, d_field_d, field_d_up);

  // eq 23
  const auto contracted_conformal_christoffel_second_kind =
      Ccz4::contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, conformal_christoffel_second_kind);
  // eq 24
  const auto d_contracted_conformal_christoffel_second_kind =
      Ccz4::deriv_contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d_up,
          conformal_christoffel_second_kind,
          d_conformal_christoffel_second_kind);

  // If Z4 constraint is 0, then \hat{\gamma} == \tilde{\gamma}
  const auto& gamma_hat = contracted_conformal_christoffel_second_kind;
  const auto& d_gamma_hat = d_contracted_conformal_christoffel_second_kind;

  // LHS time derivatives of evolved variables: eq 12a - 12m
  tnsr::ii<DataVector, SpatialDim> dt_conformal_spatial_metric_actual(
      used_for_size);
  Scalar<DataVector> dt_ln_lapse_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_shift_actual(used_for_size);
  Scalar<DataVector> dt_ln_conformal_factor_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> dt_a_tilde_actual(used_for_size);
  Scalar<DataVector> dt_trace_extrinsic_curvature_actual(used_for_size);
  Scalar<DataVector> dt_theta_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_gamma_hat_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> dt_b_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> dt_field_a_actual(used_for_size);
  tnsr::iJ<DataVector, SpatialDim> dt_field_b_actual(used_for_size);
  tnsr::ijj<DataVector, SpatialDim> dt_field_d_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> dt_field_p_actual(used_for_size);
  // quantities we need for computing eq 12 - 27
  Scalar<DataVector> conformal_factor_squared_actual(used_for_size);
  Scalar<DataVector> det_conformal_spatial_metric_actual(used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_conformal_spatial_metric_actual(
      used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_spatial_metric_actual(used_for_size);
  Scalar<DataVector> lapse_actual(used_for_size);
  Scalar<DataVector> slicing_condition_actual(used_for_size);
  Scalar<DataVector> d_slicing_condition_actual(used_for_size);
  tnsr::II<DataVector, SpatialDim> inv_a_tilde_actual(used_for_size);
  // quantities we need for computing eq 12 - 27
  tnsr::ij<DataVector, SpatialDim> a_tilde_times_field_b_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim>
      a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde_actual(
          used_for_size);
  Scalar<DataVector> contracted_field_b_actual(used_for_size);
  tnsr::ijK<DataVector, SpatialDim> symmetrized_d_field_b_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> contracted_symmetrized_d_field_b_actual(
      used_for_size);
  tnsr::ijk<DataVector, SpatialDim> field_b_times_field_d_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> field_d_up_times_a_tilde_actual(
      used_for_size);
  tnsr::I<DataVector, SpatialDim> contracted_field_d_up_actual(used_for_size);
  Scalar<DataVector> half_conformal_factor_squared_actual(used_for_size);
  tnsr::ij<DataVector, SpatialDim> conformal_metric_times_field_b_actual(
      used_for_size);
  tnsr::ijk<DataVector, SpatialDim>
      conformal_metric_times_symmetrized_d_field_b_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> conformal_metric_times_trace_a_tilde_actual(
      used_for_size);
  tnsr::i<DataVector, SpatialDim> inv_conformal_metric_times_d_a_tilde_actual(
      used_for_size);
  tnsr::I<DataVector, SpatialDim>
      gamma_hat_minus_contracted_conformal_christoffel_actual(used_for_size);
  tnsr::iJ<DataVector, SpatialDim>
      d_gamma_hat_minus_contracted_conformal_christoffel_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> contracted_christoffel_second_kind_actual(
      used_for_size);
  tnsr::ij<DataVector, SpatialDim>
      contracted_d_conformal_christoffel_difference_actual(used_for_size);
  Scalar<DataVector> k_minus_2_theta_c_actual(used_for_size);
  Scalar<DataVector> k_minus_k0_minus_2_theta_c_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> lapse_times_a_tilde_actual(used_for_size);
  tnsr::ijj<DataVector, SpatialDim> lapse_times_d_a_tilde_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> lapse_times_field_a_actual(used_for_size);
  tnsr::ii<DataVector, SpatialDim> lapse_times_conformal_spatial_metric_actual(
      used_for_size);
  Scalar<DataVector> lapse_times_slicing_condition_actual(used_for_size);
  Scalar<DataVector>
      lapse_times_ricci_scalar_plus_divergence_z4_constraint_actual(
          used_for_size);
  tnsr::I<DataVector, SpatialDim> shift_times_deriv_gamma_hat_actual(
      used_for_size);
  tnsr::ii<DataVector, SpatialDim> inv_tau_times_conformal_metric_actual(
      used_for_size);
  // expressions and identities needed for evolution equations: eq 13 - 27
  Scalar<DataVector> trace_a_tilde_actual(used_for_size);
  tnsr::iJJ<DataVector, SpatialDim> field_d_up_actual(used_for_size);
  tnsr::Ijj<DataVector, SpatialDim> conformal_christoffel_second_kind_actual(
      used_for_size);
  tnsr::iJkk<DataVector, SpatialDim> d_conformal_christoffel_second_kind_actual(
      used_for_size);
  tnsr::Ijj<DataVector, SpatialDim> christoffel_second_kind_actual(
      used_for_size);
  tnsr::ii<DataVector, SpatialDim> spatial_ricci_tensor_actual(used_for_size);
  tnsr::ij<DataVector, SpatialDim> grad_grad_lapse_actual(used_for_size);
  Scalar<DataVector> divergence_lapse_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim>
      contracted_conformal_christoffel_second_kind_actual(used_for_size);
  tnsr::iJ<DataVector, SpatialDim>
      d_contracted_conformal_christoffel_second_kind_actual(used_for_size);
  tnsr::i<DataVector, SpatialDim> spatial_z4_constraint_actual(used_for_size);
  tnsr::I<DataVector, SpatialDim> upper_spatial_z4_constraint_actual(
      used_for_size);
  tnsr::ij<DataVector, SpatialDim> grad_spatial_z4_constraint_actual(
      used_for_size);
  Scalar<DataVector> ricci_scalar_plus_divergence_z4_constraint_actual(
      used_for_size);

  ::Ccz4::TimeDerivative<SpatialDim>::apply(
      make_not_null(&dt_conformal_spatial_metric_actual),
      make_not_null(&dt_ln_lapse_actual), make_not_null(&dt_shift_actual),
      make_not_null(&dt_ln_conformal_factor_actual),
      make_not_null(&dt_a_tilde_actual),
      make_not_null(&dt_trace_extrinsic_curvature_actual),
      make_not_null(&dt_theta_actual), make_not_null(&dt_gamma_hat_actual),
      make_not_null(&dt_b_actual), make_not_null(&dt_field_a_actual),
      make_not_null(&dt_field_b_actual), make_not_null(&dt_field_d_actual),
      make_not_null(&dt_field_p_actual),
      make_not_null(&conformal_factor_squared_actual),
      make_not_null(&det_conformal_spatial_metric_actual),
      make_not_null(&inv_conformal_spatial_metric_actual),
      make_not_null(&inv_spatial_metric_actual), make_not_null(&lapse_actual),
      make_not_null(&slicing_condition_actual),
      make_not_null(&d_slicing_condition_actual),
      make_not_null(&inv_a_tilde_actual),
      make_not_null(&a_tilde_times_field_b_actual),
      make_not_null(
          &a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde_actual),
      make_not_null(&contracted_field_b_actual),
      make_not_null(&symmetrized_d_field_b_actual),
      make_not_null(&contracted_symmetrized_d_field_b_actual),
      make_not_null(&field_b_times_field_d_actual),
      make_not_null(&field_d_up_times_a_tilde_actual),
      make_not_null(&contracted_field_d_up_actual),
      make_not_null(&half_conformal_factor_squared_actual),
      make_not_null(&conformal_metric_times_field_b_actual),
      make_not_null(&conformal_metric_times_symmetrized_d_field_b_actual),
      make_not_null(&conformal_metric_times_trace_a_tilde_actual),
      make_not_null(&inv_conformal_metric_times_d_a_tilde_actual),
      make_not_null(&gamma_hat_minus_contracted_conformal_christoffel_actual),
      make_not_null(&d_gamma_hat_minus_contracted_conformal_christoffel_actual),
      make_not_null(&contracted_christoffel_second_kind_actual),
      make_not_null(&contracted_d_conformal_christoffel_difference_actual),
      make_not_null(&k_minus_2_theta_c_actual),
      make_not_null(&k_minus_k0_minus_2_theta_c_actual),
      make_not_null(&lapse_times_a_tilde_actual),
      make_not_null(&lapse_times_d_a_tilde_actual),
      make_not_null(&lapse_times_field_a_actual),
      make_not_null(&lapse_times_conformal_spatial_metric_actual),
      make_not_null(&lapse_times_slicing_condition_actual),
      make_not_null(
          &lapse_times_ricci_scalar_plus_divergence_z4_constraint_actual),
      make_not_null(&shift_times_deriv_gamma_hat_actual),
      make_not_null(&inv_tau_times_conformal_metric_actual),
      make_not_null(&trace_a_tilde_actual), make_not_null(&field_d_up_actual),
      make_not_null(&conformal_christoffel_second_kind_actual),
      make_not_null(&d_conformal_christoffel_second_kind_actual),
      make_not_null(&christoffel_second_kind_actual),
      make_not_null(&spatial_ricci_tensor_actual),
      make_not_null(&grad_grad_lapse_actual),
      make_not_null(&divergence_lapse_actual),
      make_not_null(&contracted_conformal_christoffel_second_kind_actual),
      make_not_null(&d_contracted_conformal_christoffel_second_kind_actual),
      make_not_null(&spatial_z4_constraint_actual),
      make_not_null(&upper_spatial_z4_constraint_actual),
      make_not_null(&grad_spatial_z4_constraint_actual),
      make_not_null(&ricci_scalar_plus_divergence_z4_constraint_actual), c,
      cleaning_speed, eta, f, k_0, d_k_0, kappa_1, kappa_2, kappa_3, mu,
      one_over_relaxation_time, evolve_shift, slicing_condition_type,
      conformal_spatial_metric, ln_lapse, shift, ln_conformal_factor, a_tilde,
      trace_extrinsic_curvature, theta, gamma_hat, b, field_a, field_b, field_d,
      field_p, d_a_tilde, d_trace_extrinsic_curvature, d_theta, d_gamma_hat,
      d_b, d_field_a, d_field_b, d_field_d, d_field_p);

  const auto zero = DataVector(used_for_size.size(), 0.0);

  // Check time derivatives eq (12a) - (12m)

  // eq 12a
  for (auto& component : dt_conformal_spatial_metric_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  // eq 12b
  for (auto& component : dt_ln_lapse_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  // eq 12c
  for (auto& component : dt_shift_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  // eq 12d
  for (auto& component : dt_ln_conformal_factor_actual) {
    CHECK_ITERABLE_APPROX(component, zero);
  }
  // eq 12e
  Approx approx_12e = Approx::custom().epsilon(1e-11).scale(1.0);
  for (auto& component : dt_a_tilde_actual) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, approx_12e);
  }
  // eq 12f
  Approx approx_12f = Approx::custom().epsilon(1e-11).scale(1.0);
  for (auto& component : dt_trace_extrinsic_curvature_actual) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, approx_12f);
  }
  // eq 12g
  Approx approx_12g = Approx::custom().epsilon(1e-11).scale(1.0);
  for (auto& component : dt_theta_actual) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, approx_12g);
  }
  // eq 12h
  // \partial_t \hat{\gamma}^i will not be 0 for KerrSchild if
  // evolve_shift == false
  const tnsr::I<DataVector, SpatialDim, FrameType> dt_gamma_hat_expected =
      get_dt_gamma_hat_kerr_expected(
          evolve_shift, inverse_conformal_spatial_metric, d_field_b);
  Approx approx_12h = Approx::custom().epsilon(1e-11).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_gamma_hat_actual, dt_gamma_hat_expected,
                               approx_12h);
  // eq 12i
  // \partial_t b will not be 0 for KerrSchild if evolve_shift == true
  const tnsr::I<DataVector, SpatialDim, FrameType> dt_b_expected =
      get_dt_b_kerr_expected(evolve_shift, eta, shift, d_gamma_hat, b, d_b);
  Approx approx_12i = Approx::custom().epsilon(1e-11).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_b_actual, dt_b_expected, approx_12i);
  // eq 12j
  Approx approx_12j = Approx::custom().epsilon(1e-11).scale(1.0);
  for (auto& component : dt_field_a_actual) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, approx_12j);
  }
  // eq 12k
  Approx approx_12k = Approx::custom().epsilon(1e-11).scale(1.0);
  for (auto& component : dt_field_b_actual) {
    CHECK_ITERABLE_CUSTOM_APPROX(component, zero, approx_12k);
  }
  // eq 12l
  // \partial_t D_{kij} will not be 0 for KerrSchild if evolve_shift == false
  const tnsr::ijj<DataVector, SpatialDim, FrameType> dt_field_d_expected =
      get_dt_field_d_kerr_expected(evolve_shift, conformal_spatial_metric,
                                   d_field_b);
  Approx approx_12l = Approx::custom().epsilon(1e-11).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_field_d_actual, dt_field_d_expected,
                               approx_12l);
  // eq 12m
  // \partial_t P_i will not be 0 for KerrSchild if evolve_shift == false
  const tnsr::i<DataVector, SpatialDim, FrameType> dt_field_p_expected =
      get_dt_field_p_kerr_expected(evolve_shift, d_field_b);
  Approx approx_12m = Approx::custom().epsilon(1e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(dt_field_p_actual, dt_field_p_expected,
                               approx_12m);
}

// Test first order CCZ4 against Minkowski and KerrSchild
void test(const Ccz4::EvolveShift evolve_shift,
          const Ccz4::SlicingConditionType slicing_condition_type) {
  test_minkowski(evolve_shift, slicing_condition_type);
  test_kerrschild(evolve_shift, slicing_condition_type);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.TimeDerivative",
                  "[Unit][Evolution]") {
  // Test first order CCZ4 with different settings
  test(Ccz4::EvolveShift::True, Ccz4::SlicingConditionType::Harmonic);
  test(Ccz4::EvolveShift::True, Ccz4::SlicingConditionType::Log);
  test(Ccz4::EvolveShift::False, Ccz4::SlicingConditionType::Harmonic);
  test(Ccz4::EvolveShift::False, Ccz4::SlicingConditionType::Log);
}
