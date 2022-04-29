// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/TimeDerivative.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"
#include "Evolution/Systems/Ccz4/DerivLapse.hpp"
#include "Evolution/Systems/Ccz4/DerivZ4Constraint.hpp"
#include "Evolution/Systems/Ccz4/Ricci.hpp"
#include "Evolution/Systems/Ccz4/RicciScalarPlusDivergenceZ4Constraint.hpp"
#include "Evolution/Systems/Ccz4/Z4Constraint.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Ccz4 {
template <size_t Dim>
void TimeDerivative<Dim>::apply(
    // LHS time derivatives of evolved variables: eq 12a - 12m
    const gsl::not_null<tnsr::ii<DataVector, Dim>*>
        dt_conformal_spatial_metric,                                  // eq 12a
    const gsl::not_null<Scalar<DataVector>*> dt_ln_lapse,             // eq 12b
    const gsl::not_null<tnsr::I<DataVector, Dim>*> dt_shift,          // eq 12c
    const gsl::not_null<Scalar<DataVector>*> dt_ln_conformal_factor,  // eq 12d
    const gsl::not_null<tnsr::ii<DataVector, Dim>*> dt_a_tilde,       // eq 12e
    const gsl::not_null<Scalar<DataVector>*>
        dt_trace_extrinsic_curvature,                             // eq 12f
    const gsl::not_null<Scalar<DataVector>*> dt_theta,            // eq 12g
    const gsl::not_null<tnsr::I<DataVector, Dim>*> dt_gamma_hat,  // eq 12h
    const gsl::not_null<tnsr::I<DataVector, Dim>*> dt_b,          // eq 12i
    const gsl::not_null<tnsr::i<DataVector, Dim>*> dt_field_a,    // eq 12j
    const gsl::not_null<tnsr::iJ<DataVector, Dim>*> dt_field_b,   // eq 12k
    const gsl::not_null<tnsr::ijj<DataVector, Dim>*> dt_field_d,  // eq 12l
    const gsl::not_null<tnsr::i<DataVector, Dim>*> dt_field_p,    // eq 12m
    // quantities we need for computing eq 12 - 27
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_squared,
    const gsl::not_null<Scalar<DataVector>*> det_conformal_spatial_metric,
    const gsl::not_null<tnsr::II<DataVector, Dim>*>
        inv_conformal_spatial_metric,
    const gsl::not_null<tnsr::II<DataVector, Dim>*> inv_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<Scalar<DataVector>*> slicing_condition,    // g(\alpha)
    const gsl::not_null<Scalar<DataVector>*> d_slicing_condition,  // g'(\alpha)
    const gsl::not_null<tnsr::II<DataVector, Dim>*> inv_a_tilde,
    // temporary expressions
    const gsl::not_null<tnsr::ij<DataVector, Dim>*> a_tilde_times_field_b,
    const gsl::not_null<tnsr::ii<DataVector, Dim>*>
        a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde,
    const gsl::not_null<Scalar<DataVector>*> contracted_field_b,
    const gsl::not_null<tnsr::ijK<DataVector, Dim>*> symmetrized_d_field_b,
    const gsl::not_null<tnsr::i<DataVector, Dim>*>
        contracted_symmetrized_d_field_b,
    const gsl::not_null<tnsr::ijk<DataVector, Dim>*> field_b_times_field_d,
    const gsl::not_null<tnsr::i<DataVector, Dim>*> field_d_up_times_a_tilde,
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        contracted_field_d_up,  // buffer for eq 18 -20
    const gsl::not_null<tnsr::ij<DataVector, Dim>*>
        conformal_metric_times_field_b,
    const gsl::not_null<tnsr::ijk<DataVector, Dim>*>
        conformal_metric_times_symmetrized_d_field_b,
    const gsl::not_null<tnsr::ii<DataVector, Dim>*>
        conformal_metric_times_trace_a_tilde,
    const gsl::not_null<tnsr::i<DataVector, Dim>*>
        inv_conformal_metric_times_d_a_tilde,
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        gamma_hat_minus_contracted_conformal_christoffel,
    const gsl::not_null<tnsr::iJ<DataVector, Dim>*>
        d_gamma_hat_minus_contracted_conformal_christoffel,
    const gsl::not_null<tnsr::i<DataVector, Dim>*>
        contracted_christoffel_second_kind,  // buffer for eq 18 -20
    const gsl::not_null<tnsr::ij<DataVector, Dim>*>
        contracted_d_conformal_christoffel_difference,  // buffer for eq 18 -20
    const gsl::not_null<Scalar<DataVector>*> k_minus_2_theta_c,
    const gsl::not_null<Scalar<DataVector>*> k_minus_k0_minus_2_theta_c,
    const gsl::not_null<tnsr::ii<DataVector, Dim>*> lapse_times_a_tilde,
    const gsl::not_null<tnsr::ijj<DataVector, Dim>*> lapse_times_d_a_tilde,
    const gsl::not_null<tnsr::i<DataVector, Dim>*> lapse_times_field_a,
    const gsl::not_null<tnsr::ii<DataVector, Dim>*>
        lapse_times_conformal_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> lapse_times_slicing_condition,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_ricci_scalar_plus_divergence_z4_constraint,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> shift_times_deriv_gamma_hat,
    const gsl::not_null<tnsr::ii<DataVector, Dim>*>
        inv_tau_times_conformal_metric,
    // expressions and identities needed for evolution equations: eq 13 - 27
    const gsl::not_null<Scalar<DataVector>*> trace_a_tilde,       // eq 13
    const gsl::not_null<tnsr::iJJ<DataVector, Dim>*> field_d_up,  // eq 14
    const gsl::not_null<tnsr::Ijj<DataVector, Dim>*>
        conformal_christoffel_second_kind,  // eq 15
    const gsl::not_null<tnsr::iJkk<DataVector, Dim>*>
        d_conformal_christoffel_second_kind,  // eq 16
    const gsl::not_null<tnsr::Ijj<DataVector, Dim>*>
        christoffel_second_kind,  // eq 17
    const gsl::not_null<tnsr::ii<DataVector, Dim>*>
        spatial_ricci_tensor,  // eq 18 - 20
    const gsl::not_null<tnsr::ij<DataVector, Dim>*> grad_grad_lapse,  // eq 21
    const gsl::not_null<Scalar<DataVector>*> divergence_lapse,        // eq 22
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        contracted_conformal_christoffel_second_kind,  // eq 23
    const gsl::not_null<tnsr::iJ<DataVector, Dim>*>
        d_contracted_conformal_christoffel_second_kind,  // eq 24
    const gsl::not_null<tnsr::i<DataVector, Dim>*>
        spatial_z4_constraint,  // eq 25
    const gsl::not_null<Scalar<DataVector>*>
        upper_spatial_z4_constraint_buffer,  // buffer for eq 25
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        upper_spatial_z4_constraint,  // eq 25
    const gsl::not_null<tnsr::ij<DataVector, Dim>*>
        grad_spatial_z4_constraint,  // eq 26
    const gsl::not_null<Scalar<DataVector>*>
        ricci_scalar_plus_divergence_z4_constraint,  // eq 27
    // free params
    const double c, const double cleaning_speed,  // e in the paper
    const Scalar<DataVector>& eta, const double f,
    const Scalar<DataVector>& k_0, const tnsr::i<DataVector, Dim>& d_k_0,
    const double kappa_1, const double kappa_2, const double kappa_3,
    const double mu, const double one_over_relaxation_time,  // \tau^{-1}
    const EvolveShift evolve_shift,
    const SlicingConditionType slicing_condition_type,
    // evolved variables
    const tnsr::ii<DataVector, Dim>& conformal_spatial_metric,
    const Scalar<DataVector>& ln_lapse, const tnsr::I<DataVector, Dim>& shift,
    const Scalar<DataVector>& ln_conformal_factor,
    const tnsr::ii<DataVector, Dim>& a_tilde,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& theta, const tnsr::I<DataVector, Dim>& gamma_hat,
    const tnsr::I<DataVector, Dim>& b, const tnsr::i<DataVector, Dim>& field_a,
    const tnsr::iJ<DataVector, Dim>& field_b,
    const tnsr::ijj<DataVector, Dim>& field_d,
    const tnsr::i<DataVector, Dim>& field_p,
    // spatial derivatives of evolved variables
    const tnsr::ijj<DataVector, Dim>& d_a_tilde,
    const tnsr::i<DataVector, Dim>& d_trace_extrinsic_curvature,
    const tnsr::i<DataVector, Dim>& d_theta,
    const tnsr::iJ<DataVector, Dim>& d_gamma_hat,
    const tnsr::iJ<DataVector, Dim>& d_b,
    const tnsr::ij<DataVector, Dim>& d_field_a,
    const tnsr::ijK<DataVector, Dim>& d_field_b,
    const tnsr::ijkk<DataVector, Dim>& d_field_d,
    const tnsr::ij<DataVector, Dim>& d_field_p) {
  ASSERT(
      evolve_shift == EvolveShift::True or evolve_shift == EvolveShift::False,
      "Unknown Ccz4::EvolveShift.");

  constexpr double one_third = 1.0 / 3.0;

  // quantities we need for computing eq 12 - 27

  determinant_and_inverse(det_conformal_spatial_metric,
                          inv_conformal_spatial_metric,
                          conformal_spatial_metric);

  const size_t num_points = get_size(get(ln_conformal_factor));
  for (size_t i = 0; i < num_points; i++) {
    get(*conformal_factor_squared)[i] = exp(2.0 * get(ln_conformal_factor)[i]);
  }

  ::tenex::evaluate<ti_I, ti_J>(
      inv_spatial_metric, (*conformal_factor_squared)() *
                              (*inv_conformal_spatial_metric)(ti_I, ti_J));

  ::tenex::evaluate<ti_I, ti_J>(
      inv_a_tilde, a_tilde(ti_k, ti_l) *
                       (*inv_conformal_spatial_metric)(ti_I, ti_K) *
                       (*inv_conformal_spatial_metric)(ti_J, ti_L));

  for (size_t i = 0; i < num_points; i++) {
    get(*lapse)[i] = exp(get(ln_lapse)[i]);
  }

  if (slicing_condition_type == SlicingConditionType::Harmonic) {
    get(*slicing_condition) = 1.0;
    get(*d_slicing_condition) = 0.0;
    get(*lapse_times_slicing_condition) = get(*lapse);
  } else if (slicing_condition_type == SlicingConditionType::Log) {
    ASSERT(min(get(*lapse)) > 0.0,
           "The lapse must be positive when using "
           "Ccz4::SlicingConditionType::Harmonic.");

    // g(\alpha) == 2 / \alpha
    get(*slicing_condition) = 2.0 / get(*lapse);
    // g'(\alpha)  == -2 / \alpha^2 == -0.5 * g(\alpha)^2
    get(*d_slicing_condition) = -0.5 * square(get(*slicing_condition));
    // \alpha g(\alpha)  == 2
    get(*lapse_times_slicing_condition) = 2.0;
  } else {
    ERROR("Unknown Ccz4::SlicingConditionType");  // LCOV_EXCL_LINE
  }

  // expressions and identities needed for evolution equations: eq 13 - 27

  // eq 13
  ::tenex::evaluate(trace_a_tilde, (*inv_conformal_spatial_metric)(ti_I, ti_J) *
                                       a_tilde(ti_i, ti_j));

  // eq 14
  ::tenex::evaluate<ti_k, ti_I, ti_J>(
      field_d_up, (*inv_conformal_spatial_metric)(ti_I, ti_N) *
                      (*inv_conformal_spatial_metric)(ti_M, ti_J) *
                      field_d(ti_k, ti_n, ti_m));

  // eq 15
  ::Ccz4::conformal_christoffel_second_kind(conformal_christoffel_second_kind,
                                            *inv_conformal_spatial_metric,
                                            field_d);

  // eq 16
  ::Ccz4::deriv_conformal_christoffel_second_kind(
      d_conformal_christoffel_second_kind, *inv_conformal_spatial_metric,
      field_d, d_field_d, *field_d_up);

  // eq 17
  ::Ccz4::christoffel_second_kind(christoffel_second_kind,
                                  conformal_spatial_metric,
                                  *inv_conformal_spatial_metric, field_p,
                                  *conformal_christoffel_second_kind);

  // temporary expressions needed for eq 18 - 20
  ::tenex::evaluate<ti_l>(contracted_christoffel_second_kind,
                          (*christoffel_second_kind)(ti_M, ti_l, ti_m));

  ::tenex::evaluate<ti_i, ti_j>(
      contracted_d_conformal_christoffel_difference,
      (*d_conformal_christoffel_second_kind)(ti_m, ti_M, ti_i, ti_j) -
          (*d_conformal_christoffel_second_kind)(ti_j, ti_M, ti_i, ti_m));

  ::tenex::evaluate<ti_L>(contracted_field_d_up,
                          (*field_d_up)(ti_m, ti_M, ti_L));

  // eq 18 - 20
  ::Ccz4::spatial_ricci_tensor(
      spatial_ricci_tensor, *christoffel_second_kind,
      *contracted_christoffel_second_kind,
      *contracted_d_conformal_christoffel_difference, conformal_spatial_metric,
      *inv_conformal_spatial_metric, field_d, *field_d_up,
      *contracted_field_d_up, field_p, d_field_p);

  // eq 21
  ::Ccz4::grad_grad_lapse(grad_grad_lapse, *lapse, *christoffel_second_kind,
                          field_a, d_field_a);

  // eq 22
  ::Ccz4::divergence_lapse(divergence_lapse, *conformal_factor_squared,
                           *inv_conformal_spatial_metric, *grad_grad_lapse);

  // eq 23
  ::Ccz4::contracted_conformal_christoffel_second_kind(
      contracted_conformal_christoffel_second_kind,
      *inv_conformal_spatial_metric, *conformal_christoffel_second_kind);

  // eq 24
  ::Ccz4::deriv_contracted_conformal_christoffel_second_kind(
      d_contracted_conformal_christoffel_second_kind,
      *inv_conformal_spatial_metric, *field_d_up,
      *conformal_christoffel_second_kind, *d_conformal_christoffel_second_kind);

  // temp for eq 25
  ::tenex::evaluate<ti_I>(
      gamma_hat_minus_contracted_conformal_christoffel,
      gamma_hat(ti_I) - (*contracted_conformal_christoffel_second_kind)(ti_I));

  // eq 25
  ::Ccz4::spatial_z4_constraint(
      spatial_z4_constraint, conformal_spatial_metric,
      *gamma_hat_minus_contracted_conformal_christoffel);

  // eq 25
  ::Ccz4::upper_spatial_z4_constraint(
      upper_spatial_z4_constraint, upper_spatial_z4_constraint_buffer,
      *conformal_factor_squared,
      *gamma_hat_minus_contracted_conformal_christoffel);

  // temp for eq 26
  ::tenex::evaluate<ti_i, ti_L>(
      d_gamma_hat_minus_contracted_conformal_christoffel,
      d_gamma_hat(ti_i, ti_L) -
          (*d_contracted_conformal_christoffel_second_kind)(ti_i, ti_L));

  // eq 26
  ::Ccz4::grad_spatial_z4_constraint(
      grad_spatial_z4_constraint, *spatial_z4_constraint,
      conformal_spatial_metric, *christoffel_second_kind, field_d,
      *gamma_hat_minus_contracted_conformal_christoffel,
      *d_gamma_hat_minus_contracted_conformal_christoffel);

  // eq 27
  ::Ccz4::ricci_scalar_plus_divergence_z4_constraint(
      ricci_scalar_plus_divergence_z4_constraint, *conformal_factor_squared,
      *inv_conformal_spatial_metric, *spatial_ricci_tensor,
      *grad_spatial_z4_constraint);

  // temporary expressions not already computed above

  ::tenex::evaluate(contracted_field_b, field_b(ti_k, ti_K));

  ::tenex::evaluate<ti_k, ti_j, ti_I>(
      symmetrized_d_field_b,
      0.5 * (d_field_b(ti_k, ti_j, ti_I) + d_field_b(ti_j, ti_k, ti_I)));

  for (size_t k = 0; k < Dim; k++) {
    contracted_symmetrized_d_field_b->get(k) = d_field_b.get(k, 0, 0);
    for (size_t i = 1; i < Dim; i++) {
      contracted_symmetrized_d_field_b->get(k) += d_field_b.get(k, i, i);
    }
  }

  ::tenex::evaluate<ti_i, ti_j, ti_k>(
      field_b_times_field_d, field_b(ti_i, ti_L) * field_d(ti_j, ti_l, ti_k));

  ::tenex::evaluate<ti_k>(
      field_d_up_times_a_tilde,
      (*field_d_up)(ti_k, ti_I, ti_J) * a_tilde(ti_i, ti_j));

  ::tenex::evaluate<ti_i, ti_j>(
      conformal_metric_times_field_b,
      conformal_spatial_metric(ti_k, ti_i) * field_b(ti_j, ti_K));

  ::tenex::evaluate<ti_i, ti_k, ti_j>(
      conformal_metric_times_symmetrized_d_field_b,
      conformal_spatial_metric(ti_m, ti_i) *
          (*symmetrized_d_field_b)(ti_k, ti_j, ti_M));

  ::tenex::evaluate<ti_i, ti_j>(
      conformal_metric_times_trace_a_tilde,
      conformal_spatial_metric(ti_i, ti_j) * (*trace_a_tilde)());

  ::tenex::evaluate<ti_k>(inv_conformal_metric_times_d_a_tilde,
                          (*inv_conformal_spatial_metric)(ti_I, ti_J) *
                              d_a_tilde(ti_k, ti_i, ti_j));

  ::tenex::evaluate<ti_i, ti_j>(a_tilde_times_field_b,
                                a_tilde(ti_k, ti_i) * field_b(ti_j, ti_K));

  ::tenex::evaluate<ti_i, ti_j>(
      a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde,
      a_tilde(ti_i, ti_j) -
          one_third * (*conformal_metric_times_trace_a_tilde)(ti_i, ti_j));

  ::tenex::evaluate(k_minus_2_theta_c,
                    trace_extrinsic_curvature() - 2.0 * c * theta());

  ::tenex::evaluate(k_minus_k0_minus_2_theta_c, (*k_minus_2_theta_c)() - k_0());

  ::tenex::evaluate<ti_i, ti_j>(lapse_times_a_tilde,
                                (*lapse)() * a_tilde(ti_i, ti_j));

  tenex::evaluate<ti_k, ti_i, ti_j>(lapse_times_d_a_tilde,
                                    (*lapse)() * d_a_tilde(ti_k, ti_i, ti_j));

  ::tenex::evaluate<ti_k>(lapse_times_field_a, (*lapse)() * field_a(ti_k));

  ::tenex::evaluate<ti_i, ti_j>(
      lapse_times_conformal_spatial_metric,
      (*lapse)() * conformal_spatial_metric(ti_i, ti_j));

  ::tenex::evaluate(
      lapse_times_ricci_scalar_plus_divergence_z4_constraint,
      (*lapse)() * (*ricci_scalar_plus_divergence_z4_constraint)());

  ::tenex::evaluate<ti_I>(shift_times_deriv_gamma_hat,
                          shift(ti_K) * d_gamma_hat(ti_k, ti_I));

  ::tenex::evaluate<ti_i, ti_j>(
      inv_tau_times_conformal_metric,
      one_over_relaxation_time * conformal_spatial_metric(ti_i, ti_j));

  // time derivative computation: eq 12a - 12m

  // eq 12a : time derivative of the conformal spatial metric
  ::tenex::evaluate<ti_i, ti_j>(
      dt_conformal_spatial_metric,
      2.0 * shift(ti_K) * field_d(ti_k, ti_i, ti_j) +
          (*conformal_metric_times_field_b)(ti_i, ti_j) +
          (*conformal_metric_times_field_b)(ti_j, ti_i) -
          2.0 * one_third * conformal_spatial_metric(ti_i, ti_j) *
              (*contracted_field_b)() -
          2.0 * (*lapse)() *
              (*a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde)(
                  ti_i, ti_j) -
          (*inv_tau_times_conformal_metric)(ti_i, ti_j) *
              ((*det_conformal_spatial_metric)() - 1.0));

  // eq 12b : time derivative of the natural log of the lapse
  ::tenex::evaluate(dt_ln_lapse, shift(ti_K) * field_a(ti_k) -
                                     (*lapse_times_slicing_condition)() *
                                         (*k_minus_k0_minus_2_theta_c)());

  // eq 12c : time derivative of the shift
  // if s == 0
  if (not static_cast<bool>(evolve_shift)) {
    for (auto& component : *dt_shift) {
      component = 0.0;
    }
  } else {
    ::tenex::evaluate<ti_I>(dt_shift,
                            f * b(ti_I) + shift(ti_K) * field_b(ti_k, ti_I));
  }

  // eq 12d : time derivative of the natural log of the conformal factor
  ::tenex::evaluate(dt_ln_conformal_factor,
                    shift(ti_K) * field_p(ti_k) +
                        one_third * ((*lapse)() * trace_extrinsic_curvature() -
                                     (*contracted_field_b)()));

  // eq 12e : time derivative of the trace-free part of the extrinsic curvature
  ::tenex::evaluate<ti_i, ti_j>(
      dt_a_tilde,
      shift(ti_K) * d_a_tilde(ti_k, ti_i, ti_j) +
          (*conformal_factor_squared)() *
              ((*lapse)() * ((*spatial_ricci_tensor)(ti_i, ti_j) +
                             (*grad_spatial_z4_constraint)(ti_i, ti_j) +
                             (*grad_spatial_z4_constraint)(ti_j, ti_i)) -
               (*grad_grad_lapse)(ti_i, ti_j)) -
          one_third * conformal_spatial_metric(ti_i, ti_j) *
              ((*lapse_times_ricci_scalar_plus_divergence_z4_constraint)() -
               (*divergence_lapse)()) +
          (*a_tilde_times_field_b)(ti_i, ti_j) +
          (*a_tilde_times_field_b)(ti_j, ti_i) -
          2.0 * one_third * a_tilde(ti_i, ti_j) * (*contracted_field_b)() +
          (*lapse_times_a_tilde)(ti_i, ti_j) * (*k_minus_2_theta_c)() -
          2.0 * (*lapse_times_a_tilde)(ti_i, ti_l) *
              (*inv_conformal_spatial_metric)(ti_L, ti_M) *
              a_tilde(ti_m, ti_j) -
          (*inv_tau_times_conformal_metric)(ti_i, ti_j) * (*trace_a_tilde)());

  // eq. (12f) : time derivative of the trace of the extrinsic curvature
  ::tenex::evaluate(
      dt_trace_extrinsic_curvature,
      shift(ti_K) * d_trace_extrinsic_curvature(ti_k) - (*divergence_lapse)() +
          (*lapse_times_ricci_scalar_plus_divergence_z4_constraint)() +
          (*lapse)() * (trace_extrinsic_curvature() * (*k_minus_2_theta_c)() -
                        3.0 * kappa_1 * (1.0 + kappa_2) * theta()));

  // eq. (12g) : time derivative of the projection of the Z4 four-vector along
  // the normal direction
  ::tenex::evaluate(
      dt_theta,
      shift(ti_K) * d_theta(ti_k) +
          (*lapse)() *
              (0.5 * square(cleaning_speed) *
                   ((*ricci_scalar_plus_divergence_z4_constraint)() +
                    2.0 * one_third * square(trace_extrinsic_curvature()) -
                    a_tilde(ti_i, ti_j) * (*inv_a_tilde)(ti_I, ti_J)) -
               c * theta() * trace_extrinsic_curvature() -
               (*upper_spatial_z4_constraint)(ti_I)*field_a(ti_i) -
               kappa_1 * (2.0 + kappa_2) * theta()));

  // eq. (12h) : time derivative \hat{\Gamma}^i
  // first, compute terms without s
  ::tenex::evaluate<ti_I>(
      dt_gamma_hat,
      // terms without lapse nor s
      (*shift_times_deriv_gamma_hat)(ti_I) +
          2.0 * one_third *
              (*contracted_conformal_christoffel_second_kind)(ti_I) *
              (*contracted_field_b)() -
          (*contracted_conformal_christoffel_second_kind)(ti_K)*field_b(ti_k,
                                                                        ti_I) +
          2.0 * kappa_3 * (*spatial_z4_constraint)(ti_j) *
              (2.0 * one_third * (*inv_conformal_spatial_metric)(ti_I, ti_J) *
                   (*contracted_field_b)() -
               (*inv_conformal_spatial_metric)(ti_J, ti_K) *
                   field_b(ti_k, ti_I)) +
          // terms with lapse but not s
          2.0 * (*lapse)() *
              (-2.0 * one_third * (*inv_conformal_spatial_metric)(ti_I, ti_J) *
                   d_trace_extrinsic_curvature(ti_j) +
               (*inv_conformal_spatial_metric)(ti_K, ti_I) * d_theta(ti_k) +
               (*conformal_christoffel_second_kind)(ti_I, ti_j, ti_k) *
                   (*inv_a_tilde)(ti_J, ti_K) -
               3.0 * (*inv_a_tilde)(ti_I, ti_J) * field_p(ti_j) -
               (*inv_conformal_spatial_metric)(ti_K, ti_I) *
                   (theta() * field_a(ti_k) +
                    2.0 * one_third * trace_extrinsic_curvature() *
                        (*spatial_z4_constraint)(ti_k)) -
               (*inv_a_tilde)(ti_I, ti_J) * field_a(ti_j) -
               kappa_1 * (*inv_conformal_spatial_metric)(ti_I, ti_J) *
                   (*spatial_z4_constraint)(ti_j)));
  // now, if s == 1, also add terms with s
  if (static_cast<bool>(evolve_shift)) {
    ::tenex::evaluate<ti_I>(
        dt_gamma_hat,
        (*dt_gamma_hat)(ti_I) +
            // terms with lapse and s
            2.0 * (*lapse)() *
                ((*inv_conformal_spatial_metric)(ti_I, ti_K) *
                     (*inv_conformal_spatial_metric)(ti_N, ti_M) *
                     d_a_tilde(ti_k, ti_n, ti_m) -
                 2.0 * (*inv_conformal_spatial_metric)(ti_I, ti_K) *
                     (*field_d_up)(ti_k, ti_N, ti_M) * a_tilde(ti_n, ti_m)) +
            // terms with s but not not lapse
            (*inv_conformal_spatial_metric)(ti_K, ti_L) *
                (*symmetrized_d_field_b)(ti_k, ti_l, ti_I) +
            one_third * (*inv_conformal_spatial_metric)(ti_I, ti_K) *
                (*contracted_symmetrized_d_field_b)(ti_k));
  }

  // eq. (12i) : time derivative b^i
  // if s == 0
  if (not static_cast<bool>(evolve_shift)) {
    for (auto& component : *dt_b) {
      component = 0.0;
    }
  } else {
    ::tenex::evaluate<ti_I>(
        dt_b, (*dt_gamma_hat)(ti_I)-eta() * b(ti_I) +
                  shift(ti_K) * (d_b(ti_k, ti_I) - d_gamma_hat(ti_k, ti_I)));
  }

  // eq. (12j) : time derivative of auxiliary variable A_i
  // first, compute terms without s
  ::tenex::evaluate<ti_k>(
      dt_field_a,
      shift(ti_L) * d_field_a(ti_l, ti_k) -
          (*lapse_times_field_a)(ti_k) * (*k_minus_k0_minus_2_theta_c)() *
              ((*slicing_condition)() + (*lapse)() * (*d_slicing_condition)()) +
          field_b(ti_k, ti_L) * field_a(ti_l) -
          (*lapse_times_slicing_condition)() *
              (d_trace_extrinsic_curvature(ti_k) - d_k_0(ti_k) -
               2.0 * c * d_theta(ti_k)));
  // now, if s == 1, also add terms with s
  if (static_cast<bool>(evolve_shift)) {
    ::tenex::evaluate<ti_k>(
        dt_field_a, (*dt_field_a)(ti_k) -
                        (*lapse_times_slicing_condition)() *
                            ((*inv_conformal_metric_times_d_a_tilde)(ti_k) -
                             (2.0 * (*field_d_up_times_a_tilde)(ti_k))));
  }

  // eq. (12k) : time derivative of auxiliary variable B_k{}^i
  // if s == 0
  if (not static_cast<bool>(evolve_shift)) {
    for (auto& component : *dt_field_b) {
      component = 0.0;
    }
  } else {
    // first, compute expression without advective terms
    ::tenex::evaluate<ti_k, ti_I>(
        dt_field_b, shift(ti_L) * d_field_b(ti_l, ti_k, ti_I) +
                        f * d_b(ti_k, ti_I) +
                        mu * square((*lapse)()) *
                            (*inv_conformal_spatial_metric)(ti_I, ti_J) *
                            (d_field_p(ti_k, ti_j) - d_field_p(ti_j, ti_k) -
                             (*inv_conformal_spatial_metric)(ti_N, ti_L) *
                                 (d_field_d(ti_k, ti_l, ti_j, ti_n) -
                                  d_field_d(ti_l, ti_k, ti_j, ti_n))) +
                        field_b(ti_k, ti_L) * field_b(ti_l, ti_I));
  }

  // eq. (12l) : time derivative of auxiliary variable D_{kij}
  // first, compute terms without s
  ::tenex::evaluate<ti_k, ti_i, ti_j>(
      dt_field_d,
      shift(ti_L) * d_field_d(ti_l, ti_k, ti_i, ti_j) -
          (*lapse_times_d_a_tilde)(ti_k, ti_i, ti_j) +
          field_b(ti_k, ti_L) * field_d(ti_l, ti_i, ti_j) +
          (*field_b_times_field_d)(ti_j, ti_k, ti_i) +
          (*field_b_times_field_d)(ti_i, ti_k, ti_j) -
          (*lapse_times_field_a)(ti_k) *
              (*a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde)(
                  ti_i, ti_j) +
          one_third *
              ((*lapse_times_conformal_spatial_metric)(ti_i, ti_j) *
                   (*inv_conformal_metric_times_d_a_tilde)(ti_k) -
               (2.0 * (*contracted_field_b)() * field_d(ti_k, ti_i, ti_j)) -
               2.0 * (*lapse_times_conformal_spatial_metric)(ti_i, ti_j) *
                   (*field_d_up_times_a_tilde)(ti_k)));
  // now, if s == 1, also add terms with s
  if (static_cast<bool>(evolve_shift)) {
    ::tenex::evaluate<ti_k, ti_i, ti_j>(
        dt_field_d, (*dt_field_d)(ti_k, ti_i, ti_j) +
                        0.5 * ((*conformal_metric_times_symmetrized_d_field_b)(
                                   ti_i, ti_k, ti_j) +
                               (*conformal_metric_times_symmetrized_d_field_b)(
                                   ti_j, ti_k, ti_i)) -
                        one_third * conformal_spatial_metric(ti_i, ti_j) *
                            (*contracted_symmetrized_d_field_b)(ti_k));
  }

  // eq. (12m) : time derivative of auxiliary variable P_i
  // first, compute terms without s
  ::tenex::evaluate<ti_k>(
      dt_field_p, shift(ti_L) * d_field_p(ti_l, ti_k) +
                      field_b(ti_k, ti_L) * field_p(ti_l) +
                      one_third * (*lapse)() *
                          (d_trace_extrinsic_curvature(ti_k) +
                           field_a(ti_k) * trace_extrinsic_curvature()));
  // now, if s == 1, also add terms with s
  if (static_cast<bool>(evolve_shift)) {
    ::tenex::evaluate<ti_k>(
        dt_field_p,
        (*dt_field_p)(ti_k) +
            one_third *
                ((*lapse)() * ((*inv_conformal_metric_times_d_a_tilde)(ti_k) -
                               (2.0 * (*field_d_up_times_a_tilde)(ti_k))) -
                 (*contracted_symmetrized_d_field_b)(ti_k)));
  }
}
}  // namespace Ccz4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data) template struct Ccz4::TimeDerivative<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
