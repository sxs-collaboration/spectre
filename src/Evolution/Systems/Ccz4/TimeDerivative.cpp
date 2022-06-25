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
        contracted_field_d_up,  // temp for eq 18 -20
    const gsl::not_null<Scalar<DataVector>*>
        half_conformal_factor_squared,  // temp for eq 25
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
        contracted_christoffel_second_kind,  // temp for eq 18 -20
    const gsl::not_null<tnsr::ij<DataVector, Dim>*>
        contracted_d_conformal_christoffel_difference,  // temp for eq 18 -20
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

  ::tenex::evaluate<ti::I, ti::J>(
      inv_spatial_metric, (*conformal_factor_squared)() *
                              (*inv_conformal_spatial_metric)(ti::I, ti::J));

  ::tenex::evaluate<ti::I, ti::J>(
      inv_a_tilde, a_tilde(ti::k, ti::l) *
                       (*inv_conformal_spatial_metric)(ti::I, ti::K) *
                       (*inv_conformal_spatial_metric)(ti::J, ti::L));

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
  ::tenex::evaluate(
      trace_a_tilde,
      (*inv_conformal_spatial_metric)(ti::I, ti::J) * a_tilde(ti::i, ti::j));

  // eq 14
  ::tenex::evaluate<ti::k, ti::I, ti::J>(
      field_d_up, (*inv_conformal_spatial_metric)(ti::I, ti::N) *
                      (*inv_conformal_spatial_metric)(ti::M, ti::J) *
                      field_d(ti::k, ti::n, ti::m));

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
  ::tenex::evaluate<ti::l>(contracted_christoffel_second_kind,
                           (*christoffel_second_kind)(ti::M, ti::l, ti::m));

  ::tenex::evaluate<ti::i, ti::j>(
      contracted_d_conformal_christoffel_difference,
      (*d_conformal_christoffel_second_kind)(ti::m, ti::M, ti::i, ti::j) -
          (*d_conformal_christoffel_second_kind)(ti::j, ti::M, ti::i, ti::m));

  ::tenex::evaluate<ti::L>(contracted_field_d_up,
                           (*field_d_up)(ti::m, ti::M, ti::L));

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
  ::tenex::evaluate<ti::I>(
      gamma_hat_minus_contracted_conformal_christoffel,
      gamma_hat(ti::I) -
          (*contracted_conformal_christoffel_second_kind)(ti::I));

  // eq 25
  ::Ccz4::spatial_z4_constraint(
      spatial_z4_constraint, conformal_spatial_metric,
      *gamma_hat_minus_contracted_conformal_christoffel);

  // temp for eq 25
  ::tenex::evaluate(half_conformal_factor_squared,
                    0.5 * (*conformal_factor_squared)());

  // eq 25
  ::Ccz4::upper_spatial_z4_constraint(
      upper_spatial_z4_constraint, *half_conformal_factor_squared,
      *gamma_hat_minus_contracted_conformal_christoffel);

  // temp for eq 26
  ::tenex::evaluate<ti::i, ti::L>(
      d_gamma_hat_minus_contracted_conformal_christoffel,
      d_gamma_hat(ti::i, ti::L) -
          (*d_contracted_conformal_christoffel_second_kind)(ti::i, ti::L));

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

  ::tenex::evaluate(contracted_field_b, field_b(ti::k, ti::K));

  ::tenex::evaluate<ti::k, ti::j, ti::I>(
      symmetrized_d_field_b,
      0.5 * (d_field_b(ti::k, ti::j, ti::I) + d_field_b(ti::j, ti::k, ti::I)));

  ::tenex::evaluate<ti::k>(contracted_symmetrized_d_field_b,
                           (*symmetrized_d_field_b)(ti::k, ti::i, ti::I));

  ::tenex::evaluate<ti::i, ti::j, ti::k>(
      field_b_times_field_d,
      field_b(ti::i, ti::L) * field_d(ti::j, ti::l, ti::k));

  ::tenex::evaluate<ti::k>(
      field_d_up_times_a_tilde,
      (*field_d_up)(ti::k, ti::I, ti::J) * a_tilde(ti::i, ti::j));

  ::tenex::evaluate<ti::i, ti::j>(
      conformal_metric_times_field_b,
      conformal_spatial_metric(ti::k, ti::i) * field_b(ti::j, ti::K));

  ::tenex::evaluate<ti::i, ti::k, ti::j>(
      conformal_metric_times_symmetrized_d_field_b,
      conformal_spatial_metric(ti::m, ti::i) *
          (*symmetrized_d_field_b)(ti::k, ti::j, ti::M));

  ::tenex::evaluate<ti::i, ti::j>(
      conformal_metric_times_trace_a_tilde,
      conformal_spatial_metric(ti::i, ti::j) * (*trace_a_tilde)());

  ::tenex::evaluate<ti::k>(inv_conformal_metric_times_d_a_tilde,
                           (*inv_conformal_spatial_metric)(ti::I, ti::J) *
                               d_a_tilde(ti::k, ti::i, ti::j));

  ::tenex::evaluate<ti::i, ti::j>(
      a_tilde_times_field_b, a_tilde(ti::k, ti::i) * field_b(ti::j, ti::K));

  ::tenex::evaluate<ti::i, ti::j>(
      a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde,
      a_tilde(ti::i, ti::j) -
          one_third * (*conformal_metric_times_trace_a_tilde)(ti::i, ti::j));

  ::tenex::evaluate(k_minus_2_theta_c,
                    trace_extrinsic_curvature() - 2.0 * c * theta());

  ::tenex::evaluate(k_minus_k0_minus_2_theta_c, (*k_minus_2_theta_c)() - k_0());

  ::tenex::evaluate<ti::i, ti::j>(lapse_times_a_tilde,
                                  (*lapse)() * a_tilde(ti::i, ti::j));

  tenex::evaluate<ti::k, ti::i, ti::j>(
      lapse_times_d_a_tilde, (*lapse)() * d_a_tilde(ti::k, ti::i, ti::j));

  ::tenex::evaluate<ti::k>(lapse_times_field_a, (*lapse)() * field_a(ti::k));

  ::tenex::evaluate<ti::i, ti::j>(
      lapse_times_conformal_spatial_metric,
      (*lapse)() * conformal_spatial_metric(ti::i, ti::j));

  ::tenex::evaluate(
      lapse_times_ricci_scalar_plus_divergence_z4_constraint,
      (*lapse)() * (*ricci_scalar_plus_divergence_z4_constraint)());

  ::tenex::evaluate<ti::I>(shift_times_deriv_gamma_hat,
                           shift(ti::K) * d_gamma_hat(ti::k, ti::I));

  ::tenex::evaluate<ti::i, ti::j>(
      inv_tau_times_conformal_metric,
      one_over_relaxation_time * conformal_spatial_metric(ti::i, ti::j));

  // time derivative computation: eq 12a - 12m

  // eq 12a : time derivative of the conformal spatial metric
  ::tenex::evaluate<ti::i, ti::j>(
      dt_conformal_spatial_metric,
      2.0 * shift(ti::K) * field_d(ti::k, ti::i, ti::j) +
          (*conformal_metric_times_field_b)(ti::i, ti::j) +
          (*conformal_metric_times_field_b)(ti::j, ti::i) -
          2.0 * one_third * conformal_spatial_metric(ti::i, ti::j) *
              (*contracted_field_b)() -
          2.0 * (*lapse)() *
              (*a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde)(
                  ti::i, ti::j) -
          (*inv_tau_times_conformal_metric)(ti::i, ti::j) *
              ((*det_conformal_spatial_metric)() - 1.0));

  // eq 12b : time derivative of the natural log of the lapse
  ::tenex::evaluate(dt_ln_lapse, shift(ti::K) * field_a(ti::k) -
                                     (*lapse_times_slicing_condition)() *
                                         (*k_minus_k0_minus_2_theta_c)());

  // eq 12c : time derivative of the shift
  // if s == 0
  if (not static_cast<bool>(evolve_shift)) {
    for (auto& component : *dt_shift) {
      component = 0.0;
    }
  } else {
    ::tenex::evaluate<ti::I>(
        dt_shift, f * b(ti::I) + shift(ti::K) * field_b(ti::k, ti::I));
  }

  // eq 12d : time derivative of the natural log of the conformal factor
  ::tenex::evaluate(dt_ln_conformal_factor,
                    shift(ti::K) * field_p(ti::k) +
                        one_third * ((*lapse)() * trace_extrinsic_curvature() -
                                     (*contracted_field_b)()));

  // eq 12e : time derivative of the trace-free part of the extrinsic curvature
  ::tenex::evaluate<ti::i, ti::j>(
      dt_a_tilde,
      shift(ti::K) * d_a_tilde(ti::k, ti::i, ti::j) +
          (*conformal_factor_squared)() *
              ((*lapse)() * ((*spatial_ricci_tensor)(ti::i, ti::j) +
                             (*grad_spatial_z4_constraint)(ti::i, ti::j) +
                             (*grad_spatial_z4_constraint)(ti::j, ti::i)) -
               (*grad_grad_lapse)(ti::i, ti::j)) -
          one_third * conformal_spatial_metric(ti::i, ti::j) *
              ((*lapse_times_ricci_scalar_plus_divergence_z4_constraint)() -
               (*divergence_lapse)()) +
          (*a_tilde_times_field_b)(ti::i, ti::j) +
          (*a_tilde_times_field_b)(ti::j, ti::i) -
          2.0 * one_third * a_tilde(ti::i, ti::j) * (*contracted_field_b)() +
          (*lapse_times_a_tilde)(ti::i, ti::j) * (*k_minus_2_theta_c)() -
          2.0 * (*lapse_times_a_tilde)(ti::i, ti::l) *
              (*inv_conformal_spatial_metric)(ti::L, ti::M) *
              a_tilde(ti::m, ti::j) -
          (*inv_tau_times_conformal_metric)(ti::i, ti::j) * (*trace_a_tilde)());

  // eq. (12f) : time derivative of the trace of the extrinsic curvature
  ::tenex::evaluate(
      dt_trace_extrinsic_curvature,
      shift(ti::K) * d_trace_extrinsic_curvature(ti::k) -
          (*divergence_lapse)() +
          (*lapse_times_ricci_scalar_plus_divergence_z4_constraint)() +
          (*lapse)() * (trace_extrinsic_curvature() * (*k_minus_2_theta_c)() -
                        3.0 * kappa_1 * (1.0 + kappa_2) * theta()));

  // eq. (12g) : time derivative of the projection of the Z4 four-vector along
  // the normal direction
  ::tenex::evaluate(
      dt_theta,
      shift(ti::K) * d_theta(ti::k) +
          (*lapse)() *
              (0.5 * square(cleaning_speed) *
                   ((*ricci_scalar_plus_divergence_z4_constraint)() +
                    2.0 * one_third * square(trace_extrinsic_curvature()) -
                    a_tilde(ti::i, ti::j) * (*inv_a_tilde)(ti::I, ti::J)) -
               c * theta() * trace_extrinsic_curvature() -
               (*upper_spatial_z4_constraint)(ti::I)*field_a(ti::i) -
               kappa_1 * (2.0 + kappa_2) * theta()));

  // eq. (12h) : time derivative \hat{\Gamma}^i
  // first, compute terms without s
  ::tenex::evaluate<ti::I>(
      dt_gamma_hat,
      // terms without lapse nor s
      (*shift_times_deriv_gamma_hat)(ti::I) +
          2.0 * one_third *
              (*contracted_conformal_christoffel_second_kind)(ti::I) *
              (*contracted_field_b)() -
          (*contracted_conformal_christoffel_second_kind)(ti::K)*field_b(
              ti::k, ti::I) +
          2.0 * kappa_3 * (*spatial_z4_constraint)(ti::j) *
              (2.0 * one_third * (*inv_conformal_spatial_metric)(ti::I, ti::J) *
                   (*contracted_field_b)() -
               (*inv_conformal_spatial_metric)(ti::J, ti::K) *
                   field_b(ti::k, ti::I)) +
          // terms with lapse but not s
          2.0 * (*lapse)() *
              (-2.0 * one_third *
                   (*inv_conformal_spatial_metric)(ti::I, ti::J) *
                   d_trace_extrinsic_curvature(ti::j) +
               (*inv_conformal_spatial_metric)(ti::K, ti::I) * d_theta(ti::k) +
               (*conformal_christoffel_second_kind)(ti::I, ti::j, ti::k) *
                   (*inv_a_tilde)(ti::J, ti::K) -
               3.0 * (*inv_a_tilde)(ti::I, ti::J) * field_p(ti::j) -
               (*inv_conformal_spatial_metric)(ti::K, ti::I) *
                   (theta() * field_a(ti::k) +
                    2.0 * one_third * trace_extrinsic_curvature() *
                        (*spatial_z4_constraint)(ti::k)) -
               (*inv_a_tilde)(ti::I, ti::J) * field_a(ti::j) -
               kappa_1 * (*inv_conformal_spatial_metric)(ti::I, ti::J) *
                   (*spatial_z4_constraint)(ti::j)));
  // now, if s == 1, also add terms with s
  if (static_cast<bool>(evolve_shift)) {
    ::tenex::update<ti::I>(
        dt_gamma_hat,
        (*dt_gamma_hat)(ti::I) +
            // terms with lapse and s
            2.0 * (*lapse)() *
                ((*inv_conformal_spatial_metric)(ti::I, ti::K) *
                     (*inv_conformal_spatial_metric)(ti::N, ti::M) *
                     d_a_tilde(ti::k, ti::n, ti::m) -
                 2.0 * (*inv_conformal_spatial_metric)(ti::I, ti::K) *
                     (*field_d_up)(ti::k, ti::N, ti::M) *
                     a_tilde(ti::n, ti::m)) +
            // terms with s but not not lapse
            (*inv_conformal_spatial_metric)(ti::K, ti::L) *
                (*symmetrized_d_field_b)(ti::k, ti::l, ti::I) +
            one_third * (*inv_conformal_spatial_metric)(ti::I, ti::K) *
                (*contracted_symmetrized_d_field_b)(ti::k));
  }

  // eq. (12i) : time derivative b^i
  // if s == 0
  if (not static_cast<bool>(evolve_shift)) {
    for (auto& component : *dt_b) {
      component = 0.0;
    }
  } else {
    ::tenex::evaluate<ti::I>(
        dt_b,
        (*dt_gamma_hat)(ti::I)-eta() * b(ti::I) +
            shift(ti::K) * (d_b(ti::k, ti::I) - d_gamma_hat(ti::k, ti::I)));
  }

  // eq. (12j) : time derivative of auxiliary variable A_i
  // first, compute terms without s
  ::tenex::evaluate<ti::k>(
      dt_field_a,
      shift(ti::L) * d_field_a(ti::l, ti::k) -
          (*lapse_times_field_a)(ti::k) * (*k_minus_k0_minus_2_theta_c)() *
              ((*slicing_condition)() + (*lapse)() * (*d_slicing_condition)()) +
          field_b(ti::k, ti::L) * field_a(ti::l) -
          (*lapse_times_slicing_condition)() *
              (d_trace_extrinsic_curvature(ti::k) - d_k_0(ti::k) -
               2.0 * c * d_theta(ti::k)));
  // now, if s == 1, also add terms with s
  if (static_cast<bool>(evolve_shift)) {
    ::tenex::update<ti::k>(
        dt_field_a, (*dt_field_a)(ti::k) -
                        (*lapse_times_slicing_condition)() *
                            ((*inv_conformal_metric_times_d_a_tilde)(ti::k) -
                             (2.0 * (*field_d_up_times_a_tilde)(ti::k))));
  }

  // eq. (12k) : time derivative of auxiliary variable B_k{}^i
  // if s == 0
  if (not static_cast<bool>(evolve_shift)) {
    for (auto& component : *dt_field_b) {
      component = 0.0;
    }
  } else {
    // first, compute expression without advective terms
    ::tenex::evaluate<ti::k, ti::I>(
        dt_field_b, shift(ti::L) * d_field_b(ti::l, ti::k, ti::I) +
                        f * d_b(ti::k, ti::I) +
                        mu * square((*lapse)()) *
                            (*inv_conformal_spatial_metric)(ti::I, ti::J) *
                            (d_field_p(ti::k, ti::j) - d_field_p(ti::j, ti::k) -
                             (*inv_conformal_spatial_metric)(ti::N, ti::L) *
                                 (d_field_d(ti::k, ti::l, ti::j, ti::n) -
                                  d_field_d(ti::l, ti::k, ti::j, ti::n))) +
                        field_b(ti::k, ti::L) * field_b(ti::l, ti::I));
  }

  // eq. (12l) : time derivative of auxiliary variable D_{kij}
  // first, compute terms without s
  ::tenex::evaluate<ti::k, ti::i, ti::j>(
      dt_field_d,
      shift(ti::L) * d_field_d(ti::l, ti::k, ti::i, ti::j) -
          (*lapse_times_d_a_tilde)(ti::k, ti::i, ti::j) +
          field_b(ti::k, ti::L) * field_d(ti::l, ti::i, ti::j) +
          (*field_b_times_field_d)(ti::j, ti::k, ti::i) +
          (*field_b_times_field_d)(ti::i, ti::k, ti::j) -
          (*lapse_times_field_a)(ti::k) *
              (*a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde)(
                  ti::i, ti::j) +
          one_third *
              ((*lapse_times_conformal_spatial_metric)(ti::i, ti::j) *
                   (*inv_conformal_metric_times_d_a_tilde)(ti::k) -
               (2.0 * (*contracted_field_b)() * field_d(ti::k, ti::i, ti::j)) -
               2.0 * (*lapse_times_conformal_spatial_metric)(ti::i, ti::j) *
                   (*field_d_up_times_a_tilde)(ti::k)));
  // now, if s == 1, also add terms with s
  if (static_cast<bool>(evolve_shift)) {
    ::tenex::update<ti::k, ti::i, ti::j>(
        dt_field_d, (*dt_field_d)(ti::k, ti::i, ti::j) +
                        0.5 * ((*conformal_metric_times_symmetrized_d_field_b)(
                                   ti::i, ti::k, ti::j) +
                               (*conformal_metric_times_symmetrized_d_field_b)(
                                   ti::j, ti::k, ti::i)) -
                        one_third * conformal_spatial_metric(ti::i, ti::j) *
                            (*contracted_symmetrized_d_field_b)(ti::k));
  }

  // eq. (12m) : time derivative of auxiliary variable P_i
  // first, compute terms without s
  ::tenex::evaluate<ti::k>(
      dt_field_p, shift(ti::L) * d_field_p(ti::l, ti::k) +
                      field_b(ti::k, ti::L) * field_p(ti::l) +
                      one_third * (*lapse)() *
                          (d_trace_extrinsic_curvature(ti::k) +
                           field_a(ti::k) * trace_extrinsic_curvature()));
  // now, if s == 1, also add terms with s
  if (static_cast<bool>(evolve_shift)) {
    ::tenex::update<ti::k>(
        dt_field_p,
        (*dt_field_p)(ti::k) +
            one_third *
                ((*lapse)() * ((*inv_conformal_metric_times_d_a_tilde)(ti::k) -
                               (2.0 * (*field_d_up_times_a_tilde)(ti::k))) -
                 (*contracted_symmetrized_d_field_b)(ti::k)));
  }
}
}  // namespace Ccz4

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data) template struct Ccz4::TimeDerivative<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
