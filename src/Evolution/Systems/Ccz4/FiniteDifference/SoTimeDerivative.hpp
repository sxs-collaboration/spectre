// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once

#include <cmath>
#include <cstddef>

#include "DataStructures/DataBox/AsAccess.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedContainers.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Evolution/DgSubcell/Tags/GhostDataForReconstruction.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"
#include "Evolution/Systems/Ccz4/DerivLapse.hpp"
#include "Evolution/Systems/Ccz4/DerivZ4Constraint.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/BoundaryConditionGhostData.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/Ricci.hpp"
#include "Evolution/Systems/Ccz4/RicciScalarPlusDivergenceZ4Constraint.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/TagsDeclarations.hpp"
#include "Evolution/Systems/Ccz4/TempTags.hpp"
#include "Evolution/Systems/Ccz4/TimeDerivative.hpp"
#include "Evolution/Systems/Ccz4/Z4Constraint.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \brief The namespace for evolving the second-order Ccz4 system.
 * Spatial derivatives are computed using 4-th order finite
 * differencing. Currently this system only works in 3D.
 */
namespace Ccz4::fd {
const size_t Dim = 3;

namespace detail {
// Calculate the time derivative of the evolved variables for the second-order
// Ccz4 system. There is quite some overlap between this apply() funcion
// and the apply() function in the first-order Ccz4 system. However,
// it is not straightforward to directly reuse the first-order
// apply() function as the function signatures differ significantly.
template <size_t Dim>
static void apply(
    // LHS time derivatives of evolved variables: eq 4(a) - 4(i)
    const gsl::not_null<tnsr::ii<DataVector, Dim>*> dt_conformal_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> dt_lapse,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> dt_shift,
    const gsl::not_null<Scalar<DataVector>*> dt_conformal_factor,
    const gsl::not_null<tnsr::ii<DataVector, Dim>*> dt_a_tilde,
    const gsl::not_null<Scalar<DataVector>*> dt_trace_extrinsic_curvature,
    const gsl::not_null<Scalar<DataVector>*> dt_theta,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> dt_gamma_hat,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> dt_b,

    // quantities we need for computing eq 4, 13-27
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_squared,
    const gsl::not_null<Scalar<DataVector>*> det_conformal_spatial_metric,
    const gsl::not_null<tnsr::II<DataVector, Dim>*>
        inv_conformal_spatial_metric,
    const gsl::not_null<tnsr::II<DataVector, Dim>*> inv_spatial_metric,
    const gsl::not_null<tnsr::II<DataVector, Dim>*> inv_a_tilde,
    // temporary expressions
    const gsl::not_null<tnsr::ij<DataVector, Dim>*> a_tilde_times_field_b,
    const gsl::not_null<tnsr::ii<DataVector, Dim>*>
        a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde,
    const gsl::not_null<Scalar<DataVector>*> contracted_field_b,
    const gsl::not_null<tnsr::ijK<DataVector, Dim>*> symmetrized_d_field_b,
    const gsl::not_null<tnsr::i<DataVector, Dim>*>
        contracted_symmetrized_d_field_b,
    const gsl::not_null<tnsr::i<DataVector, Dim>*> field_d_up_times_a_tilde,
    const gsl::not_null<tnsr::I<DataVector, Dim>*>
        contracted_field_d_up,  // temp for eq 18 -20
    const gsl::not_null<Scalar<DataVector>*>
        half_conformal_factor_squared,  // temp for eq 25
    const gsl::not_null<tnsr::ij<DataVector, Dim>*>
        conformal_metric_times_field_b,
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

    // fixed params for SO-CCZ4
    // c = 1.0, cleaning_speed = 1.0
    // one_over_relaxation_time = 0.0
    const double c, const double cleaning_speed,  // e in the paper
    const double one_over_relaxation_time,        // \tau^{-1}

    // free params for SO-CCZ4
    const Scalar<DataVector>& eta, const double f, const double kappa_1,
    const double kappa_2, const double kappa_3, const Scalar<DataVector>& k_0,

    // evolved variables
    const tnsr::ii<DataVector, Dim>& conformal_spatial_metric,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::ii<DataVector, Dim>& a_tilde,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& theta, const tnsr::I<DataVector, Dim>& gamma_hat,
    const tnsr::I<DataVector, Dim>& b,

    // auxilliary fields and their derivatives
    const tnsr::i<DataVector, Dim>& field_a,
    const tnsr::iJ<DataVector, Dim>& field_b,
    const tnsr::ijj<DataVector, Dim>& field_d,
    const tnsr::i<DataVector, Dim>& field_p,
    const tnsr::ii<DataVector, Dim>& d_field_a,
    const tnsr::iiJ<DataVector, Dim>& d_field_b,
    const tnsr::iijj<DataVector, Dim>& d_field_d,
    const tnsr::ii<DataVector, Dim>& d_field_p,

    // spatial derivatives of other evolved variables
    const tnsr::ijj<DataVector, Dim>& d_a_tilde,
    const tnsr::i<DataVector, Dim>& d_trace_extrinsic_curvature,
    const tnsr::i<DataVector, Dim>& d_theta,
    const tnsr::iJ<DataVector, Dim>& d_gamma_hat,
    const tnsr::iJ<DataVector, Dim>& d_b, const bool shifting_shift,
    const bool evolve_lapse_and_shift) {
  constexpr double one_third = 1.0 / 3.0;
  // quantities we need for computing eq 4, 13 - 27

  determinant_and_inverse(det_conformal_spatial_metric,
                          inv_conformal_spatial_metric,
                          conformal_spatial_metric);

  get(*conformal_factor_squared) = square(get(conformal_factor));

  ::tenex::evaluate<ti::I, ti::J>(
      inv_spatial_metric, (*conformal_factor_squared)() *
                              (*inv_conformal_spatial_metric)(ti::I, ti::J));

  ::tenex::evaluate<ti::I, ti::J>(
      inv_a_tilde, a_tilde(ti::k, ti::l) *
                       (*inv_conformal_spatial_metric)(ti::I, ti::K) *
                       (*inv_conformal_spatial_metric)(ti::J, ti::L));

  ASSERT(min(get(lapse)) > 0.0,
         "The lapse must be positive when using 1+log slicing but is: "
             << get(lapse));

  // slicing_condition and d_slicing_condition is not used in SO-CCZ4
  // \alpha g(\alpha)  == 2
  *lapse_times_slicing_condition =
      make_with_value<Scalar<DataVector>>(lapse, 2.0);

  // expressions and identities needed for evolution equations: eq 13 - 27

  // eq 13
  ::tenex::evaluate(
      trace_a_tilde,
      (*inv_conformal_spatial_metric)(ti::I, ti::J) * a_tilde(ti::i, ti::j));

  // from eq 14: field_d_up is the D_k^{ij} tensor
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

  // comment for the future: we should probably ensure the traces are taken
  // before computing the differences as the off-diagonal terms are not
  // needed
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
  ::Ccz4::grad_grad_lapse(grad_grad_lapse, lapse, *christoffel_second_kind,
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

  ::tenex::evaluate<ti::k>(
      field_d_up_times_a_tilde,
      (*field_d_up)(ti::k, ti::I, ti::J) * a_tilde(ti::i, ti::j));

  ::tenex::evaluate<ti::i, ti::j>(
      conformal_metric_times_field_b,
      conformal_spatial_metric(ti::k, ti::i) * field_b(ti::j, ti::K));

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
                                  (lapse)() * a_tilde(ti::i, ti::j));

  tenex::evaluate<ti::k, ti::i, ti::j>(
      lapse_times_d_a_tilde, (lapse)() * d_a_tilde(ti::k, ti::i, ti::j));

  ::tenex::evaluate<ti::k>(lapse_times_field_a, (lapse)() * field_a(ti::k));

  ::tenex::evaluate<ti::i, ti::j>(
      lapse_times_conformal_spatial_metric,
      (lapse)() * conformal_spatial_metric(ti::i, ti::j));

  ::tenex::evaluate(
      lapse_times_ricci_scalar_plus_divergence_z4_constraint,
      (lapse)() * (*ricci_scalar_plus_divergence_z4_constraint)());

  ::tenex::evaluate<ti::I>(shift_times_deriv_gamma_hat,
                           shift(ti::K) * d_gamma_hat(ti::k, ti::I));

  ::tenex::evaluate<ti::i, ti::j>(
      inv_tau_times_conformal_metric,
      one_over_relaxation_time * conformal_spatial_metric(ti::i, ti::j));

  // time derivative computation: eq 4(a) - 4(i)
  // The way we evaluate the following may seem weird compared
  // to 4(a) - 4(i). This is because we try to reuse the first-order
  // Ccz4 codes as much as possible. Hence, the following is written
  // based on 12a-12i, with s=1, and red terms canceled.

  // eq 12a : time derivative of the conformal spatial metric
  // this reduces to eq 4a for our choice of \tau^{-1}=0.
  ::tenex::evaluate<ti::i, ti::j>(
      dt_conformal_spatial_metric,
      2.0 * shift(ti::K) * field_d(ti::k, ti::i, ti::j) +
          (*conformal_metric_times_field_b)(ti::i, ti::j) +
          (*conformal_metric_times_field_b)(ti::j, ti::i) -
          2.0 * one_third * conformal_spatial_metric(ti::i, ti::j) *
              (*contracted_field_b)() -
          2.0 * (lapse)() *
              (*a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde)(
                  ti::i, ti::j) -
          (*inv_tau_times_conformal_metric)(ti::i, ti::j) *
              ((*det_conformal_spatial_metric)() - 1.0));

  if (evolve_lapse_and_shift) {
    // time derivative of the lapse
    ::tenex::evaluate<>(dt_lapse, (shift(ti::K) * field_a(ti::k) -
                                   (*lapse_times_slicing_condition)() *
                                       (*k_minus_k0_minus_2_theta_c)()) *
                                      lapse());
    // time derivative of the shift
    ::tenex::evaluate<ti::I>(dt_shift, f * b(ti::I));
    if (shifting_shift) {
      ::tenex::update<ti::I>(
          dt_shift, (*dt_shift)(ti::I) + shift(ti::K) * field_b(ti::k, ti::I));
    }
  } else {
    *dt_lapse = make_with_value<Scalar<DataVector>>(lapse, 0.0);
    *dt_shift = make_with_value<tnsr::I<DataVector, Dim>>(shift, 0.0);
  }

  // time derivative of the the conformal factor
  ::tenex::evaluate(dt_conformal_factor,
                    (shift(ti::K) * field_p(ti::k) +
                     one_third * ((lapse)() * trace_extrinsic_curvature() -
                                  (*contracted_field_b)())) *
                        conformal_factor());

  // time derivative of the trace-free part of the extrinsic curvature
  ::tenex::evaluate<ti::i, ti::j>(
      dt_a_tilde,
      shift(ti::K) * d_a_tilde(ti::k, ti::i, ti::j) +
          (*conformal_factor_squared)() *
              ((lapse)() * ((*spatial_ricci_tensor)(ti::i, ti::j) +
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

  // time derivative of the trace of the extrinsic curvature
  ::tenex::evaluate(
      dt_trace_extrinsic_curvature,
      shift(ti::K) * d_trace_extrinsic_curvature(ti::k) -
          (*divergence_lapse)() +
          (*lapse_times_ricci_scalar_plus_divergence_z4_constraint)() +
          (lapse)() * (trace_extrinsic_curvature() * (*k_minus_2_theta_c)() -
                       3.0 * kappa_1 * (1.0 + kappa_2) * theta()));

  // time derivative of the projection of the Z4 four-vector along
  // the normal direction
  ::tenex::evaluate(
      dt_theta,
      shift(ti::K) * d_theta(ti::k) +
          (lapse)() *
              (0.5 * square(cleaning_speed) *
                   ((*ricci_scalar_plus_divergence_z4_constraint)() +
                    2.0 * one_third * square(trace_extrinsic_curvature()) -
                    a_tilde(ti::i, ti::j) * (*inv_a_tilde)(ti::I, ti::J)) -
               c * theta() * trace_extrinsic_curvature() -
               (*upper_spatial_z4_constraint)(ti::I)*field_a(ti::i) -
               kappa_1 * (2.0 + kappa_2) * theta()));

  // time derivative \hat{\Gamma}^i
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
          2.0 * (lapse)() *
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
  // We add the following since s=1 (Gamma-driver gauge) is assumed in SoCcz4
  ::tenex::update<ti::I>(dt_gamma_hat,
                         (*dt_gamma_hat)(ti::I) +
                             // red terms should cancel
                             // terms with s but not lapse
                             (*inv_conformal_spatial_metric)(ti::K, ti::L) *
                                 (*symmetrized_d_field_b)(ti::k, ti::l, ti::I) +
                             one_third *
                                 (*inv_conformal_spatial_metric)(ti::I, ti::K) *
                                 (*contracted_symmetrized_d_field_b)(ti::k));

  // time derivative b^i
  if (evolve_lapse_and_shift) {
    ::tenex::evaluate<ti::I>(dt_b, (*dt_gamma_hat)(ti::I)-eta() * b(ti::I));
    if (shifting_shift) {
      ::tenex::update<ti::I>(
          dt_b, (*dt_b)(ti::I) + shift(ti::K) * (d_b(ti::k, ti::I) -
                                                 d_gamma_hat(ti::k, ti::I)));
    }
  } else {
    (*dt_b).get(0) = 0.0;
    (*dt_b).get(1) = 0.0;
    (*dt_b).get(2) = 0.0;
  }

  // Note that, we do not need to evolve the auxiliary variables in SO-CCZ4.
}
}  // namespace detail

/*!
 * \brief Compute the RHS of the second-order CCZ4 formulation of Einstein's
 * equations \cite Dumbser2017okk with finite differencing.
 *
 * \details The evolution equations are equations 4(a) - 4(i) of
 * \cite Dumbser2017okk Equations 13 - 27 define identities
 * used in the evolution equations.
 *
 * \note Different from the first-order system, the lapse
 * and the conformal factor are evolved instead of their natural logs.
 * The four auxiliary varialbels \f$A_i\f$, \f$B_k{}^{i}\f$,
 * \f$D_{kij}\f$, and \f$P_i\f$ are NOT evolved.
 */
struct SoTimeDerivative {
  template <typename DbTagsList>
  static void apply(const gsl::not_null<db::DataBox<DbTagsList>*> box) {
    // compute the first spatial derivatives of the evolved variables
    using evolved_vars_tag = typename System::variables_tag;
    using gradients_tags = typename System::gradients_tags;

    // only 4-th order accurate second derivatives have been implemented
    // To keep the same order of accuracy we use the same order for
    // both first and second derivatives
    constexpr size_t fd_order = 4;
    const auto& evolved_vars = db::get<evolved_vars_tag>(*box);
    const Mesh<Dim>& subcell_mesh =
        db::get<evolution::dg::subcell::Tags::Mesh<Dim>>(*box);
    const size_t num_pts = subcell_mesh.number_of_grid_points();
    Variables<db::wrap_tags_in<::Tags::deriv, gradients_tags, tmpl::size_t<Dim>,
                               Frame::Inertial>>
        cell_centered_Ccz4_derivs{num_pts};
    const auto& cell_centered_logical_to_inertial_inv_jacobian =
        db::get<evolution::dg::subcell::fd::Tags::
                    InverseJacobianLogicalToInertial<Dim>>(*box);

    constexpr bool subcell_enabled_at_external_boundary =
        std::decay_t<decltype(db::get<Parallel::Tags::Metavariables>(
            *box))>::SubcellOptions::subcell_enabled_at_external_boundary;

    const Element<3>& element = db::get<domain::Tags::Element<3>>(*box);

    const Ccz4::fd::Reconstructor& recons =
        db::get<Ccz4::fd::Tags::Reconstructor>(*box);

    // If the element has external boundaries and subcell is enabled for
    // boundary elements, compute FD ghost data with a given boundary condition.
    if constexpr (subcell_enabled_at_external_boundary) {
      if (not element.external_boundaries().empty()) {
        fd::BoundaryConditionGhostData::apply(box, element, recons);
      }
    }

    const bool evolve_lapse_and_shift =
        get<::Ccz4::fd::Tags::EvolveLapseAndShift>(*box);

    ::Ccz4::fd::spacetime_derivatives(
        make_not_null(&cell_centered_Ccz4_derivs), evolved_vars,
        db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>>(
            *box),
        fd_order, subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian);

    // calculate the four auxiliary fields in eq. 6
    // auxiliary variables NOT evolved in SO-CCZ4
    const auto& d_lapse =
        get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                          Frame::Inertial>>(cell_centered_Ccz4_derivs);
    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(evolved_vars);
    const auto field_a = ::tenex::evaluate<ti::i>(d_lapse(ti::i) / lapse());

    const auto& field_b =
        get<::Tags::deriv<gr::Tags::Shift<DataVector, Dim>, tmpl::size_t<Dim>,
                          Frame::Inertial>>(cell_centered_Ccz4_derivs);

    const auto& d_spatial_conformal_metric =
        get<::Tags::deriv<::Ccz4::Tags::ConformalMetric<DataVector, Dim>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(
            cell_centered_Ccz4_derivs);
    tnsr::ijj<DataVector, Dim> field_d;
    ::tenex::evaluate<ti::i, ti::j, ti::k>(
        make_not_null(&field_d),
        0.5 * d_spatial_conformal_metric(ti::i, ti::j, ti::k));

    const auto& conformal_factor =
        get<::Ccz4::Tags::ConformalFactor<DataVector>>(evolved_vars);
    const auto& d_conformal_factor =
        get<::Tags::deriv<::Ccz4::Tags::ConformalFactor<DataVector>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(
            cell_centered_Ccz4_derivs);
    const auto field_p = ::tenex::evaluate<ti::i>(d_conformal_factor(ti::i) /
                                                  conformal_factor());

    // compute second derivatives of the evolved variables
    Variables<db::wrap_tags_in<::Tags::second_deriv, gradients_tags,
                               tmpl::size_t<Dim>, Frame::Inertial>>
        cell_centered_Ccz4_second_derivs{num_pts};

    Ccz4::fd::second_spacetime_derivatives(
        make_not_null(&cell_centered_Ccz4_second_derivs), evolved_vars,
        db::get<evolution::dg::subcell::Tags::GhostDataForReconstruction<Dim>>(
            *box),
        fd_order, subcell_mesh, cell_centered_logical_to_inertial_inv_jacobian);

    // compute spatial derivative of the four auxiliary fields
    const auto& d_d_lapse =
        get<::Tags::second_deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                                 Frame::Inertial>>(
            cell_centered_Ccz4_second_derivs);
    tnsr::ii<DataVector, Dim> d_field_a;
    ::tenex::evaluate<ti::i, ti::j>(
        make_not_null(&d_field_a),
        (d_d_lapse(ti::i, ti::j) - d_lapse(ti::i) * d_lapse(ti::j) / lapse()) /
            lapse());

    const auto& d_field_b =
        get<::Tags::second_deriv<gr::Tags::Shift<DataVector, Dim>,
                                 tmpl::size_t<Dim>, Frame::Inertial>>(
            cell_centered_Ccz4_second_derivs);

    const auto& d_d_conformal_metric =
        get<::Tags::second_deriv<::Ccz4::Tags::ConformalMetric<DataVector, Dim>,
                                 tmpl::size_t<Dim>, Frame::Inertial>>(
            cell_centered_Ccz4_second_derivs);

    tnsr::iijj<DataVector, Dim> d_field_d;
    ::tenex::evaluate<ti::i, ti::j, ti::k, ti::l>(
        make_not_null(&d_field_d),
        0.5 * d_d_conformal_metric(ti::i, ti::j, ti::k, ti::l));

    const auto& d_d_conformal_factor =
        get<::Tags::second_deriv<::Ccz4::Tags::ConformalFactor<DataVector>,
                                 tmpl::size_t<Dim>, Frame::Inertial>>(
            cell_centered_Ccz4_second_derivs);

    tnsr::ii<DataVector, Dim> d_field_p;
    ::tenex::evaluate<ti::i, ti::j>(
        make_not_null(&d_field_p),
        (d_d_conformal_factor(ti::i, ti::j) - d_conformal_factor(ti::i) *
                                                  d_conformal_factor(ti::j) /
                                                  conformal_factor()) /
            conformal_factor());

    // intialize containers to be supplied in the SO-CCZ4 TimeDerivative.cpp
    // apply() function quantities we need for computing eq 4, 13 - 27
    using TempVars = Variables<tmpl::list<
        ::Ccz4::Tags::ConformalFactorSquared<DataVector>,
        ::Ccz4::Tags::DetConformalSpatialMetric<DataVector>,
        ::Ccz4::Tags::InverseConformalMetric<DataVector, Dim>,
        gr::Tags::InverseSpatialMetric<DataVector, Dim>,
        ::Ccz4::Tags::InvATilde<DataVector, Dim>,
        ::Ccz4::Tags::ATildeTimesFieldB<DataVector, Dim>,
        ::Ccz4::Tags::ATildeMinusOneThirdConformalMetricTimesTraceATilde<
            DataVector, Dim>,
        ::Ccz4::Tags::ContractedFieldB<DataVector>,
        ::Ccz4::Tags::SymmetrizedDerivFieldB<DataVector, Dim>,
        ::Ccz4::Tags::ContractedSymmetrizedDerivFieldB<DataVector, Dim>,
        ::Ccz4::Tags::FieldDUpTimesATilde<DataVector, Dim>,
        ::Ccz4::Tags::ContractedFieldDUp<DataVector, Dim>,
        ::Ccz4::Tags::HalfConformalFactorSquared<DataVector>,
        ::Ccz4::Tags::ConformalMetricTimesFieldB<DataVector, Dim>,
        ::Ccz4::Tags::ConformalMetricTimesTraceATilde<DataVector, Dim>,
        ::Ccz4::Tags::InverseConformalMetricTimesDerivATilde<DataVector, Dim>,
        ::Ccz4::Tags::GammaHatMinusContractedConformalChristoffel<DataVector,
                                                                  Dim>,
        ::Ccz4::Tags::DerivGammaHatMinusContractedConformalChristoffel<
            DataVector, Dim>,
        ::Ccz4::Tags::ContractedChristoffelSecondKind<DataVector, Dim>,
        ::Ccz4::Tags::ContractedDerivConformalChristoffelDifference<DataVector,
                                                                    Dim>,
        ::Ccz4::Tags::KMinus2ThetaC<DataVector>,
        ::Ccz4::Tags::KMinusK0Minus2ThetaC<DataVector>,
        ::Ccz4::Tags::LapseTimesATilde<DataVector, Dim>,
        ::Ccz4::Tags::LapseTimesDerivATilde<DataVector, Dim>,
        ::Ccz4::Tags::LapseTimesFieldA<DataVector, Dim>,
        ::Ccz4::Tags::LapseTimesConformalMetric<DataVector, Dim>,
        ::Ccz4::Tags::LapseTimesSlicingCondition<DataVector>,
        ::Ccz4::Tags::LapseTimesRicciScalarPlus2DivergenceZ4Constraint<
            DataVector>,
        ::Ccz4::Tags::ShiftTimesDerivGammaHat<DataVector, Dim>,
        ::Ccz4::Tags::InverseTauTimesConformalMetric<DataVector, Dim>,
        ::Ccz4::Tags::TraceATilde<DataVector>,
        ::Ccz4::Tags::FieldDUp<DataVector, Dim>,
        ::Ccz4::Tags::ConformalChristoffelSecondKind<DataVector, Dim>,
        ::Ccz4::Tags::DerivConformalChristoffelSecondKind<DataVector, Dim>,
        ::Ccz4::Tags::ChristoffelSecondKind<DataVector, Dim>,
        ::Ccz4::Tags::SpatialRicciTensor<DataVector, Dim>,
        ::Ccz4::Tags::GradGradLapse<DataVector, Dim>,
        ::Ccz4::Tags::DivergenceLapse<DataVector>,
        ::Ccz4::Tags::ContractedConformalChristoffelSecondKind<DataVector, Dim>,
        ::Ccz4::Tags::DerivContractedConformalChristoffelSecondKind<DataVector,
                                                                    Dim>,
        ::Ccz4::Tags::SpatialZ4Constraint<DataVector, Dim>,
        ::Ccz4::Tags::GradSpatialZ4Constraint<DataVector, Dim>,
        ::Ccz4::Tags::RicciScalarPlusDivergenceZ4Constraint<DataVector>>>;

    TempVars temp_vars(num_pts);
    tnsr::I<DataVector, Dim> upper_spatial_z4_constraint(num_pts);

    // free params
    const double c = 1.0;               // c = 1.0 in SO-CCZ4
    const double cleaning_speed = 1.0;  // e in the paper; e = 1.0 for SO-CCZ4
    const Scalar<DataVector>& eta = get<::Ccz4::Tags::Eta<DataVector>>(*box);
    const double f = Ccz4::fd::System::f;
    const Scalar<DataVector>& k_0 = get<::Ccz4::Tags::K0<DataVector>>(*box);
    const double kappa_1 = get<::Ccz4::Tags::Kappa1>(*box);
    const double kappa_2 = get<::Ccz4::Tags::Kappa2>(*box);
    const double kappa_3 = get<::Ccz4::Tags::Kappa3>(*box);
    const double one_over_relaxation_time = 0.0;  // \tau^{-1} = 0 in SO-CCZ4
    const bool shifting_shift = Ccz4::fd::System::shifting_shift;

    // we assume the databox already has tags corresponding to dt of the evolved
    // variables
    using dt_variables_tag = db::add_tag_prefix<::Tags::dt, evolved_vars_tag>;

    // resize here

    db::mutate<dt_variables_tag,
               ::Ccz4::Tags::SpatialZ4ConstraintUp<DataVector, Dim>>(
        [&](const auto dt_vars_ptr,
            const auto upper_spatial_z4_constraint_ptr) {
          dt_vars_ptr->initialize(subcell_mesh.number_of_grid_points());
          auto& [conformal_factor_squared, det_conformal_spatial_metric,
                 inv_conformal_spatial_metric, inv_spatial_metric, inv_a_tilde,
                 a_tilde_times_field_b,
                 a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde,
                 contracted_field_b, symmetrized_d_field_b,
                 contracted_symmetrized_d_field_b, field_d_up_times_a_tilde,
                 contracted_field_d_up, half_conformal_factor_squared,
                 conformal_metric_times_field_b,
                 conformal_metric_times_trace_a_tilde,
                 inv_conformal_metric_times_d_a_tilde,
                 gamma_hat_minus_contracted_conformal_christoffel,
                 d_gamma_hat_minus_contracted_conformal_christoffel,
                 contracted_christoffel_second_kind,
                 contracted_d_conformal_christoffel_difference,
                 k_minus_2_theta_c, k_minus_k0_minus_2_theta_c,
                 lapse_times_a_tilde, lapse_times_d_a_tilde,
                 lapse_times_field_a, lapse_times_conformal_spatial_metric,
                 lapse_times_slicing_condition,
                 lapse_times_ricci_scalar_plus_divergence_z4_constraint,
                 shift_times_deriv_gamma_hat, inv_tau_times_conformal_metric,
                 trace_a_tilde, field_d_up, conformal_christoffel_second_kind,
                 d_conformal_christoffel_second_kind, christoffel_second_kind,
                 spatial_ricci_tensor, grad_grad_lapse, divergence_lapse,
                 contracted_conformal_christoffel_second_kind,
                 d_contracted_conformal_christoffel_second_kind,
                 spatial_z4_constraint, grad_spatial_z4_constraint,
                 ricci_scalar_plus_divergence_z4_constraint] = temp_vars;
          detail::apply(
              // LHS time derivatives of evolved variables: eq 4a - 4i
              make_not_null(
                  &get<::Tags::dt<
                      ::Ccz4::Tags::ConformalMetric<DataVector, Dim>>>(
                      *dt_vars_ptr)),  // eq 4a
              make_not_null(&get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(
                  *dt_vars_ptr)),  // eq 4g
              make_not_null(&get<::Tags::dt<gr::Tags::Shift<DataVector, Dim>>>(
                  *dt_vars_ptr)),  // eq 4h
              make_not_null(
                  &get<::Tags::dt<::Ccz4::Tags::ConformalFactor<DataVector>>>(
                      *dt_vars_ptr)),  // eq 4c
              make_not_null(
                  &get<::Tags::dt<::Ccz4::Tags::ATilde<DataVector, Dim>>>(
                      *dt_vars_ptr)),  // eq 4b
              make_not_null(&get<::Tags::dt<
                                gr::Tags::TraceExtrinsicCurvature<DataVector>>>(
                  *dt_vars_ptr)),  // eq 4d
              make_not_null(&get<::Tags::dt<::Ccz4::Tags::Theta<DataVector>>>(
                  *dt_vars_ptr)),  // eq 4e
              make_not_null(
                  &get<::Tags::dt<::Ccz4::Tags::GammaHat<DataVector, Dim>>>(
                      *dt_vars_ptr)),  // eq 4f
              make_not_null(
                  &get<::Tags::dt<
                      ::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim>>>(
                      *dt_vars_ptr)),  // eq 4i

              // quantities we need for computing eq 4, 13 - 27
              make_not_null(&conformal_factor_squared),
              make_not_null(&det_conformal_spatial_metric),
              make_not_null(&inv_conformal_spatial_metric),
              make_not_null(&inv_spatial_metric), make_not_null(&inv_a_tilde),
              // temporary expressions
              make_not_null(&a_tilde_times_field_b),
              make_not_null(
                &a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde),
              make_not_null(&contracted_field_b),
              make_not_null(&symmetrized_d_field_b),
              make_not_null(&contracted_symmetrized_d_field_b),
              make_not_null(&field_d_up_times_a_tilde),
              make_not_null(&contracted_field_d_up),  // temp for eq 18 -20
              make_not_null(&half_conformal_factor_squared),  // temp for eq 25
              make_not_null(&conformal_metric_times_field_b),
              make_not_null(&conformal_metric_times_trace_a_tilde),
              make_not_null(&inv_conformal_metric_times_d_a_tilde),
              make_not_null(&gamma_hat_minus_contracted_conformal_christoffel),
              make_not_null(
                  &d_gamma_hat_minus_contracted_conformal_christoffel),
              make_not_null(
                  &contracted_christoffel_second_kind),  // temp for eq 18 -20
              make_not_null(
                  &contracted_d_conformal_christoffel_difference),  // temp for
                                                                    // eq 18 -20
              make_not_null(&k_minus_2_theta_c),
              make_not_null(&k_minus_k0_minus_2_theta_c),
              make_not_null(&lapse_times_a_tilde),
              make_not_null(&lapse_times_d_a_tilde),
              make_not_null(&lapse_times_field_a),
              make_not_null(&lapse_times_conformal_spatial_metric),
              make_not_null(&lapse_times_slicing_condition),
              make_not_null(
                  &lapse_times_ricci_scalar_plus_divergence_z4_constraint),
              make_not_null(&shift_times_deriv_gamma_hat),
              make_not_null(&inv_tau_times_conformal_metric),
              // expressions and identities needed for evolution equations: eq
              // 13
              // - 27
              make_not_null(&trace_a_tilde),                        // eq 13
              make_not_null(&field_d_up),                           // eq 14
              make_not_null(&conformal_christoffel_second_kind),    // eq 15
              make_not_null(&d_conformal_christoffel_second_kind),  // eq 16
              make_not_null(&christoffel_second_kind),              // eq 17
              make_not_null(&spatial_ricci_tensor),  // eq 18 - 20
              make_not_null(&grad_grad_lapse),       // eq 21
              make_not_null(&divergence_lapse),      // eq 22
              make_not_null(
                  &contracted_conformal_christoffel_second_kind),  // eq 23
              make_not_null(
                  &d_contracted_conformal_christoffel_second_kind),  // eq 24
              make_not_null(&spatial_z4_constraint),                 // eq 25
              make_not_null(&upper_spatial_z4_constraint),           // eq 25
              make_not_null(&grad_spatial_z4_constraint),            // eq 26
              make_not_null(
                  &ricci_scalar_plus_divergence_z4_constraint),  // eq 27
              // fixed params
              c, cleaning_speed,         // e in the paper
              one_over_relaxation_time,  // \tau^{-1}
              // free params
              eta, f, kappa_1, kappa_2, kappa_3, k_0,
              // evolved variables
              get<::Ccz4::Tags::ConformalMetric<DataVector, Dim>>(evolved_vars),
              get<gr::Tags::Lapse<DataVector>>(evolved_vars),
              get<gr::Tags::Shift<DataVector, Dim>>(evolved_vars),
              get<::Ccz4::Tags::ConformalFactor<DataVector>>(evolved_vars),
              get<::Ccz4::Tags::ATilde<DataVector, Dim>>(evolved_vars),
              get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars),
              get<::Ccz4::Tags::Theta<DataVector>>(evolved_vars),
              get<::Ccz4::Tags::GammaHat<DataVector, Dim>>(evolved_vars),
              get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim>>(evolved_vars),

              field_a,  // auxiliary variables NOT evolved in SO-CCZ4
              field_b, field_d, field_p,
              d_field_a,  // spatial derivative of auxiliary variables
              d_field_b, d_field_d, d_field_p,

              // spatial derivatives of other evolved variables
              get<::Tags::deriv<::Ccz4::Tags::ATilde<DataVector, Dim>,
                                tmpl::size_t<Dim>, Frame::Inertial>>(
                  cell_centered_Ccz4_derivs),
              get<::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                                tmpl::size_t<Dim>, Frame::Inertial>>(
                  cell_centered_Ccz4_derivs),
              get<::Tags::deriv<::Ccz4::Tags::Theta<DataVector>,
                                tmpl::size_t<Dim>, Frame::Inertial>>(
                  cell_centered_Ccz4_derivs),
              get<::Tags::deriv<::Ccz4::Tags::GammaHat<DataVector, Dim>,
                                tmpl::size_t<Dim>, Frame::Inertial>>(
                  cell_centered_Ccz4_derivs),
              get<::Tags::deriv<::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim>,
                                tmpl::size_t<Dim>, Frame::Inertial>>(
                  cell_centered_Ccz4_derivs),
              shifting_shift, evolve_lapse_and_shift);

          *upper_spatial_z4_constraint_ptr =
              std::move(upper_spatial_z4_constraint);
        },
        box);
  }
};
}  // namespace Ccz4::fd
