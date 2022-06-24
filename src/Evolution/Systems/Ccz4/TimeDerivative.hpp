// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace Ccz4 {
/*!
 * \brief Indicates whether or not to evolve the shift in a system evolved using
 * first order CCZ4 \cite Dumbser2017okk
 *
 * \details In \cite Dumbser2017okk , evolving the shift corresponds to
 * \f$s = 1\f$ and not evolving it corresponds to \f$s = 0\f$
 */
enum class EvolveShift : bool { False = false, True = true };

/*!
 * \brief Indicates which slicing condition to use in a system evolved using
 * first order CCZ4 \cite Dumbser2017okk
 *
 * \details In \cite Dumbser2017okk , harmonic slicing corresponds to
 * \f$g(\alpha) = 1\f$ and 1 + log slicing corresponds to
 * \f$g(\alpha) = 2 / \alpha\f$ where \f$\alpha\f$ is the lapse.
 */
enum class SlicingConditionType : char { Harmonic, Log };

/*!
 * \brief Compute the RHS of the first order CCZ4 formulation of Einstein's
 * equations \cite Dumbser2017okk
 *
 * \details We define \f$\phi = (\det(\gamma_{ij}))^{-1/6}\f$ as the conformal
 * factor, \f$\alpha\f$ as the lapse, \f$\beta^i\f$ as the shift, \f$K_{ij}\f$
 * as the extrinsic curvature, and \f$Z_{a}\f$ as the Z4 constraint.
 *
 * The evolved variables are the conformal spatial metric
 * \f$\tilde{\gamma}_{ij} = \phi^2 \gamma_{ij}\f$, the natural log of the lapse
 * \f$\ln \alpha\f$, the shift \f$\beta^i\f$, the natural log of the conformal
 * factor \f$\ln \phi\f$, the trace-free part of the extrinsic curvature
 * \f$\tilde A_{ij} = \phi^2 \left(K_{ij} - \frac{1}{3} K \gamma_{ij}\right)\f$,
 * the trace of the extrinsic curvature \f$K = K_{ij} \gamma^{ij}\f$, the
 * projection of the Z4 four-vector along the normal direction
 * \f$\Theta = Z^0 \alpha\f$, \f$\hat{\Gamma}^{i}\f$ defined by
 * `Ccz4::Tags::GammaHat`, the free variable \f$b^i\f$ that controls the
 * evolution of the shift and its time derivative, the auxiliary variable
 * \f$A_i = \partial_i \ln(\alpha) = \frac{\partial_i \alpha}{\alpha}\f$, the
 * auxiliary variable \f$B_k{}^{i} = \partial_k \beta^i\f$, the auxiliary
 * variable \f$D_{kij} = \frac{1}{2} \partial_k \tilde{\gamma}_{ij}\f$, and the
 * auxiliary variable
 * \f$P_i = \partial_i \ln(\phi) = \frac{\partial_i \phi}{\phi}\f$.
 *
 * The evolution equations are equations 12a - 12m of \cite Dumbser2017okk .
 * Equations 13 - 27 define identities used in the evolution equations.
 *
 * This evolution uses two settings that can be toggled:
 * - `evolve_shift` governs whether or not the shift is evolved by setting
 * \f$s = 1\f$ or \f$s = 0\f$
 * - `slicing_condition_type` governs which slicing condition to use by setting
 * the value of \f$g(\alpha)\f$)
 */
template <size_t Dim>
struct TimeDerivative {
  static void apply(
      const gsl::not_null<tnsr::ii<DataVector, Dim>*>
          dt_conformal_spatial_metric,
      const gsl::not_null<Scalar<DataVector>*> dt_ln_lapse,
      const gsl::not_null<tnsr::I<DataVector, Dim>*> dt_shift,
      const gsl::not_null<Scalar<DataVector>*> dt_ln_conformal_factor,
      const gsl::not_null<tnsr::ii<DataVector, Dim>*> dt_a_tilde,
      const gsl::not_null<Scalar<DataVector>*> dt_trace_extrinsic_curvature,
      const gsl::not_null<Scalar<DataVector>*> dt_theta,
      const gsl::not_null<tnsr::I<DataVector, Dim>*> dt_gamma_hat,
      const gsl::not_null<tnsr::I<DataVector, Dim>*> dt_b,
      const gsl::not_null<tnsr::i<DataVector, Dim>*> dt_field_a,
      const gsl::not_null<tnsr::iJ<DataVector, Dim>*> dt_field_b,
      const gsl::not_null<tnsr::ijj<DataVector, Dim>*> dt_field_d,
      const gsl::not_null<tnsr::i<DataVector, Dim>*> dt_field_p,
      const gsl::not_null<Scalar<DataVector>*> conformal_factor_squared,
      const gsl::not_null<Scalar<DataVector>*> det_conformal_spatial_metric,
      const gsl::not_null<tnsr::II<DataVector, Dim>*>
          inv_conformal_spatial_metric,
      const gsl::not_null<tnsr::II<DataVector, Dim>*> inv_spatial_metric,
      const gsl::not_null<Scalar<DataVector>*> lapse,
      const gsl::not_null<Scalar<DataVector>*> slicing_condition,
      const gsl::not_null<Scalar<DataVector>*> d_slicing_condition,
      const gsl::not_null<tnsr::II<DataVector, Dim>*> inv_a_tilde,
      const gsl::not_null<tnsr::ij<DataVector, Dim>*> a_tilde_times_field_b,
      const gsl::not_null<tnsr::ii<DataVector, Dim>*>
          a_tilde_minus_one_third_conformal_metric_times_trace_a_tilde,
      const gsl::not_null<Scalar<DataVector>*> contracted_field_b,
      const gsl::not_null<tnsr::ijK<DataVector, Dim>*> symmetrized_d_field_b,
      const gsl::not_null<tnsr::i<DataVector, Dim>*>
          contracted_symmetrized_d_field_b,
      const gsl::not_null<tnsr::ijk<DataVector, Dim>*> field_b_times_field_d,
      const gsl::not_null<tnsr::i<DataVector, Dim>*> field_d_up_times_a_tilde,
      const gsl::not_null<tnsr::I<DataVector, Dim>*> contracted_field_d_up,
      const gsl::not_null<Scalar<DataVector>*> half_conformal_factor_squared,
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
          contracted_christoffel_second_kind,
      const gsl::not_null<tnsr::ij<DataVector, Dim>*>
          contracted_d_conformal_christoffel_difference,
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
      const gsl::not_null<tnsr::I<DataVector, Dim>*>
          shift_times_deriv_gamma_hat,
      const gsl::not_null<tnsr::ii<DataVector, Dim>*>
          inv_tau_times_conformal_metric,
      const gsl::not_null<Scalar<DataVector>*> trace_a_tilde,
      const gsl::not_null<tnsr::iJJ<DataVector, Dim>*> field_d_up,
      const gsl::not_null<tnsr::Ijj<DataVector, Dim>*>
          conformal_christoffel_second_kind,
      const gsl::not_null<tnsr::iJkk<DataVector, Dim>*>
          d_conformal_christoffel_second_kind,
      const gsl::not_null<tnsr::Ijj<DataVector, Dim>*> christoffel_second_kind,
      const gsl::not_null<tnsr::ii<DataVector, Dim>*> spatial_ricci_tensor,
      const gsl::not_null<tnsr::ij<DataVector, Dim>*> grad_grad_lapse,
      const gsl::not_null<Scalar<DataVector>*> divergence_lapse,
      const gsl::not_null<tnsr::I<DataVector, Dim>*>
          contracted_conformal_christoffel_second_kind,
      const gsl::not_null<tnsr::iJ<DataVector, Dim>*>
          d_contracted_conformal_christoffel_second_kind,
      const gsl::not_null<tnsr::i<DataVector, Dim>*> spatial_z4_constraint,
      const gsl::not_null<tnsr::I<DataVector, Dim>*>
          upper_spatial_z4_constraint,
      const gsl::not_null<tnsr::ij<DataVector, Dim>*>
          grad_spatial_z4_constraint,
      const gsl::not_null<Scalar<DataVector>*>
          ricci_scalar_plus_divergence_z4_constraint,
      const double c, const double cleaning_speed,
      const Scalar<DataVector>& eta, const double f,
      const Scalar<DataVector>& k_0, const tnsr::i<DataVector, Dim>& d_k_0,
      const double kappa_1, const double kappa_2, const double kappa_3,
      const double mu, const double one_over_relaxation_time,
      const EvolveShift evolve_shift,
      const SlicingConditionType slicing_condition_type,
      const tnsr::ii<DataVector, Dim>& conformal_spatial_metric,
      const Scalar<DataVector>& ln_lapse, const tnsr::I<DataVector, Dim>& shift,
      const Scalar<DataVector>& ln_conformal_factor,
      const tnsr::ii<DataVector, Dim>& a_tilde,
      const Scalar<DataVector>& trace_extrinsic_curvature,
      const Scalar<DataVector>& theta,
      const tnsr::I<DataVector, Dim>& gamma_hat,
      const tnsr::I<DataVector, Dim>& b,
      const tnsr::i<DataVector, Dim>& field_a,
      const tnsr::iJ<DataVector, Dim>& field_b,
      const tnsr::ijj<DataVector, Dim>& field_d,
      const tnsr::i<DataVector, Dim>& field_p,
      const tnsr::ijj<DataVector, Dim>& d_a_tilde,
      const tnsr::i<DataVector, Dim>& d_trace_extrinsic_curvature,
      const tnsr::i<DataVector, Dim>& d_theta,
      const tnsr::iJ<DataVector, Dim>& d_gamma_hat,
      const tnsr::iJ<DataVector, Dim>& d_b,
      const tnsr::ij<DataVector, Dim>& d_field_a,
      const tnsr::ijK<DataVector, Dim>& d_field_b,
      const tnsr::ijkk<DataVector, Dim>& d_field_d,
      const tnsr::ij<DataVector, Dim>& d_field_p);
};
}  // namespace Ccz4
