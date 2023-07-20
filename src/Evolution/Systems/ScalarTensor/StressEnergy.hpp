// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarTensor {

/*!
 * \brief Add in the trace-reversed stress-energy source term to the \f$\Pi\f$
 * evolved variable of the ::gh system.
 *
 * \details The only stress energy source term in the Generalized Harmonic
 * evolution equations is in the equation for \f$\Pi_{a b}\f$:
 * \f[
 * \partial_t \Pi_{ab} + \text{\{spatial derivative terms\}} =
 * \text{\{GH source terms\}}
 * - 16 \pi \alpha (T^{(\Psi)}_{ab} - \frac{1}{2} g_{a b} g^{cd}T^{(\Psi)}_{cd})
 * ~.
 * \f]
 *
 * (note that this function takes as argument the trace-reversed stress-energy
 * tensor)
 *
 * This function adds that contribution to the existing value of `dt_pi`. The
 * spacetime terms in the GH equation should be computed before passing the
 * `dt_pi` to this function for updating.
 *
 * \param dt_pi Time derivative of the $\Pi_{ab}$ variable in the ::gh system.
 * The vacuum part should be computed before with ::gh::TimeDerivative
 * \param trace_reversed_stress_energy Trace-reversed stress energy tensor of
 * the scalar $T^{(\Psi), \text{TR}}_{a b} \equiv T^{(\Psi)}_{ab} - \frac{1}{2}
 *  g_{a b} g^{cd}T^{(\Psi)}_{cd} = \partial_a \Psi \partial_b \Psi $.
 * \param lapse Lapse $\alpha$.
 *
 * \see `gh::TimeDerivative` for details about the spacetime
 * part of the time derivative calculation.
 */
void add_stress_energy_term_to_dt_pi(
    gsl::not_null<tnsr::aa<DataVector, 3_st>*> dt_pi,
    const tnsr::aa<DataVector, 3_st>& trace_reversed_stress_energy,
    const Scalar<DataVector>& lapse);

/*!
 * \brief Compute the trace-reversed stress-energy tensor of the scalar field.
 *
 * \details The trace-reversed stress energy tensor is needed to compute the
 * backreaction of the scalar to the spacetime evolution and is given by
 * \f{align*}{
 * T^{(\Psi), \text{TR}}_{a b} &\equiv T^{(\Psi)}_{ab} - \frac{1}{2}
 *  g_{a b} g^{cd}T^{(\Psi)}_{cd} \\
 *  &= \partial_a \Psi \partial_b \Psi ~,
 * \f}
 *
 * where \f$T^{(\Psi)}_{ab}\f$ is the standard stress-energy tensor of the
 * scalar.
 *
 * In terms of the evolved variables of the scalar,
 * \f{align*}{
    T^{(\Psi), \text{TR}}_{00} &= \alpha^2 \Pi^2 ~, \\
    T^{(\Psi), \text{TR}}_{j 0} &= T^{(\Psi), \text{TR}}_{0j}
                                 = - \alpha \Pi \Phi_j ~, \\
    T^{(\Psi), \text{TR}}_{ij} &= \Phi_i \Phi_j ~,
 * \f}
 *
 * where \f$\alpha\f$ is the lapse.
 *
 * \param stress_energy Trace-reversed stress energy tensor of
 * the scalar $T^{(\Psi), \text{TR}}_{a b} \equiv T^{(\Psi)}_{ab} - \frac{1}{2}
 *  g_{a b} g^{cd}T^{(\Psi)}_{cd} = \partial_a \Psi \partial_b \Psi $.
 * \param pi_scalar Scalar evolution variable $\Pi$.
 * \param phi_scalar Scalar evolution variable $\Phi_i$.
 * \param lapse Lapse $\alpha$.
 */
void trace_reversed_stress_energy(
    gsl::not_null<tnsr::aa<DataVector, 3_st>*> stress_energy,
    const Scalar<DataVector>& pi_scalar,
    const tnsr::i<DataVector, 3_st>& phi_scalar,
    const Scalar<DataVector>& lapse);

namespace Tags {

/*!
 * \brief Compute tag for the trace reversed stress energy tensor.
 *
 * \details Compute using ScalarTensor::trace_reversed_stress_energy.
 */
struct TraceReversedStressEnergyCompute
    : TraceReversedStressEnergy<DataVector, 3_st, Frame::Inertial>,
      db::ComputeTag {
  static constexpr size_t Dim = 3;
  using argument_tags =
      tmpl::list<CurvedScalarWave::Tags::Pi, CurvedScalarWave::Tags::Phi<Dim>,
                 gr::Tags::Lapse<DataVector>>;
  using return_type = tnsr::aa<DataVector, Dim, Frame::Inertial>;
  static constexpr void (*function)(
      const gsl::not_null<tnsr::aa<DataVector, Dim>*> result,
      const Scalar<DataVector>&, const tnsr::i<DataVector, Dim>&,
      const Scalar<DataVector>&) = &trace_reversed_stress_energy;
  using base = TraceReversedStressEnergy<DataVector, Dim, Frame::Inertial>;
};
}  // namespace Tags

}  // namespace ScalarTensor
