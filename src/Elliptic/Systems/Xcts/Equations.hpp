// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Xcts {
namespace detail {
// Tensor-contraction helper functions that should be replaced by tensor
// expressions once those work
void fully_contract_flat_cartesian(
    gsl::not_null<Scalar<DataVector>*> result,
    const tnsr::II<DataVector, 3>& tensor1,
    const tnsr::II<DataVector, 3>& tensor2) noexcept;
void fully_contract(gsl::not_null<Scalar<DataVector>*> result,
                    gsl::not_null<DataVector*> buffer1,
                    gsl::not_null<DataVector*> buffer2,
                    const tnsr::II<DataVector, 3>& tensor1,
                    const tnsr::II<DataVector, 3>& tensor2,
                    const tnsr::ii<DataVector, 3>& metric) noexcept;
}  // namespace detail

/*!
 * \brief Add the nonlinear source to the Hamiltonian constraint on a flat
 * conformal background in Cartesian coordinates and with
 * \f$\bar{u}_{ij}=0=\beta^i\f$.
 *
 * Adds \f$\frac{1}{12}\psi^5 K^2 - 2\pi\psi^{5-n}\bar{\rho}\f$ where \f$n\f$ is
 * the `ConformalMatterScale` and \f$\bar{\rho}=\psi^n\rho\f$ is the
 * conformally-scaled energy density. Additional sources can be added with
 * `add_distortion_hamiltonian_sources` and
 * `add_curved_hamiltonian_or_lapse_sources`.
 *
 * \see Xcts
 */
template <int ConformalMatterScale>
void add_hamiltonian_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const Scalar<DataVector>& conformal_energy_density,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor) noexcept;

/// The linearization of `add_hamiltonian_sources`
template <int ConformalMatterScale>
void add_linearized_hamiltonian_sources(
    gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    const Scalar<DataVector>& conformal_energy_density,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction) noexcept;

/*!
 * \brief Add the "distortion" source term to the Hamiltonian constraint.
 *
 * Adds \f$-\frac{1}{8}\psi^{-7}\bar{A}^{ij}\bar{A}_{ij}\f$.
 *
 * \see Xcts
 * \see Xcts::Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare
 */
void add_distortion_hamiltonian_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& conformal_factor) noexcept;

/*!
 * \brief The linearization of `add_distortion_hamiltonian_sources`
 *
 * Note that this linearization is only w.r.t. \f$\psi\f$.
 */
void add_linearized_distortion_hamiltonian_sources(
    gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction) noexcept;

/*!
 * \brief Add the contributions from a curved background geometry to the
 * Hamiltonian constraint or lapse equation
 *
 * Adds \f$\frac{1}{8}\psi\bar{R}\f$. This term appears both in the Hamiltonian
 * constraint and the lapse equation (where in the latter \f$\psi\f$ is replaced
 * by \f$\alpha\psi\f$).
 *
 * This term is linear.
 *
 * \see Xcts
 */
void add_curved_hamiltonian_or_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_or_lapse_equation,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const Scalar<DataVector>& field) noexcept;

/*!
 * \brief Add the nonlinear source to the lapse equation on a flat conformal
 * background in Cartesian coordinates and with \f$\bar{u}_{ij}=0=\beta^i\f$.
 *
 * Adds \f$(\alpha\psi)\left(\frac{5}{12}\psi^4 K^2 + 2\pi\psi^{4-n}
 * \left(\bar{\rho} + 2\bar{S}\right)\right) + \psi^5
 * \left(\beta^i\partial_i K - \partial_t K\right)\f$ where \f$n\f$ is the
 * `ConformalMatterScale` and matter quantities are conformally-scaled.
 * Additional sources can be added with
 * `add_distortion_hamiltonian_and_lapse_sources` and
 * `add_curved_hamiltonian_or_lapse_sources`.
 *
 * \see Xcts
 */
template <int ConformalMatterScale>
void add_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const Scalar<DataVector>& conformal_energy_density,
    const Scalar<DataVector>& conformal_stress_trace,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& dt_extrinsic_curvature_trace,
    const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept;

/*!
 * \brief The linearization of `add_lapse_sources`
 *
 * Note that this linearization is only w.r.t. \f$\psi\f$ and \f$\alpha\psi\f$.
 * The linearization w.r.t. \f$\beta^i\f$ is added in
 * `add_curved_linearized_momentum_sources` or
 * `add_flat_cartesian_linearized_momentum_sources`.
 */
template <int ConformalMatterScale>
void add_linearized_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
    const Scalar<DataVector>& conformal_energy_density,
    const Scalar<DataVector>& conformal_stress_trace,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const Scalar<DataVector>& dt_extrinsic_curvature_trace,
    const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction) noexcept;

/*!
 * \brief Add the "distortion" source term to the Hamiltonian constraint and the
 * lapse equation.
 *
 * Adds \f$-\frac{1}{8}\psi^{-7}\bar{A}^{ij}\bar{A}_{ij}\f$ to the Hamiltonian
 * constraint and \f$\frac{7}{8}\alpha\psi^{-7}\bar{A}^{ij}\bar{A}_{ij}\f$ to
 * the lapse equation.
 *
 * \see Xcts
 * \see Xcts::Tags::LongitudinalShiftMinusDtConformalMetricSquare
 */
void add_distortion_hamiltonian_and_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept;

/*!
 * \brief The linearization of `add_distortion_hamiltonian_and_lapse_sources`
 *
 * Note that this linearization is only w.r.t. \f$\psi\f$ and \f$\alpha\psi\f$.
 */
void add_linearized_distortion_hamiltonian_and_lapse_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> lapse_equation,
    const Scalar<DataVector>&
        longitudinal_shift_minus_dt_conformal_metric_square,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction) noexcept;

/*!
 * \brief Add the nonlinear source to the momentum constraint and add the
 * "distortion" source term to the Hamiltonian constraint and lapse equation.
 *
 * Adds \f$\left((\bar{L}\beta)^{ij} - \bar{u}^{ij}\right)
 * \left(\frac{\partial_j (\alpha\psi)}{\alpha\psi}
 * - 7 \frac{\partial_j \psi}{\psi}\right)
 * + \partial_j\bar{u}^{ij}
 * + \frac{4}{3}\frac{\alpha\psi}{\psi}\bar{\gamma}^{ij}\partial_j K
 * + 16\pi\left(\alpha\psi\right)\psi^{3-n} \bar{S}^i\f$ to the momentum
 * constraint, where \f$n\f$ is the `ConformalMatterScale` and
 * \f$\bar{S}^i=\psi^n S^i\f$ is the conformally-scaled momentum density.
 *
 * Note that the \f$\partial_j\bar{u}^{ij}\f$ term is not the full covariant
 * divergence, but only the partial-derivatives part of it. The curved
 * contribution to this term can be added together with the curved contribution
 * to the flux divergence of the dynamic shift variable with the
 * `Elasticity::add_curved_sources` function.
 *
 * Also adds \f$-\frac{1}{8}\psi^{-7}\bar{A}^{ij}\bar{A}_{ij}\f$ to the
 * Hamiltonian constraint and
 * \f$\frac{7}{8}\alpha\psi^{-7}\bar{A}^{ij}\bar{A}_{ij}\f$ to the lapse
 * equation.
 *
 * \see Xcts
 */
/// @{
template <int ConformalMatterScale>
void add_curved_momentum_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> lapse_equation,
    gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
    const tnsr::I<DataVector, 3>& conformal_momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_minus_dt_conformal_metric) noexcept;

template <int ConformalMatterScale>
void add_flat_cartesian_momentum_sources(
    gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> lapse_equation,
    gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
    const tnsr::I<DataVector, 3>& conformal_momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const tnsr::I<DataVector, 3>& minus_div_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_minus_dt_conformal_metric) noexcept;
/// @}

/// The linearization of `add_curved_momentum_sources`
template <int ConformalMatterScale>
void add_curved_linearized_momentum_sources(
    gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
    gsl::not_null<tnsr::I<DataVector, 3>*> linearized_momentum_constraint,
    const tnsr::I<DataVector, 3>& conformal_momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>& longitudinal_shift_minus_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3>& shift_correction,
    const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux_correction,
    const tnsr::II<DataVector, 3>& longitudinal_shift_correction) noexcept;

/// The linearization of `add_flat_cartesian_momentum_sources`
template <int ConformalMatterScale>
void add_flat_cartesian_linearized_momentum_sources(
    gsl::not_null<Scalar<DataVector>*> linearized_hamiltonian_constraint,
    gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
    gsl::not_null<tnsr::I<DataVector, 3>*> linearized_momentum_constraint,
    const tnsr::I<DataVector, 3>& conformal_momentum_density,
    const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& conformal_factor_flux,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
    const tnsr::II<DataVector, 3>& longitudinal_shift_minus_dt_conformal_metric,
    const Scalar<DataVector>& conformal_factor_correction,
    const Scalar<DataVector>& lapse_times_conformal_factor_correction,
    const tnsr::I<DataVector, 3>& shift_correction,
    const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
    const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux_correction,
    const tnsr::II<DataVector, 3>& longitudinal_shift_correction) noexcept;
}  // namespace Xcts
