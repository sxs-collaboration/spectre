// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts {

/// @{
/*!
 * \brief Extrinsic curvature computed from the conformal decomposition used in
 * the XCTS system.
 *
 * The extrinsic curvature decomposition is (see Eq. 3.113 in
 * \cite BaumgarteShapiro):
 *
 * \begin{equation}
 *   K_{ij} = A_{ij} + \frac{1}{3}\gamma_{ij}K
 *     = \frac{\psi^4}{2\lapse}\left((\bar{L}\beta)_{ij} - \bar{u}_{ij}\right)
 *       + \frac{\psi^4}{3} \bar{\gamma}_{ij} K
 * \end{equation}
 *
 * \param result output buffer for the extrinsic curvature
 * \param conformal_factor the conformal factor $\psi$
 * \param lapse the lapse $\alpha$
 * \param conformal_metric the conformal metric $\bar{\gamma}_{ij}$
 * \param longitudinal_shift_minus_dt_conformal_metric the term
 * $(\bar{L}\beta)^{ij} - \bar{u}^{ij}$. Note that $(\bar{L}\beta)^{ij}$ is the
 * _conformal_ longitudinal shift, and $(\bar{L}\beta)^{ij}=\psi^4(L\beta)^{ij}$
 * (Eq. 3.98 in \cite BaumgarteShapiro). See also Xcts::longitudinal_operator.
 * \param trace_extrinsic_curvature the trace of the extrinsic curvature,
 * $K=\gamma^{ij}K_{ij}$. Note that it is a conformal invariant, $K=\bar{K}$ (by
 * choice).
 */
template <typename DataType>
void extrinsic_curvature(
    const gsl::not_null<tnsr::ii<DataType, 3>*> result,
    const Scalar<DataType>& conformal_factor, const Scalar<DataType>& lapse,
    const tnsr::ii<DataType, 3>& conformal_metric,
    const tnsr::II<DataType, 3>& longitudinal_shift_minus_dt_conformal_metric,
    const Scalar<DataType>& trace_extrinsic_curvature);

/// Return-by-value overload
template <typename DataType>
tnsr::ii<DataType, 3> extrinsic_curvature(
    const Scalar<DataType>& conformal_factor, const Scalar<DataType>& lapse,
    const tnsr::ii<DataType, 3>& conformal_metric,
    const tnsr::II<DataType, 3>& longitudinal_shift_minus_dt_conformal_metric,
    const Scalar<DataType>& trace_extrinsic_curvature);
/// @}

}  // namespace Xcts
