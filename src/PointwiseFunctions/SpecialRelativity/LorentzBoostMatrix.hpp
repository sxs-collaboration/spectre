// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

/// \endcond

/// \ingroup SpecialRelativityGroup
/// Holds functions related to special relativity.
namespace sr {
/// @{
/*!
 * \ingroup SpecialRelativityGroup
 * \brief Computes the matrix for a Lorentz boost from a single
 * velocity vector (i.e., not a velocity field).
 *
 * \details Given a spatial velocity vector \f$v^i\f$ (with \f$c=1\f$),
 * compute the matrix \f$\Lambda^{a}{}_{\bar{a}}\f$ for a Lorentz boost with
 * that velocity [e.g. Eq. (2.38) of \cite ThorneBlandford2017]:
 *
 * \f{align}{
 * \Lambda^t{}_{\bar{t}} &= \gamma, \\
 * \Lambda^t{}_{\bar{i}} = \Lambda^i{}_{\bar{t}} &= \gamma v^i, \\
 * \Lambda^i{}_{\bar{j}} = \Lambda^j{}_{\bar{i}} &= [(\gamma - 1)/v^2] v^i v^j
 *                                              + \delta^{ij}.
 * \f}
 *
 * Here \f$v = \sqrt{\delta_{ij} v^i v^j}\f$, \f$\gamma = 1/\sqrt{1-v^2}\f$,
 * and \f$\delta^{ij}\f$ is the Kronecker delta. Note that this matrix boosts
 * a one-form from the unbarred to the barred frame, and its inverse
 * (obtained via \f$v \rightarrow -v\f$) boosts a vector from the barred to
 * the unbarred frame.
 *
 * Note that while the Lorentz boost matrix is symmetric, the returned
 * boost matrix is of type `tnsr::Ab`, because `Tensor` does not support
 * symmetric tensors unless both indices have the same valence.
 */
template <size_t SpatialDim>
tnsr::Ab<double, SpatialDim, Frame::NoFrame> lorentz_boost_matrix(
    const tnsr::I<double, SpatialDim, Frame::NoFrame>& velocity) noexcept;

template <size_t SpatialDim>
void lorentz_boost_matrix(
    gsl::not_null<tnsr::Ab<double, SpatialDim, Frame::NoFrame>*> boost_matrix,
    const tnsr::I<double, SpatialDim, Frame::NoFrame>& velocity) noexcept;
/// @}
}  // namespace sr
