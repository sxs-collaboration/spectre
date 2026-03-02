// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute null normal one-form to the boundary of a closed
 * region in a spatial slice of spacetime.
 *
 * \details Consider an \f$n-1\f$-dimensional boundary \f$S\f$ of a closed
 * region in an \f$n\f$-dimensional spatial hypersurface \f$\Sigma\f$. Let
 * \f$s^a\f$ be the unit spacelike vector orthogonal to \f$S\f$ in \f$\Sigma\f$,
 * and \f$n^a\f$ be the timelike unit vector orthogonal to \f$\Sigma\f$.
 * This function returns the null one-form that is outgoing/incoming on \f$S\f$:
 *
 * \f{align*}
 * k_a = \frac{1}{\sqrt{2}}\left(n_a \pm s_a\right).
 * \f}
 *
 * Here \f$n_a = g_{ab} n^b\f$ and \f$s_a = g_{ab} s^b\f$ are the spacetime
 * one-forms corresponding to the spacetime vectors \f$n^a\f$ and \f$s^a\f$.
 *
 * If \f$t^a=(1,0,0,0)\f$ is a vector in the time direction, then the unit
 * normal to the spatial slice \f$n^a\f$ is determined by the relation
 * \f$t^a = \alpha n^a + \beta^a\f$ (e.g. Eq. (2.98) of \cite BaumgarteShapiro),
 * where \f$\alpha\f$ is the lapse, \f$\beta^a = (0, \beta^i)\f$, and
 * \f$\beta^i\f$ is the shift. Solving for \f$n^a\f$ then gives
 * \f$n^a = \alpha^{-1}(t^a - \beta^a)\f$. Then since \f$n_a = g_{ab} n^b\f$,
 * the normal one-form is given by
 * \f$n_a = g_{ab} n^b = \alpha^{-1}(g_{at} - g_{ab} \beta^b)\f$. This implies
 * \f$n_i = \alpha^{-1}(g_{it} - g_{ij} \beta^j)\f$, or
 * \f$ n_i = -\alpha^{-1}(\beta_i - \beta_i) = 0\f$. Only \f$n_t\f$ is nonzero:
 * it is \f$n_t = \alpha^{-1}(g_{tt} - g_{tj} \beta^j)\f$, or
 * \f$n_t = \alpha^{-1}(-\alpha^2 + \beta_j \beta^j - \beta_j \beta^j)\f$, or
 * \f$n_t = -\alpha\f$. Note that \f$n^a\f$ is a unit timelike vector, since
 * \f$n^a n_a = n^t n_t = \alpha^{-1}(-\alpha) = -1\f$.
 *
 * The unit normal to the boundary \f$s^a\f$ is orthogonal to \f$n^a\f$ and has
 * components \f$s^a = (0, s^i)\f$, where \f$s^i\f$ is the spatial unit normal
 * vector to the boundary. Since it is a spacelike unit normal vector whose
 * time component vanishes, \f$s^a s_a = s^i s_i = 1\f$.
 * Note that \f$s_a = g_{ab} s^b = g_{aj} s^j\f$, so
 * \f$s_i = g_{ij} s^j = \gamma_{ij} s^j\f$, where \f$\gamma_{ij}\f$ is the
 * spatial metric, while \f$s_t = g_{tj} s^j = \beta_j s^j = \beta^j s_j\f$.
 * Thus \f$n^a\f$ and \f$s_a\f$ are orthogonal, since
 * \f$n^a s_a = \alpha^{-1}(t^a s_a - \beta^a s_a)\f$, or
 * \f$n^a s_a = \alpha^{-1}(\beta^j s_j - \beta^j s_j) = 0\f$.
 *
 * This function computes \f$s_a\f$ from the inputs \f$s_i\f$ (provided as
 * `interface_unit_normal_one_form`) and \f$\beta^i\f$ (provided as `shift`).
 */
template <typename DataType, size_t VolumeDim, typename Frame>
tnsr::a<DataType, VolumeDim, Frame> interface_null_normal(
    const tnsr::a<DataType, VolumeDim, Frame>& spacetime_normal_one_form,
    const tnsr::i<DataType, VolumeDim, Frame>& interface_unit_normal_one_form,
    const tnsr::I<DataType, VolumeDim, Frame>& shift, double sign);

template <typename DataType, size_t VolumeDim, typename Frame>
void interface_null_normal(
    gsl::not_null<tnsr::a<DataType, VolumeDim, Frame>*> null_one_form,
    const tnsr::a<DataType, VolumeDim, Frame>& spacetime_normal_one_form,
    const tnsr::i<DataType, VolumeDim, Frame>& interface_unit_normal_one_form,
    const tnsr::I<DataType, VolumeDim, Frame>& shift, double sign);
/// @}

/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute null normal vector to the boundary of a closed
 * region in a spatial slice of spacetime.
 *
 * \details Consider an \f$n-1\f$-dimensional boundary \f$S\f$ of a closed
 * region in an \f$n\f$-dimensional spatial hypersurface \f$\Sigma\f$. Let
 * \f$s^a\f$ be the unit spacelike vector orthogonal to \f$S\f$ in \f$\Sigma\f$,
 * and \f$n^a\f$ be the timelike unit vector orthogonal to \f$\Sigma\f$.
 * This function returns the null vector that is outgoing/ingoing on \f$S\f$:
 *
 * \f{align*}
 * k^a = \frac{1}{\sqrt{2}}\left(n^a \pm s^a\right).
 * \f}
 */
template <typename DataType, size_t VolumeDim, typename Frame>
tnsr::A<DataType, VolumeDim, Frame> interface_null_normal(
    const tnsr::A<DataType, VolumeDim, Frame>& spacetime_normal_vector,
    const tnsr::I<DataType, VolumeDim, Frame>& interface_unit_normal_vector,
    double sign);

/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute null normal vector to the boundary of a closed
 * region in a spatial slice of spacetime.
 *
 * \details Consider an \f$n-1\f$-dimensional boundary \f$S\f$ of a closed
 * region in an \f$n\f$-dimensional spatial hypersurface \f$\Sigma\f$. Let
 * \f$s^a\f$ be the unit spacelike vector orthogonal to \f$S\f$ in \f$\Sigma\f$,
 * and \f$n^a\f$ be the timelike unit vector orthogonal to \f$\Sigma\f$.
 * This function returns the null vector that is outgoing/ingoing on \f$S\f$:
 *
 * \f{align*}
 * k^a = \frac{1}{\sqrt{2}}\left(n^a \pm s^a\right).
 * \f}
 */
template <typename DataType, size_t VolumeDim, typename Frame>
void interface_null_normal(
    gsl::not_null<tnsr::A<DataType, VolumeDim, Frame>*> null_vector,
    const tnsr::A<DataType, VolumeDim, Frame>& spacetime_normal_vector,
    const tnsr::I<DataType, VolumeDim, Frame>& interface_unit_normal_vector,
    double sign);
}  // namespace gr
