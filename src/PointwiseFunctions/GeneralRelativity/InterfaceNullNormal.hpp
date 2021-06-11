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
 */
template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::a<DataType, VolumeDim, Frame> interface_null_normal(
    const tnsr::a<DataType, VolumeDim, Frame>& spacetime_normal_one_form,
    const tnsr::i<DataType, VolumeDim, Frame>& interface_unit_normal_one_form,
    const double sign) noexcept;

template <size_t VolumeDim, typename Frame, typename DataType>
void interface_null_normal(
    gsl::not_null<tnsr::a<DataType, VolumeDim, Frame>*> null_one_form,
    const tnsr::a<DataType, VolumeDim, Frame>& spacetime_normal_one_form,
    const tnsr::i<DataType, VolumeDim, Frame>& interface_unit_normal_one_form,
    const double sign) noexcept;
/// @}

/// @{
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
template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::A<DataType, VolumeDim, Frame> interface_null_normal(
    const tnsr::A<DataType, VolumeDim, Frame>& spacetime_normal_vector,
    const tnsr::I<DataType, VolumeDim, Frame>& interface_unit_normal_vector,
    const double sign) noexcept;

template <size_t VolumeDim, typename Frame, typename DataType>
void interface_null_normal(
    gsl::not_null<tnsr::A<DataType, VolumeDim, Frame>*> null_vector,
    const tnsr::A<DataType, VolumeDim, Frame>& spacetime_normal_vector,
    const tnsr::I<DataType, VolumeDim, Frame>& interface_unit_normal_vector,
    const double sign) noexcept;
/// @}
}  // namespace gr
