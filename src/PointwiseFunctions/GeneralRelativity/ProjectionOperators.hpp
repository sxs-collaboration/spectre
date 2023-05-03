// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

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
 * \brief Compute projection operator onto an interface
 *
 * \details Returns the operator \f$P^{ij} = \gamma^{ij} - n^i n^j\f$,
 * where \f$\gamma^{ij}\f$ is the inverse spatial metric, and
 * \f$n^i\f$ is the normal vector to the interface in question.
 *
 */
template <typename DataType, size_t VolumeDim, typename Frame>
tnsr::II<DataType, VolumeDim, Frame> transverse_projection_operator(
    const tnsr::II<DataType, VolumeDim, Frame>& inverse_spatial_metric,
    const tnsr::I<DataType, VolumeDim, Frame>& normal_vector);

template <typename DataType, size_t VolumeDim, typename Frame>
void transverse_projection_operator(
    gsl::not_null<tnsr::II<DataType, VolumeDim, Frame>*> projection_tensor,
    const tnsr::II<DataType, VolumeDim, Frame>& inverse_spatial_metric,
    const tnsr::I<DataType, VolumeDim, Frame>& normal_vector);
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute projection operator onto an interface
 *
 * \details Returns the operator \f$P_{ij} = \gamma_{ij} - n_i n_j\f$,
 * where \f$ \gamma_{ij}\f$ is the spatial metric, and \f$ n_i\f$ is
 * the normal one-form to the interface in question.
 */
template <typename DataType, size_t VolumeDim, typename Frame>
tnsr::ii<DataType, VolumeDim, Frame> transverse_projection_operator(
    const tnsr::ii<DataType, VolumeDim, Frame>& spatial_metric,
    const tnsr::i<DataType, VolumeDim, Frame>& normal_one_form);

template <typename DataType, size_t VolumeDim, typename Frame>
void transverse_projection_operator(
    gsl::not_null<tnsr::ii<DataType, VolumeDim, Frame>*> projection_tensor,
    const tnsr::ii<DataType, VolumeDim, Frame>& spatial_metric,
    const tnsr::i<DataType, VolumeDim, Frame>& normal_one_form);
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute projection operator onto an interface
 *
 * \details Returns the operator \f$P^{i}_{j} = \delta^{i}_{j} - n^i n_j\f$,
 * where \f$n^i\f$ and \f$n_i\f$ are the normal vector and normal one-form
 * to the interface in question.
 */
template <typename DataType, size_t VolumeDim, typename Frame>
tnsr::Ij<DataType, VolumeDim, Frame> transverse_projection_operator(
    const tnsr::I<DataType, VolumeDim, Frame>& normal_vector,
    const tnsr::i<DataType, VolumeDim, Frame>& normal_one_form);

template <typename DataType, size_t VolumeDim, typename Frame>
void transverse_projection_operator(
    gsl::not_null<tnsr::Ij<DataType, VolumeDim, Frame>*> projection_tensor,
    const tnsr::I<DataType, VolumeDim, Frame>& normal_vector,
    const tnsr::i<DataType, VolumeDim, Frame>& normal_one_form);
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute spacetime projection operator onto an interface
 *
 * \details Consider a \f$d-1\f$-dimensional surface \f$S\f$ in a
 * \f$d\f$-dimensional spatial hypersurface \f$\Sigma\f$. Let \f$s_a\f$
 * be the unit spacelike one-form orthogonal to \f$S\f$ in \f$\Sigma\f$,
 * and \f$n_a\f$ be the timelike unit vector orthogonal to \f$\Sigma\f$.
 * This function returns the projection operator onto \f$S\f$ for
 * \f$d+1\f$ dimensional quantities:
 *
 * \f{align*}
 * P_{ab} = g_{ab} + n_a n_b - s_a s_b = \gamma_{ab} - s_a s_b.
 * \f}
 */
template <typename DataType, size_t VolumeDim, typename Frame>
tnsr::aa<DataType, VolumeDim, Frame> transverse_projection_operator(
    const tnsr::aa<DataType, VolumeDim, Frame>& spacetime_metric,
    const tnsr::a<DataType, VolumeDim, Frame>& spacetime_normal_one_form,
    const tnsr::i<DataType, VolumeDim, Frame>& interface_unit_normal_one_form);

template <typename DataType, size_t VolumeDim, typename Frame>
void transverse_projection_operator(
    gsl::not_null<tnsr::aa<DataType, VolumeDim, Frame>*> projection_tensor,
    const tnsr::aa<DataType, VolumeDim, Frame>& spacetime_metric,
    const tnsr::a<DataType, VolumeDim, Frame>& spacetime_normal_one_form,
    const tnsr::i<DataType, VolumeDim, Frame>& interface_unit_normal_one_form);
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute spacetime projection operator onto an interface
 *
 * \details Consider a \f$d-1\f$-dimensional surface \f$S\f$ in a
 * \f$d\f$-dimensional spatial hypersurface \f$\Sigma\f$. Let \f$s^a\f$
 * be the unit spacelike vector orthogonal to \f$S\f$ in \f$\Sigma\f$,
 * and \f$n^a\f$ be the timelike unit vector orthogonal to \f$\Sigma\f$.
 * This function returns the projection operator onto \f$S\f$ for
 * \f$d+1\f$ dimensional quantities:
 *
 * \f{align*}
 * P^{ab} = g^{ab} + n^a n^b - s^a s^b = \gamma_{ab} - s_a s_b.
 * \f}
 */
template <typename DataType, size_t VolumeDim, typename Frame>
tnsr::AA<DataType, VolumeDim, Frame> transverse_projection_operator(
    const tnsr::AA<DataType, VolumeDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, VolumeDim, Frame>& spacetime_normal_vector,
    const tnsr::I<DataType, VolumeDim, Frame>& interface_unit_normal_vector);

template <typename DataType, size_t VolumeDim, typename Frame>
void transverse_projection_operator(
    gsl::not_null<tnsr::AA<DataType, VolumeDim, Frame>*> projection_tensor,
    const tnsr::AA<DataType, VolumeDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, VolumeDim, Frame>& spacetime_normal_vector,
    const tnsr::I<DataType, VolumeDim, Frame>& interface_unit_normal_vector);
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute spacetime projection operator onto an interface
 *
 * \details Consider a \f$d-1\f$-dimensional surface \f$S\f$ in a
 * \f$d\f$-dimensional spatial hypersurface \f$\Sigma\f$. Let \f$s^a\f$
 * \f$(s_a)\f$ be the unit spacelike vector (one-form) orthogonal
 * to \f$S\f$ in \f$\Sigma\f$, and \f$n^a\f$ \f$(n_a)\f$ be the timelike
 * unit vector (one-form) orthogonal to \f$\Sigma\f$. This function
 * returns the projection operator onto \f$S\f$ for \f$d+1\f$ dimensional
 * quantities:
 *
 * \f{align*}
 * P^a_b = \delta^a_b + n^a n_b - s^a s_b.
 * \f}
 */
template <typename DataType, size_t VolumeDim, typename Frame>
tnsr::Ab<DataType, VolumeDim, Frame> transverse_projection_operator(
    const tnsr::A<DataType, VolumeDim, Frame>& spacetime_normal_vector,
    const tnsr::a<DataType, VolumeDim, Frame>& spacetime_normal_one_form,
    const tnsr::I<DataType, VolumeDim, Frame>& interface_unit_normal_vector,
    const tnsr::i<DataType, VolumeDim, Frame>& interface_unit_normal_one_form);

template <typename DataType, size_t VolumeDim, typename Frame>
void transverse_projection_operator(
    gsl::not_null<tnsr::Ab<DataType, VolumeDim, Frame>*> projection_tensor,
    const tnsr::A<DataType, VolumeDim, Frame>& spacetime_normal_vector,
    const tnsr::a<DataType, VolumeDim, Frame>& spacetime_normal_one_form,
    const tnsr::I<DataType, VolumeDim, Frame>& interface_unit_normal_vector,
    const tnsr::i<DataType, VolumeDim, Frame>& interface_unit_normal_one_form);
/// @}
}  // namespace gr
