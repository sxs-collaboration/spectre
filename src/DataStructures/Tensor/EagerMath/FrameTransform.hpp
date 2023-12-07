// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
/// \endcond

/// Holds functions related to transforming between frames.
namespace transform {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * Transforms tensor to different frame.
 *
 * The formula for transforming \f$T_{ij}\f$ is
 * \f{align}
 *   T_{\bar{\imath}\bar{\jmath}} &= T_{ij}
 *      \frac{\partial x^i}{\partial x^{\bar{\imath}}}
 *      \frac{\partial x^j}{\partial x^{\bar{\jmath}}}
 * \f}
 * where \f$x^i\f$ are the source coordinates and
 * \f$x^{\bar{\imath}}\f$ are the destination coordinates.
 *
 * Note that `Jacobian<DestFrame,SrcFrame>` is the same type as
 * `InverseJacobian<SrcFrame,DestFrame>` and represents
 * \f$\partial x^i/\partial x^{\bar{\jmath}}\f$.
 */
template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::ii<DataType, VolumeDim, DestFrame>*> dest,
    const tnsr::ii<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian);

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    const tnsr::ii<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian)
    -> tnsr::ii<DataType, VolumeDim, DestFrame>;
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Transforms a tensor to a different frame.
 *
 * The tensor being transformed is always assumed to have density zero. In
 * particular `Scalar` is assumed to be invariant under transformations.
 *
 * \warning The \p jacobian argument is the derivative of the *source*
 * coordinates with respect to the *destination* coordinates.
 */
template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<Scalar<DataType>*> dest, const Scalar<DataType>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian);

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    Scalar<DataType> src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) -> Scalar<DataType>;

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::I<DataType, VolumeDim, DestFrame>*> dest,
    const tnsr::I<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian);

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    const tnsr::I<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) -> tnsr::I<DataType, VolumeDim, DestFrame>;

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::i<DataType, VolumeDim, DestFrame>*> dest,
    const tnsr::i<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian);

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    const tnsr::i<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) -> tnsr::i<DataType, VolumeDim, DestFrame>;

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::iJ<DataType, VolumeDim, DestFrame>*> dest,
    const tnsr::iJ<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian);

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    const tnsr::iJ<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) -> tnsr::iJ<DataType, VolumeDim, DestFrame>;

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::ii<DataType, VolumeDim, DestFrame>*> dest,
    const tnsr::ii<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian);

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    const tnsr::ii<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) -> tnsr::ii<DataType, VolumeDim, DestFrame>;

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::II<DataType, VolumeDim, DestFrame>*> dest,
    const tnsr::II<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian);

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    const tnsr::II<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) -> tnsr::II<DataType, VolumeDim, DestFrame>;

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
void to_different_frame(
    const gsl::not_null<tnsr::ijj<DataType, VolumeDim, DestFrame>*> dest,
    const tnsr::ijj<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian);

template <typename DataType, size_t VolumeDim, typename SrcFrame,
          typename DestFrame>
auto to_different_frame(
    const tnsr::ijj<DataType, VolumeDim, SrcFrame>& src,
    const Jacobian<DataType, VolumeDim, DestFrame, SrcFrame>& jacobian,
    const InverseJacobian<DataType, VolumeDim, DestFrame, SrcFrame>&
        inv_jacobian) -> tnsr::ijj<DataType, VolumeDim, DestFrame>;
/// @}

/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * Transforms only the first index to different frame.
 *
 * ## Examples for transforming some tensors
 *
 * ### Flux vector to logical coordinates
 *
 * A common example is taking the divergence of a flux vector $F^i$, where the
 * index $i$ is in inertial coordinates $x^i$ and the divergence is taken in
 * logical coordinates $\xi^\hat{i}$. So to transform the flux vector to logical
 * coordinates before taking the divergence we do this:
 *
 * \f{align}
 *   F^\hat{i} &= F^i \frac{\partial x^\hat{i}}{\partial x^i}
 * \f}
 *
 * Here, $\frac{\partial x^\hat{i}}{\partial x^i}$ is the inverse Jacobian (see
 * definitions in TypeAliases.hpp).
 *
 * Currently, this function is tested for any tensor with a first upper spatial
 * index. It can be extended/generalized to other tensor types if needed.
 */
template <typename ResultTensor, typename InputTensor, typename DataType,
          size_t Dim, typename SourceFrame, typename TargetFrame>
void first_index_to_different_frame(
    gsl::not_null<ResultTensor*> result, const InputTensor& input,
    const InverseJacobian<DataType, Dim, SourceFrame, TargetFrame>&
        inv_jacobian);

template <typename InputTensor, typename DataType, size_t Dim,
          typename SourceFrame, typename TargetFrame,
          typename ResultTensor = TensorMetafunctions::prepend_spatial_index<
              TensorMetafunctions::remove_first_index<InputTensor>, Dim,
              UpLo::Up, SourceFrame>>
ResultTensor first_index_to_different_frame(
    const InputTensor& input,
    const InverseJacobian<DataType, Dim, SourceFrame, TargetFrame>&
        inv_jacobian);
/// @}
}  // namespace transform
