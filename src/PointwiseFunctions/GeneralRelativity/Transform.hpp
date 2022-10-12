// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
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
 * Often used for derivatives: When representing derivatives as
 * tensors, the first index is typically the derivative index.
 * Numerical derivatives must be computed in the logical frame or
 * sometimes the grid frame (independent of the frame of the tensor
 * being differentiated), and then that derivative index must later
 * be transformed into the same frame as the other indices of the
 * tensor.
 *
 * The formula for transforming \f$T_{i\bar{\jmath}\bar{k}}\f$ is
 * \f{align}
 *   T_{\bar{\imath}\bar{\jmath}\bar{k}} &= T_{i\bar{\jmath}\bar{k}}
 *      \frac{\partial x^i}{\partial x^{\bar{\imath}}},
 * \f}
 * where \f$x^i\f$ are the source coordinates and
 * \f$x^{\bar{\imath}}\f$ are the destination coordinates.
 *
 * Note that `Jacobian<DestFrame,SrcFrame>` is the same type as
 * `InverseJacobian<SrcFrame,DestFrame>` and represents
 * \f$\partial x^i/\partial x^{\bar{\jmath}}\f$.
 *
 * In principle `first_index_to_different_frame` can be
 * extended/generalized to other tensor types if needed.
 */
template <size_t VolumeDim, typename SrcFrame, typename DestFrame>
void first_index_to_different_frame(
    const gsl::not_null<tnsr::ijj<DataVector, VolumeDim, DestFrame>*> dest,
    const Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                 index_list<SpatialIndex<VolumeDim, UpLo::Lo, SrcFrame>,
                            SpatialIndex<VolumeDim, UpLo::Lo, DestFrame>,
                            SpatialIndex<VolumeDim, UpLo::Lo, DestFrame>>>& src,
    const Jacobian<DataVector, VolumeDim, DestFrame, SrcFrame>& jacobian);

template <size_t VolumeDim, typename SrcFrame, typename DestFrame>
auto first_index_to_different_frame(
    const Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1, 1>,
                 index_list<SpatialIndex<VolumeDim, UpLo::Lo, SrcFrame>,
                            SpatialIndex<VolumeDim, UpLo::Lo, DestFrame>,
                            SpatialIndex<VolumeDim, UpLo::Lo, DestFrame>>>& src,
    const Jacobian<DataVector, VolumeDim, DestFrame, SrcFrame>& jacobian)
    -> tnsr::ijj<DataVector, VolumeDim, DestFrame>;
/// @}
}  // namespace transform
