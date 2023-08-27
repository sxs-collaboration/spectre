// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr::surfaces {
/// @{
/// \ingroup SurfacesGroup
/// \brief Expansion of a `Strahlkorper`. Should be zero on apparent horizons.
///
/// \details Implements Eq. (5) in \cite Baumgarte1996hh.  The input argument
/// `grad_normal` is the quantity returned by
/// `gr::surfaces::grad_unit_normal_one_form`, and `inverse_surface_metric`
/// is the quantity returned by `gr::surfaces::inverse_surface_metric`.
template <typename Frame>
void expansion(gsl::not_null<Scalar<DataVector>*> result,
               const tnsr::ii<DataVector, 3, Frame>& grad_normal,
               const tnsr::II<DataVector, 3, Frame>& inverse_surface_metric,
               const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature);

template <typename Frame>
Scalar<DataVector> expansion(
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::II<DataVector, 3, Frame>& inverse_surface_metric,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature);
/// @}
}  // namespace gr::surfaces
