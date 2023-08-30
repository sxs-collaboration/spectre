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
/// \brief Intrinsic Ricci scalar of a 2D `Strahlkorper`.
///
/// \details Implements Eq. (D.51) of
/// Sean Carroll's Spacetime and Geometry textbook (except correcting
/// sign errors: both extrinsic curvature terms are off by a minus sign
/// in Carroll's text but correct in Carroll's errata).
/// \f$ \hat{R}=R - 2 R_{ij} S^i S^j + K^2-K^{ij}K_{ij}.\f$
/// Here \f$\hat{R}\f$ is the intrinsic Ricci scalar curvature of
/// the Strahlkorper, \f$R\f$ and \f$R_{ij}\f$ are the Ricci scalar and
/// Ricci tensor of the 3D space that contains the Strahlkorper,
/// \f$ K_{ij} \f$ the output of gr::surfaces::extrinsic_curvature,
/// \f$ K \f$ is the trace of \f$K_{ij}\f$,
/// and `unit_normal_vector` is
/// \f$S^i = g^{ij} S_j\f$ where \f$S_j\f$ is the unit normal one form.
template <typename Frame>
void ricci_scalar(gsl::not_null<Scalar<DataVector>*> result,
                  const tnsr::ii<DataVector, 3, Frame>& spatial_ricci_tensor,
                  const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
                  const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
                  const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric);

template <typename Frame>
Scalar<DataVector> ricci_scalar(
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci_tensor,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric);
/// @}
}  // namespace gr::surfaces
