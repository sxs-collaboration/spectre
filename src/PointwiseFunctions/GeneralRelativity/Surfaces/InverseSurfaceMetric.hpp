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
/// \brief Computes inverse 2-metric \f$g^{ij}-S^i S^j\f$ of a Strahlkorper.
///
/// \details See Eqs. (1--9) of \cite Baumgarte1996hh.
/// Here \f$S^i\f$ is the (normalized) unit vector to the surface,
/// and \f$g^{ij}\f$ is the 3-metric.  This object is expressed in the
/// usual 3-d Cartesian basis, so it is written as a 3-dimensional tensor.
/// But because it is orthogonal to \f$S_i\f$, it has only 3 independent
/// degrees of freedom, and could be expressed as a 2-d tensor with an
/// appropriate choice of basis. The input argument `unit_normal_vector` is
/// \f$S^i = g^{ij} S_j\f$, where \f$S_j\f$ is the unit normal one form.
template <typename Frame>
void inverse_surface_metric(
    gsl::not_null<tnsr::II<DataVector, 3, Frame>*> result,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric);

template <typename Frame>
tnsr::II<DataVector, 3, Frame> inverse_surface_metric(
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric);
/// @}
}  // namespace gr::surfaces
