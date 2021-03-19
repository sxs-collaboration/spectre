// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
class DataVector;
/// \endcond

namespace GeneralizedHarmonic {
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the covariant derivative of extrinsic curvature from
 * generalized harmonic variables and the spacetime normal vector.
 *
 * \details If \f$ \Pi_{ab} \f$ and \f$ \Phi_{iab} \f$ are the generalized
 * harmonic conjugate momentum and spatial derivative variables, and if
 * \f$n^a\f$ is the spacetime normal vector, then the extrinsic curvature
 * can be written as
 * \f{equation}\label{eq:kij}
 *     K_{ij} = \frac{1}{2} \Pi_{ij} + \Phi_{(ij)a} n^a,
 * \f}
 * and its covariant derivative as
 * \f{equation}\label{eq:covkij}
 * \nabla_k K_{ij} = \partial_k K_{ij} - \Gamma^l{}_{ik} K_{lj}
 *                                     - \Gamma^l{}_{jk} K_{li},
 * \f}
 * where \f$\Gamma^k{}_{ij}\f$ are Christoffel symbols of the second kind.
 * The partial derivatives of extrinsic curvature can be computed as
 * \f{equation}\label{eq:pdkij}
 * \partial_k K_{ij} =
 *     \frac{1}{2}\left(\partial_k \Pi_{ij} +
 *                      \left(\partial_k \Phi_{ija} +
 *                            \partial_k \Phi_{jia}\right) n^a +
 *                      \left(\Phi_{ija} + \Phi_{jia}\right) \partial_k n^a
 *               \right),
 * \f}
 * where we have access to all terms except the spatial derivatives of the
 * spacetime unit normal vector \f$\partial_k n^a\f$. Given that
 * \f$n^a=(1/\alpha, -\beta^i /\alpha)\f$, the temporal portion of
 * \f$\partial_k n^a\f$ can be computed as:
 * \f{align}
 * \partial_k n^0 =& -\frac{1}{\alpha^2} \partial_k \alpha, \nonumber \\
 *                =& -\frac{1}{\alpha^2} (-\alpha/2) n^a \Phi_{kab} n^b,
 *                   \nonumber \\
 *                =& \frac{1}{2\alpha} n^a \Phi_{kab} n^b, \nonumber \\
 *                =& \frac{1}{2} n^0 n^a \Phi_{kab} n^b, \nonumber \\
 *                =& -\left(g^{0a} +
 *                          \frac{1}{2}n^0 n^a\right) \Phi_{kab} n^b,
 * \f}
 * where we use the expression for \f$\partial_k \alpha\f$ from
 * \ref spatial_deriv_of_lapse; while the spatial portion of the same can be
 * computed as:
 * \f{align}
 * \partial_k n^i =& -\partial_k (\beta^i/\alpha)
 *                 = -\frac{1}{\alpha}\partial_k \beta^i
 *                   + \frac{\beta^i}{\alpha^2}\partial_k \alpha ,\nonumber \\
 *                =& -\frac{1}{2}\frac{\beta^i}{\alpha} n^a\Phi_{kab}n^b
 *                   -\left(g^{ia}
 *                          + n^i n^a\right) \Phi_{kab} n^b, \nonumber\\
 *                =& -\left(g^{ia} + \frac{1}{2}n^i n^a\right) \Phi_{kab}n^b,
 * \f}
 * where we use the expression for \f$\partial_k \beta^i\f$ from
 * \ref spatial_deriv_of_shift. Combining the last two equations, we find that
 * \f{equation}
 * \partial_k n^a = -\left(g^{ab} + \frac{1}{2}n^a n^b\right)\Phi_{kbc}n^c,
 * \f}
 * and using Eq.(\f$\ref{eq:covkij}\f$) and Eq.(\f$\ref{eq:pdkij}\f$) with this,
 * we can compute the covariant derivative of the extrinsic curvature as:
 * \f{equation}
 * \nabla_k K_{ij} =
 *     \frac{1}{2}\left(\partial_k \Pi_{ij} +
 *                      \left(\partial_k \Phi_{ija} +
 *                            \partial_k \Phi_{jia}\right) n^a -
 *                      \left(\Phi_{ija} + \Phi_{jia}\right)
 *                      \left(g^{ab} + \frac{1}{2}n^a n^b\right)
 *                      \Phi_{kbc}n^c
 *                \right) - \Gamma^l{}_{ik} K_{lj} - \Gamma^l{}_{jk} K_{li} \f}.
 */
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ijj<DataType, SpatialDim, Frame> covariant_deriv_of_extrinsic_curvature(
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal_vector,
    const tnsr::Ijj<DataType, SpatialDim, Frame>&
        spatial_christoffel_second_kind,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
void covariant_deriv_of_extrinsic_curvature(
    gsl::not_null<tnsr::ijj<DataType, SpatialDim, Frame>*>
        d_extrinsic_curvature,
    const tnsr::ii<DataType, SpatialDim, Frame>& extrinsic_curvature,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal_vector,
    const tnsr::Ijj<DataType, SpatialDim, Frame>&
        spatial_christoffel_second_kind,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) noexcept;
}  // namespace GeneralizedHarmonic
