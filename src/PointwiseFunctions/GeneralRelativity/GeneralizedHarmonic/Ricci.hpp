// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Declares function templates to calculate the Ricci tensor

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"

/// \cond
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace GeneralizedHarmonic {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Compute spatial Ricci tensor using evolved variables and
 * their first derivatives.
 *
 * \details Lets write the Christoffel symbols of the first kind as
 * \f{align}
 * \Gamma_{kij} = \frac{1}{2}(\partial_i g_{jk} +
 *                             \partial_j g_{ik} -
 *                             \partial_k g_{ij})
 *              = (\Phi_{(ij)k} - \frac{1}{2}\Phi_{kij})
 * \f}
 * substituting \f$\partial_k g_{ij}\rightarrow{}\Phi_{kij}\f$ by
 * subtracting out the three-index constraint
 * \f$C_{kij}=\partial_{k}g_{ij}-\Phi_{kij}\f$ from every term. We also
 * define contractions \f$d_k=\frac{1}{2}g^{ij}\Phi_{kij}\f$ and
 * \f$b_k=\frac{1}{2}g^{ij}\Phi_{ijk}\f$. This allows us to rewrite the
 * spatial Ricci tensor as:
 * \f{align}
 * R_{i j} =& \partial_k \Gamma^{k}_{ij} - \partial_i \Gamma^{k}_{kj}
 *            + \Gamma^{k}_{kl}\Gamma^{l}_{ij}
 *            - \Gamma^{l}_{ki}\Gamma^{k}_{lj},\\
 *
 *         =& g^{kl}\left(\partial_{k}\Phi_{(ij)l} -
 *                        \frac{1}{2}\partial_{k}\Phi_{lij}\right)
 *            - b^{l} (\Phi_{(ij)l} - \frac{1}{2}\Phi_{lij})\nonumber\\
 *          & - g^{kl}\left(\partial_{i}\Phi_{(kj)l}
 *                          - \frac{1}{2}\partial_{i}\Phi_{lkj}\right)
 *            - \Phi_{i}{}^{kl}\left(\Phi_{(kj)l}
 *                                   - \frac{1}{2}\Phi_{lkj}
 *                                   \right)\nonumber\\
 *          & + g^{km}\left(\Phi_{(kl)m} - \frac{1}{2}\Phi_{mkl}\right)
 *              g^{ln}\left(\Phi_{(ij)n} - \frac{1}{2}\Phi_{nij}\right)
 *              \nonumber\\
 *
 *          & - g^{km}\left(\Phi_{(il)m} - \frac{1}{2}\Phi_{mil}\right)
 *              g^{ln}\left(\Phi_{(jk)n} - \frac{1}{2}\Phi_{njk}\right).
 * \f}
 * Gathering all terms with second derivatives:
 * \f{align}
 * R_{i j} =& \frac{1}{2} g^{k l} \left(\partial_k\Phi_{ijl}
 *                                      + \partial_k\Phi_{jil}
 *                                      - \partial_k\Phi_{lij}
 *                                      + \partial_i\Phi_{lkj}
 *                                      - \partial_i\Phi_{kjl}
 *                                      - \partial_i\Phi_{jkl}\right)
 *          + \mathcal{O}(\Phi), \nonumber\\
 *         =& \frac{1}{2} g^{kl} \left(\partial_{(j}\Phi_{lki)}
 *                                     - \partial_{(j}\Phi_{i)kl}
 *                                     + \partial_k \Phi_{(ij)l}
 *                                     - \partial_l \Phi_{kij} \right)
 *          + \mathcal{O}(\Phi),
 * \f}
 * where we use the four-index constraint
 * \f$C_{klij}=\partial_k\Phi_{lij}-\partial_l\Phi_{kij}=0\f$ to swap the
 * first and second derivatives of the spatial metric, and symmetrize
 * \f$R_{ij} = R_{(ij)}\f$. Similarly gathering the remaining terms and
 * using the four-index constraint we get:
 * \f{align}
 * R_{i j} =& - b^k\left(\Phi_{ijk} + \Phi_{jik} - \Phi_{kij}\right)
 *            -\frac{1}{2} \Phi_i{}^{kl} \left(\Phi_{jkl} + \Phi_{kjl}
 *                                            - \Phi_{lkj}\right)\nonumber\\
 *
 *         &+ \frac{1}{2} d^k \left(\Phi_{ijk} + \Phi_{jik} - \Phi_{kij}\right)
 *         - \left(\Phi_{(il)}{}^k - \frac{1}{2} \Phi^k{}_{il}\right)
 *           \left(\Phi_{(kj)}{}^l - \frac{1}{2} \Phi^l{}_{kj}\right)
 *         + \mathcal{O}(\partial\Phi) \\
 *
 *         =& \frac{1}{2} \left(\Phi_{ijk} + \Phi_{jik} - \Phi_{kij}\right)
 *            (d^k - 2 b^k)
 *         + \frac{1}{4} \Phi_{ik}{}^l \Phi_{jl}{}^k
 *         + \frac{1}{2} \left(\Phi^k{}_{il} \Phi_{kj}{}^l
 *                             - \Phi^k{}_{li} \Phi^l{}_{kj}\right)
 *         + \mathcal{O}(\partial\Phi).
 * \f}
 * Gathering everything together, we compute the spatial Ricci tensor as:
 * \f{eqnarray}\label{eq:rij}
 * R_{i j} &=& \frac{1}{2} g^{kl} \left(\partial_{(j|}\Phi_{lk|i)}
 *                                     - \partial_{(j}\Phi_{i)kl}
 *                                     + \partial_k \Phi_{(ij)l}
 *                                     - \partial_l \Phi_{kij}\right)\nonumber\\
 *         &+& \frac{1}{2} \left(\Phi_{ijk} + \Phi_{jik} - \Phi_{kij}\right)
 *            (d^k - 2 b^k)
 *          + \frac{1}{4} \Phi_{ik}{}^l \Phi_{jl}{}^k
 *          + \frac{1}{2} \left(\Phi^k{}_{il} \Phi_{kj}{}^l
 *                              - \Phi^k{}_{li} \Phi^l{}_{kj}\right).
 * \f}
 * This follows from equations (2.13) - (2.20) of \cite Kidder2001tz .
 *
 * Note that, in code, the mixed-index variables \f$\Phi_{ij}{}^k\f$ and
 * \f$\Phi^i{}_{jk}\f$ in Eq.(\f$\ref{eq:rij}\f$) are computed with a factor of
 * \f$1/2\f$ and so the last 3 terms in the same equation that are quadratic in
 * these terms occur multiplied by a factor of \f$4\f$.
 */
template <size_t VolumeDim, typename Frame, typename DataType>
void spatial_ricci_tensor(
    gsl::not_null<tnsr::ii<DataType, VolumeDim, Frame>*> ricci,
    const tnsr::iaa<DataType, VolumeDim, Frame>& phi,
    const tnsr::ijaa<DataType, VolumeDim, Frame>& deriv_phi,
    const tnsr::II<DataType, VolumeDim, Frame>& inverse_spatial_metric);

template <size_t VolumeDim, typename Frame, typename DataType>
tnsr::ii<DataType, VolumeDim, Frame> spatial_ricci_tensor(
    const tnsr::iaa<DataType, VolumeDim, Frame>& phi,
    const tnsr::ijaa<DataType, VolumeDim, Frame>& deriv_phi,
    const tnsr::II<DataType, VolumeDim, Frame>& inverse_spatial_metric);
/// @}
}  // namespace GeneralizedHarmonic
