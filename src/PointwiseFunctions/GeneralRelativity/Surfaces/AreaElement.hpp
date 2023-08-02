// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TagsTypeAliases.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace StrahlkorperGr {
/// @{
/*!
 * \ingroup SurfacesGroup
 * \brief Area element of a 2D `Strahlkorper`.
 *
 * \details Implements Eq. (D.13), using Eqs. (D.4) and (D.5),
 * of \cite Baumgarte1996hh. Specifically, computes
 * \f$\sqrt{(\Theta^i\Theta_i)(\Phi^j\Phi_j)-(\Theta^i\Phi_i)^2}\f$,
 * \f$\Theta^i=\left(n^i(n_j-s_j) r J^j_\theta + r J^i_\theta\right)\f$,
 * \f$\Phi^i=\left(n^i(n_j-s_j)r J^j_\phi + r J^i_\phi\right)\f$,
 * and \f$\Theta^i\f$ and \f$\Phi^i\f$ are lowered by the
 * 3D spatial metric \f$g_{ij}\f$. Here \f$J^i_\alpha\f$, \f$s_j\f$,
 * \f$r\f$, and \f$n^i=n_i\f$ correspond to the input arguments
 * `jacobian`, `normal_one_form`, `radius`, and `r_hat`, respectively;
 * these input arguments depend only on the Strahlkorper, not on the
 * metric, and can be computed from a Strahlkorper using ComputeItems
 * in `StrahlkorperTags`. Note that this does not include the factor
 * of \f$\sin\theta\f$, i.e., this returns \f$r^2\f$ for a spherical
 * `Strahlkorper` in flat space.
 * This choice makes the area element returned here compatible with
 * `definite_integral` defined in `YlmSpherePack.hpp`.
 */
template <typename Frame>
void area_element(gsl::not_null<Scalar<DataVector>*> result,
                  const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
                  const StrahlkorperTags::aliases::Jacobian<Frame>& jacobian,
                  const tnsr::i<DataVector, 3, Frame>& normal_one_form,
                  const Scalar<DataVector>& radius,
                  const tnsr::i<DataVector, 3, Frame>& r_hat);

template <typename Frame>
Scalar<DataVector> area_element(
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const StrahlkorperTags::aliases::Jacobian<Frame>& jacobian,
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat);
/// @}

/// @{
/*!
 * \ingroup SurfacesGroup
 * \brief Euclidean area element of a 2D `Strahlkorper`.
 *
 * This is useful for computing a flat-space integral over an
 * arbitrarily-shaped `Strahlkorper`.
 *
 * \details Implements Eq. (D.13), using Eqs. (D.4) and (D.5),
 * of \cite Baumgarte1996hh. Specifically, computes
 * \f$\sqrt{(\Theta^i\Theta_i)(\Phi^j\Phi_j)-(\Theta^i\Phi_i)^2}\f$,
 * \f$\Theta^i=\left(n^i(n_j-s_j) r J^j_\theta + r J^i_\theta\right)\f$,
 * \f$\Phi^i=\left(n^i(n_j-s_j)r J^j_\phi + r J^i_\phi\right)\f$,
 * and \f$\Theta^i\f$ and \f$\Phi^i\f$ are lowered by the
 * Euclidean spatial metric. Here \f$J^i_\alpha\f$, \f$s_j\f$,
 * \f$r\f$, and \f$n^i=n_i\f$ correspond to the input arguments
 * `jacobian`, `normal_one_form`, `radius`, and `r_hat`, respectively;
 * these input arguments depend only on the Strahlkorper, not on the
 * metric, and can be computed from a Strahlkorper using ComputeItems
 * in `StrahlkorperTags`. Note that this does not include the factor
 * of \f$\sin\theta\f$, i.e., this returns \f$r^2\f$ for a spherical
 * `Strahlkorper`.
 * This choice makes the area element returned here compatible with
 * `definite_integral` defined in `YlmSpherePack.hpp`.
 */
template <typename Frame>
void euclidean_area_element(
    gsl::not_null<Scalar<DataVector>*> result,
    const StrahlkorperTags::aliases::Jacobian<Frame>& jacobian,
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat);

template <typename Frame>
Scalar<DataVector> euclidean_area_element(
    const StrahlkorperTags::aliases::Jacobian<Frame>& jacobian,
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat);
/// @}
}  // namespace StrahlkorperGr
