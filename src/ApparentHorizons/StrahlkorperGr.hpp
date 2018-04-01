// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ApparentHorizons/StrahlkorperDataBox.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

/// \ingroup SurfacesGroup
/// Contains functions that depend both on a Strahlkorper and a metric.
namespace StrahlkorperGr {

/// \ingroup SurfacesGroup
/// \brief Computes normalized unit normal one-form to a Strahlkorper.
///
/// \details The input argument `normal_one_form` \f$n_i\f$ is the
/// unnormalized surface one-form; it depends on a Strahlkorper but
/// not on a metric.  The input argument `one_over_one_form_magnitude`
/// is \f$1/\sqrt{g^{ij}n_i n_j}\f$, which can be computed using (one
/// over) the `magnitude` function.
template <typename Frame>
tnsr::i<DataVector, 3, Frame> unit_normal_one_form(
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const DataVector& one_over_one_form_magnitude) noexcept;

/// \ingroup SurfacesGroup
/// \brief Computes 3-covariant gradient \f$D_i S_j\f$ of a
/// Strahlkorper's normal.
///
/// \details See Eqs. (1--9) of https://arxiv.org/abs/gr-qc/9606010.
/// Here \f$S_j\f$ is the (normalized) unit one-form to the surface,
/// and \f$D_i\f$ is the spatial covariant derivative.  Note that this
/// object is symmetric, even though this is not obvious from the
/// definition.  The input arguments `r_hat`, `radius`, and
/// `d2x_radius` depend on the Strahlkorper but not on the metric, and
/// can be computed from a Strahlkorper using ComputeItems in
/// `StrahlkorperTags`.  The input argument
/// `one_over_one_form_magnitude` is \f$1/\sqrt{g^{ij}n_i n_j}\f$,
/// where \f$n_i\f$ is `StrahlkorperTags::NormalOneForm` (i.e.  the
/// unnormalized one-form to the Strahlkorper); it can be computed
/// using (one over) the `magnitude` function.  The input argument
/// `unit_normal_one_form` is \f$S_j\f$,the normalized one-form.
template <typename Frame>
tnsr::ii<DataVector, 3, Frame> grad_unit_normal_one_form(
    const tnsr::i<DataVector, 3, Frame>& r_hat, const DataVector& radius,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, 3, Frame>& d2x_radius,
    const DataVector& one_over_one_form_magnitude,
    const tnsr::Ijj<DataVector, 3, Frame>& christoffel_2nd_kind) noexcept;

/// \ingroup SurfacesGroup
/// \brief Computes inverse 2-metric \f$g^{ij}-S^i S^j\f$ of a Strahlkorper.
///
/// \details See Eqs. (1--9) of https://arxiv.org/abs/gr-qc/9606010.
/// Here \f$S^i\f$ is the (normalized) unit vector to the surface,
/// and \f$g^{ij}\f$ is the 3-metric.  This object is expressed in the
/// usual 3-d Cartesian basis, so it is written as a 3-dimensional tensor.
/// But because it is orthogonal to \f$S_i\f$, it has only 3 independent
/// degrees of freedom, and could be expressed as a 2-d tensor with an
/// appropriate choice of basis. The input argument `unit_normal_vector` is
/// \f$S^i = g^{ij} S_j\f$, where \f$S_j\f$ is the unit normal one form.
template <typename Frame>
tnsr::II<DataVector, 3, Frame> inverse_surface_metric(
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric) noexcept;

/// \ingroup SurfacesGroup
/// \brief Expansion of a `Strahlkorper`. Should be zero on apparent horizons.
///
/// \details Implements Eq. (5) in
/// https://arxiv.org/abs/gr-qc/9606010.  The input argument
/// `grad_normal` is the quantity returned by
/// `StrahlkorperGr::grad_unit_normal_one_form`, and `inverse_surface_metric`
/// is the quantity returned by `StrahlkorperGr::inverse_surface_metric`.
template <typename Frame>
Scalar<DataVector> expansion(
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::II<DataVector, 3, Frame>& inverse_surface_metric,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature) noexcept;

/*!
 * \ingroup SurfacesGroup
 * \brief Extrinsic curvature of a 2D `Strahlkorper` embedded in a 3D space.
 *
 * \details Implements Eq. (D.43) of Carroll's Spacetime and Geometry text.
 * Specifically,
 * \f$ K_{ij} = P^k_i P^l_j \nabla_{(k} S_{l)} \f$, where
 * \f$ P^k_i = \delta^k_i - S^k S_i \f$,
 * `grad_normal` is the quantity \f$ \nabla_k S_l \f$ returned by
 * `StrahlkorperGr::grad_unit_normal_one_form`, and `unit_normal_vector` is
 * \f$S^i = g^{ij} S_j\f$ where \f$S_j\f$ is the unit normal one form.
 * Not to be confused with the extrinsic curvature of a 3D spatial slice
 * embedded in 3+1 spacetime.
 * Because StrahlkorperGr::grad_unit_normal_one_form is symmetric, this
 * can be expanded into
 * \f$ K_{ij} = \nabla_{i}S_{j} - 2 S^k S_{(i}\nabla_{j)}S_k
 * + S_i S_j S^k S^l \nabla_{k} n_{l}\f$.
 */
template <typename Frame>
tnsr::ii<DataVector, 3, Frame> extrinsic_curvature(
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector) noexcept;

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
/// \f$ K_{ij} \f$ the output of StrahlkorperGr::extrinsic_curvature,
/// \f$ K \f$ is the trace of \f$K_{ij}\f$,
/// and `unit_normal_vector` is
/// \f$S^i = g^{ij} S_j\f$ where \f$S_j\f$ is the unit normal one form.
template <typename Frame>
Scalar<DataVector> ricci_scalar(
    const tnsr::ii<DataVector, 3, Frame>& spatial_ricci_tensor,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric) noexcept;

/*!
 * \ingroup SurfacesGroup
 * \brief Area element of a 2D `Strahlkorper`.
 *
 * \details Implements Eq. (D.13), using Eqs. (D.4) and (D.5),
 * of https://arxiv.org/abs/gr-qc/9606010.
 * Specifically, computes
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
 * of \f$\sin\theta\f$, i.e., this returns \f$r^2\f$ for flat space.
 * This choice makes the area element returned here compatible with
 * `definite_integral` defined in `YlmSpherePack.hpp`.
 */
template <typename Frame>
Scalar<DataVector> area_element(
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const StrahlkorperTags::StrahlkorperTags_detail::Jacobian<Frame>& jacobian,
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const DataVector& radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat) noexcept;
}  // namespace StrahlkorperGr
