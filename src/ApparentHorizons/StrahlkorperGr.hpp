// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

class DataVector;

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
/// \brief Expansion of a `Strahlkorper`. Should be zero on apparent horizons.
///
/// \details Implements Eq. (5) in
/// https://arxiv.org/abs/gr-qc/9606010.  The input argument
/// `grad_normal` is the quantity returned by
/// `StrahlkorperGr::grad_unit_normal_one_form`, and `unit_normal_vector` is
/// \f$S^i = g^{ij} S_j\f$ where \f$S_j\f$ is the unit normal one form.
template <typename Frame>
Scalar<DataVector> expansion(
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature) noexcept;

/// \ingroup SurfacesGroup
/// \brief Extrinsic curvature of a 2D `Strahlkorper` embedded in a 3D space.
///
/// \details Implements Eq. (D.43) of Carroll's Spacetime and Geometry text.
/// Specifically,
/// \f$ K_{ij} = P^k_i P^l_j \nabla_{(k} S_{l)} \f$, where
/// \f$ P^k_i = \delta^k_i - S^k S_i \f$,
/// `grad_normal` is the quantity \f$ \nabla_k S_l \f$ returned by
/// `StrahlkorperGr::grad_unit_normal_one_form`, and `unit_normal_vector` is
/// \f$S^i = g^{ij} S_j\f$ where \f$S_j\f$ is the unit normal one form.
/// Not to be confused with the extrinsic curvature of a 3D spatial slice
/// embedded in 3+1 spacetime.
template <typename Frame>
tnsr::ii<DataVector, 3, Frame> extrinsic_curvature(
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::II<DataVector, 3, Frame>& upper_spatial_metric) noexcept;

/// \ingroup SurfacesGroup
/// \brief Intrinsic Ricci scalar curvature of a 2D `Strahlkorper`.
///
/// \details Implements Eq. (D.51) of
/// Sean Carroll's Spacetime and Geometry textbook (except correcting
/// sign errors: both extrinsic curvature terms are off by a minus sign
/// in Carroll's text but correct in Carroll's errata).
/// \f$ \hat{R}=R - 2 R_{ij} S^i S^j + K^2-K^{ij}K_{ij}.\f$
/// Here \f$\hat{R}\f$ is the intrinsic Ricci scalar curavture of
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
}  // namespace StrahlkorperGr
