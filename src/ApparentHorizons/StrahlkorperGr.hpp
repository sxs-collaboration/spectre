// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "ApparentHorizons/TagsTypeAliases.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
class YlmSpherepack;
template <typename Frame>
class Strahlkorper;
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
/// \details See Eqs. (1--9) of \cite Baumgarte1996hh.
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
/// \details See Eqs. (1--9) of \cite Baumgarte1996hh.
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
/// \details Implements Eq. (5) in \cite Baumgarte1996hh.  The input argument
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
Scalar<DataVector> area_element(
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const StrahlkorperTags::aliases::Jacobian<Frame>& jacobian,
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const DataVector& radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat) noexcept;

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
Scalar<DataVector> euclidean_area_element(
    const StrahlkorperTags::aliases::Jacobian<Frame>& jacobian,
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const DataVector& radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat) noexcept;

/*!
 * \ingroup SurfacesGroup
 * \brief Surface integral of a scalar on a 2D `Strahlkorper`
 *
 * \details Computes the surface integral \f$\oint dA f\f$ for a scalar \f$f\f$
 * on a `Strahlkorper` with area element \f$dA\f$. The area element can be
 * computed via `StrahlkorperGr::area_element()`.
 */
template <typename Frame>
double surface_integral_of_scalar(
    const Scalar<DataVector>& area_element, const Scalar<DataVector>& scalar,
    const Strahlkorper<Frame>& strahlkorper) noexcept;

/*!
 * \ingroup SurfacesGroup
 * \brief Spin function of a 2D `Strahlkorper`.
 *
 * \details See Eqs. (2) and (10)
 * of \cite Owen2017yaj. This function computes the
 * "spin function," which is an ingredient for horizon surface integrals that
 * measure quasilocal spin. This function is proportional to the imaginary part
 * of the horizon's complex scalar curvature. For Kerr black holes, the spin
 * function is proportional to the horizon vorticity. It is also useful for
 * visualizing the direction of a black hole's spin.
 * Specifically, this function computes
 * \f$\Omega = \epsilon^{AB}\nabla_A\omega_B\f$,
 * where capital indices index the tangent bundle of the surface and
 * where \f$\omega_\mu=(K_{\rho\nu}-K g_{\rho\nu})h_\mu^\rho s^\nu\f$ is
 * the curl of the angular momentum density of the surface,
 * \f$h^\rho_\mu = \delta_\mu^\rho + u_\mu u^\rho - s_\mu s^\rho\f$
 * is the projector tangent to the 2-surface,
 * \f$g_{\rho\nu} = \psi_{\rho\nu} + u_\rho u_\nu\f$ is the spatial
 * metric of the spatial slice, \f$u^\rho\f$ is the unit normal to the
 * spatial slice, and \f$s^\nu\f$ is the unit normal vector to the surface.
 * Because the tangent basis vectors \f$e_A^\mu\f$ are
 * orthogonal to both \f$u^\mu\f$ and \f$s^\mu\f$, it is straightforward
 * to show that \f$\Omega = \epsilon^{AB} \nabla_A K_{B\mu}s^\mu\f$.
 * This function uses the tangent vectors of the `Strahlkorper` to
 * compute \f$K_{B\mu}s^\mu\f$ and then numerically computes the
 * components of its gradient. The argument `area_element`
 * can be computed via `StrahlkorperGr::area_element`.
 * The argument `unit_normal_vector` can be found by raising the
 * index of the one-form returned by `StrahlkorperGr::unit_normal_oneform`.
 * The argument `tangents` is a Tangents that can be obtained from the
 * StrahlkorperDataBox using the `StrahlkorperTags::Tangents` tag.
 */
template <typename Frame>
Scalar<DataVector> spin_function(
    const StrahlkorperTags::aliases::Jacobian<Frame>& tangents,
    const YlmSpherepack& ylm,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const Scalar<DataVector>& area_element,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature) noexcept;

/*!
 * \ingroup SurfacesGroup
 * \brief Spin magnitude measured on a 2D `Strahlkorper`.
 *
 * \details Measures the quasilocal spin magnitude of a Strahlkorper, by
 * inserting \f$\alpha=1\f$ into Eq. (10) of \cite Owen2009sb
 * and dividing by \f$8\pi\f$ to yield the spin magnitude. The
 * spin magnitude is a Euclidean norm of surface integrals over the horizon
 * \f$S = \frac{1}{8\pi}\oint z \Omega dA\f$,
 * where \f$\Omega\f$ is obtained via `StrahlkorperGr::spin_function()`,
 * \f$dA\f$ is the area element, and \f$z\f$ (the "spin potential") is a
 * solution of a generalized eigenproblem given by Eq. (9) of
 * \cite Owen2009sb. Specifically,
 * \f$\nabla^4 z + \nabla \cdot (R\nabla z) = \lambda \nabla^2 z\f$, where
 * \f$R\f$ is obtained via `StrahlkorperGr::ricci_scalar()`. The spin
 * magnitude is the Euclidean norm of the three values of \f$S\f$ obtained from
 * the eigenvectors \f$z\f$ with the 3 smallest-magnitude
 * eigenvalues \f$\lambda\f$. Note that this formulation of the eigenproblem
 * uses the "Owen" normalization, Eq. (A9) and surrounding discussion in
 * \cite Lovelace2008tw.
 * The eigenvectors are normalized  with the "Kerr normalization",
 * Eq. (A22) of \cite Lovelace2008tw.
 * The argument `spatial_metric` is the metric of the 3D spatial slice
 * evaluated on the `Strahlkorper`.
 * The argument `tangents` can be obtained from the StrahlkorperDataBox
 * using the `StrahlkorperTags::Tangents` tag, and the argument
 * `unit_normal_vector` can
 * be found by raising the index of the one-form returned by
 * `StrahlkorperGr::unit_normal_one_form`.
 * The argument `ylm` is the `YlmSpherepack` of the `Strahlkorper`.
 * The argument `area_element`
 * can be computed via `StrahlkorperGr::area_element`.
 */
template <typename Frame>
double dimensionful_spin_magnitude(
    const Scalar<DataVector>& ricci_scalar,
    const Scalar<DataVector>& spin_function,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const StrahlkorperTags::aliases::Jacobian<Frame>& tangents,
    const YlmSpherepack& ylm, const Scalar<DataVector>& area_element) noexcept;

/*!
 * \ingroup SurfacesGroup
 * \brief Spin vector of a 2D `Strahlkorper`.
 *
 * \details Computes the spin vector of a `Strahlkorper` in a `Frame`, such as
 * `Frame::Inertial`. The result is a `std::array<double, 3>` containing the
 * Cartesian components of the spin vector, whose magnitude is
 * `spin_magnitude`. This function will return the dimensionless spin
 * components if `spin_magnitude` is the dimensionless spin magnitude, and
 * it will return the dimensionful spin components if `spin_magnitude` is
 * the dimensionful spin magnitude. The spin vector is given by
 * a surface integral over the horizon \f$\mathcal{H}\f$ [Eq. (25) of
 * \cite Owen2017yaj]:
 * \f$S^i = \frac{S}{N} \oint_\mathcal{H} dA \Omega (x^i - x^i_0 - x^i_R) \f$,
 * where \f$S\f$ is the spin magnitude,
 * \f$N\f$ is a normalization factor enforcing \f$\delta_{ij}S^iS^j = S\f$,
 * \f$dA\f$ is the area element (via `StrahlkorperGr::area_element`),
 * \f$\Omega\f$ is the "spin function" (via `StrahlkorperGr::spin_function`),
 * \f$x^i\f$ are the `Frame` coordinates of points on the `Strahlkorper`,
 * \f$x^i_0\f$ are the `Frame` coordinates of the center of the Strahlkorper,
 * \f$x^i_R = \frac{1}{8\pi}\oint_\mathcal{H} dA (x^i - x^i_0) R \f$,
 * and \f$R\f$ is the intrinsic Ricci scalar of the `Strahlkorper`
 * (via `StrahlkorperGr::ricci_scalar`).
 * Note that measuring positions on the horizon relative to
 * \f$x^i_0 + x^i_R\f$ instead of \f$x^i_0\f$ ensures that the mass dipole
 * moment vanishes. Also note that \f$x^i - x^i_0\f$ is
 * is the product of `StrahlkorperTags::Rhat` and `StrahlkorperTags::Radius`.
 */
template <typename Frame>
std::array<double, 3> spin_vector(double spin_magnitude,
                                  const Scalar<DataVector>& area_element,
                                  const Scalar<DataVector>& radius,
                                  const tnsr::i<DataVector, 3, Frame>& r_hat,
                                  const Scalar<DataVector>& ricci_scalar,
                                  const Scalar<DataVector>& spin_function,
                                  const YlmSpherepack& ylm) noexcept;

/*!
 * \ingroup SurfacesGroup
 * \brief Irreducible mass of a 2D `Strahlkorper`.
 *
 * \details See Eqs. (15.38) \cite Hartle2003gravity. This function computes the
 * irreducible mass from the area of a horizon. Specifically, computes
 * \f$M_\mathrm{irr}=\sqrt{\frac{A}{16\pi}}\f$.
 */
double irreducible_mass(double area) noexcept;

/*!
 * \ingroup SurfacesGroup
 * \brief Christodoulou Mass of a 2D `Strahlkorper`.
 *
 * \details See e.g. Eq. (1) of \cite Lovelace2016uwp.
 * This function computes the Christodoulou mass from the dimensionful
 * spin angular momentum \f$S\f$ and the irreducible mass \f$M_{irr}\f$
 * of a black hole horizon. Specifically, computes
 *\f$M=\sqrt{M_{irr}^2+\frac{S^2}{4M_{irr}^2}}\f$
 */
double christodoulou_mass(double dimensionful_spin_magnitude,
                          double irreducible_mass) noexcept;
}  // namespace StrahlkorperGr
