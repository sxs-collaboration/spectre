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

namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

/// \ingroup SurfacesGroup
/// Contains functions that depend both on a Strahlkorper and a metric.
namespace StrahlkorperGr {

/// @{
/// \ingroup SurfacesGroup
/// \brief Computes normalized unit normal one-form to a Strahlkorper.
///
/// \details The input argument `normal_one_form` \f$n_i\f$ is the
/// unnormalized surface one-form; it depends on a Strahlkorper but
/// not on a metric.  The input argument `one_over_one_form_magnitude`
/// is \f$1/\sqrt{g^{ij}n_i n_j}\f$, which can be computed using (one
/// over) the `magnitude` function.
template <typename Frame>
void unit_normal_one_form(gsl::not_null<tnsr::i<DataVector, 3, Frame>*> result,
                          const tnsr::i<DataVector, 3, Frame>& normal_one_form,
                          const DataVector& one_over_one_form_magnitude);

template <typename Frame>
tnsr::i<DataVector, 3, Frame> unit_normal_one_form(
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const DataVector& one_over_one_form_magnitude);
/// @}

/// @{
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
void grad_unit_normal_one_form(
    gsl::not_null<tnsr::ii<DataVector, 3, Frame>*> result,
    const tnsr::i<DataVector, 3, Frame>& r_hat,
    const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, 3, Frame>& d2x_radius,
    const DataVector& one_over_one_form_magnitude,
    const tnsr::Ijj<DataVector, 3, Frame>& christoffel_2nd_kind);

template <typename Frame>
tnsr::ii<DataVector, 3, Frame> grad_unit_normal_one_form(
    const tnsr::i<DataVector, 3, Frame>& r_hat,
    const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::ii<DataVector, 3, Frame>& d2x_radius,
    const DataVector& one_over_one_form_magnitude,
    const tnsr::Ijj<DataVector, 3, Frame>& christoffel_2nd_kind);
/// @}

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

/// @{
/// \ingroup SurfacesGroup
/// \brief Expansion of a `Strahlkorper`. Should be zero on apparent horizons.
///
/// \details Implements Eq. (5) in \cite Baumgarte1996hh.  The input argument
/// `grad_normal` is the quantity returned by
/// `StrahlkorperGr::grad_unit_normal_one_form`, and `inverse_surface_metric`
/// is the quantity returned by `StrahlkorperGr::inverse_surface_metric`.
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

/// @{
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
void extrinsic_curvature(
    gsl::not_null<tnsr::ii<DataVector, 3, Frame>*> result,
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector);

template <typename Frame>
tnsr::ii<DataVector, 3, Frame> extrinsic_curvature(
    const tnsr::ii<DataVector, 3, Frame>& grad_normal,
    const tnsr::i<DataVector, 3, Frame>& unit_normal_one_form,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector);
/// @}

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
/// \f$ K_{ij} \f$ the output of StrahlkorperGr::extrinsic_curvature,
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

/*!
 * \ingroup SurfacesGroup
 * \brief Surface integral of a scalar on a 2D `Strahlkorper`
 *
 * \details Computes the surface integral \f$\oint dA f\f$ for a scalar \f$f\f$
 * on a `Strahlkorper` with area element \f$dA\f$. The area element can be
 * computed via `StrahlkorperGr::area_element()`.
 */
template <typename Frame>
double surface_integral_of_scalar(const Scalar<DataVector>& area_element,
                                  const Scalar<DataVector>& scalar,
                                  const Strahlkorper<Frame>& strahlkorper);

/*!
 * \ingroup SurfacesGroup
 * \brief Euclidean surface integral of a vector on a 2D `Strahlkorper`
 *
 * \details Computes the surface integral
 * \f$\oint V^i s_i (s_j s_k \delta^{jk})^{-1/2} d^2S\f$ for a
 * vector \f$V^i\f$ on a `Strahlkorper` with area element \f$d^2S\f$ and
 * normal one-form \f$s_i\f$.  Here \f$\delta^{ij}\f$ is the Euclidean
 * metric (i.e. the Kronecker delta). Note that the input `normal_one_form`
 * is not assumed to be normalized; the denominator of the integrand
 * effectively normalizes it using the Euclidean metric.
 * The area element can be computed via
 * `StrahlkorperGr::euclidean_area_element()`.
 */
template <typename Frame>
double euclidean_surface_integral_of_vector(
    const Scalar<DataVector>& area_element,
    const tnsr::I<DataVector, 3, Frame>& vector,
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const Strahlkorper<Frame>& strahlkorper);

/// @{
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
 * DataBox using the `StrahlkorperTags::Tangents` tag.
 */
template <typename Frame>
void spin_function(gsl::not_null<Scalar<DataVector>*> result,
                   const StrahlkorperTags::aliases::Jacobian<Frame>& tangents,
                   const Strahlkorper<Frame>& strahlkorper,
                   const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
                   const Scalar<DataVector>& area_element,
                   const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature);

template <typename Frame>
Scalar<DataVector> spin_function(
    const StrahlkorperTags::aliases::Jacobian<Frame>& tangents,
    const Strahlkorper<Frame>& strahlkorper,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const Scalar<DataVector>& area_element,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature);
/// @}

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
 * The argument `tangents` can be obtained from the DataBox
 * using the `StrahlkorperTags::Tangents` tag, and the argument
 * `unit_normal_vector` can
 * be found by raising the index of the one-form returned by
 * `StrahlkorperGr::unit_normal_one_form`.
 * The argument `ylm` is the `YlmSpherepack` of the `Strahlkorper`.
 * The argument `area_element`
 * can be computed via `StrahlkorperGr::area_element`.
 * The arguments `radius`, `r_hat`, `jacobian`, `inv_hessian` and
 * `cartesian_coords` can be obtained from the DataBox using the
 * tags `StrahlkorperTags::Radius`, `StrahlkorperTags::Rhat`,
 * `StrahlkorperTags::Jacobian`, `StrahlkorperTags::InvHessian` and
 * `StrahlkorperTags::CartesianCoords` respectively.
 *
 * Computing the covariant laplacian requires \f$\Gamma^C\f$ (\f$g^{AB}
 * \Gamma^C_{AB}\f$):
 * \f{align}{ \nabla^2 z = g^{AB} z_{AB} - \Gamma^C z_C, \f}
 * where \f$g^{AB}\f$ is the inverse surface metric, \f$z_C\f$ and \f$z_{AB}\f$
 * are ylm.first_and_second_derivative(f).first.get(C) and
 * ylm.first_and_second_derivative(f).second.get(A,B), respectively.
 * \f$\Gamma^C\f$ is calculated in the function
 * `get_trace_christoffel_second_kind`. Because of the non-coordinate,
 * non-orthonormal basis used by spherepack to compute derivatives the
 * calculation of \f$\Gamma^c\f$ is not straightforward. Following are the notes
 * written by Rob where he derives various quantities used in the function
 * `get_trace_christoffel_second_kind`.
 *
 * To construct the connection coefficients for the laplacian of a scalar on a
 * Strahlkorper, I need the hessian of the transformation from unit-sphere
 * coordinates \f$\{\sin\theta\cos\phi, \sin\theta\sin\phi,
 * \cos\theta\}\f$(In fact, the quantities that I actually need are the
 * derivatives of the ``star-shaped surface'' coordinates \f$\{r
 * \sin\theta\cos\phi, r \sin\theta\sin\phi, r \cos\theta\}\f$, where \f$r\f$ is
 * a nonconstant function. However the \f$r\f$-scaling can be handled with some
 * simple correction terms, summarized at the end of this note. For the moment,
 * because the relevant SpECTRE functions handle the unit-sphere case I'll defer
 * the consideration of nonconstant \f$r\f$ to the end.) to surface coordinates
 * \f$\{\theta, \phi\}\f$. Specifically, I need: \f{equation}{ H^k_{AB} := D_A
 * D_B x^k, \f} where \f$ijk...\f$ index the 3-space and \f$ABC...\f$ index the
 * 2-dimensional tangent spaces. The derivative operators \f$D_A\f$ are the
 * spherepack derivative operators \f$\{\partial_\theta, (1/\sin\theta)
 * \partial_\phi\}\f$ (which, notably, do not commute).
 *
 * These quantities could be computed simply from Spherepack's derivative
 * operators. However that
 * would involve taking three numerical second derivatives, which can be avoided
 * using the inverse hessian (`StrahlkorperTags::InvHessian`),
 * which I'll call \f$h^C_{ij}\f$. The most familiar expression for this is:
 * \f{equation}{ H^k_{AB} \stackrel{?}{=} - J^k_C J^i_A J^j_B
 * h^C_{ij},\label{e:QuestionableIdentity} \f} where \f$J^k_C\f$ is the jacobian
 * (`StrahlkorperTags::Jacobian`). Unfortunately, in this case, the
 * relationship is slightly more elaborate than that. The issue has to do with
 * the fact that the "coordinate transformation" is from a 3-dimensional space
 * to a 2-dimensional space, so the jacobian and inverse jacobian are inverse
 * matrices in one ordering but not the other:
 * \f{align}{
 *   I^A_k J^k_B &= \delta^A_B\label{e:RightInverse}\\
 *   J^k_B I^B_i &= P^k_i = \delta^k_i - n^k n_i \neq
 * \delta^k_i.\label{e:Projector} \f}
 *
 * Here, \f$I^B_i\f$ refers to the inverse
 * jacobian. Because these jacobians are not
 * square matrices, \f$I\f$ can be the left inverse of \f$J\f$ but not its right
 * inverse. The product traced on 2-d indices is instead the projector to the
 * 2-surface, where \f$n\f$ is the normal (radial unit vector, since we're
 * working with a coordinate sphere).
 *
 * The derivation that normally leads to Eq.
 * (\f$\ref{e:QuestionableIdentity}\f$) starts with a decomposition of unity.
 * We'll take the same strategy here, but include the term involving \f$n\f$ in
 * Eq. (\f$\ref{e:Projector}\f$): \f{align}{
 *   0 &= D_A \left( \delta^k_i \right)\\
 *   &= D_A \left( J^k_B I^B_i + n^k n_i \right)\\
 *   &= J^k_B D_A I^B_i + I^B_i D_A J^k_B + D_A \left(n^k n_i  \right)\\
 *   &= J^k_B J^j_A \partial_j I^B_i + I^B_i H^k_{AB} + D_A \left(n^k n_i
 * \right)\\
 *   &= J^k_B J^j_A h^B_{ji} + I^B_i H^k_{AB} + D_A \left(n^k n_i  \right)
 * \f}
 * So we can rewrite this as:
 * \f{align}{
 * I^B_i H^k_{AB} = - J^k_C J^j_A h^C_{ji} - D_A \left(n^k n_i  \right),
 * \f}
 * And then because \f$I\f$ has a true right-inverse
 * (Eq. (\f$\ref{e:RightInverse}\f$)), we conclude:
 * \f{align}{
 *   H^k_{AB} &= - J^i_B J^k_C J^j_A h^C_{ji} - J^i_B D_A \left(n^k n_i
 * \right)\\
 *   &= - J^k_C J^i_A J^j_B h^C_{ij} - J^i_B D_A \left(n^k n_i  \right).
 * \f}
 *
 * So we've recovered the standard formula apart from the correction term:
 * \f{align}{
 *   J^i_B D_A \left(n^k n_i  \right) &= J^i_B J^j_A \partial_j \left(n^k n_i
 * \right)\\
 *   &= J^i_B J^j_A \left(n^k \partial_j n_i + n_i \partial_j n^k \right).
 * \f}
 *
 * From here, we can apply the familiar result \f$\partial_j n^k = P^k_j\f$,
 * which comes from taking \f$n^k = (1/r) x^k\f$, working out the derivative,
 * and letting \f$r=1\f$ at the end since we're specializing to a unit sphere.
 * This then gives us the result: \f{align}{ H^k_{AB} &= - J^k_C J^i_A J^j_B
 * h^C_{ij} - n^k g_{AB},\label{e:Result} \f} where \f$g_{AB}\f$ is the
 * intrinsic metric on the unit sphere (an identity matrix in Spherepack's
 * basis). I have confirmed in a few test cases that this gives the same result
 * (up to truncation error) as the explicit second derivative taken with
 * Spherepack.
 *
 * As a final note, I should also clarify the extra correction terms that arise
 * from the true surface being a star-shaped surface rather than a unit-sphere.
 * The deformed surface is parametrized as: \f{align}{ x^i = r x_0^i, \f} where
 * \f$r(\theta, \phi)\f$ is the non-constant radius and now \f$x_0^i\f$ refers
 * to the unit-sphere coordinates \f$\{\sin\theta\cos\phi, \sin\theta\sin\phi,
 * \cos\theta\}\f$. Explicitly taking surface derivatives of this, we get:
 * \f{align}{
 *   D_B x^i &= r D_B x_0^i + x_0^i D_B r\\
 *   D_A D_B x^i &= r D_A D_B x_0^i + 2 J^i_{0(A} D_{B)} r + x_0^i D_A D_B r\\
 *   H^i_{AB} &= r H^i_{0AB} + 2 J^i_{0(A} D_{B)} r + x_0^i D_A D_B r.
 * \f}
 *
 * The quantity \f$H^i_{0AB}\f$ is the unit-sphere hessian computed via
 * Eq. (\f$\ref{e:Result}\f$), and \f$J^i_{0A}\f$ is the
 * Jacobian(`StrahlkorperTags::Jacobian`). So to get the hessian of the deformed
 * sphere, we simply add correction terms. The same maneuver is used for the
 * SurfaceTangents method. This does mean that the calculation once again
 * involves numerical derivatives, but only of \f$r\f$ rather than all three
 * spatial coordinates. The derivatives of \f$r\f$ are unfortunately unavoidable
 * unless the whole calculation is done in coordinates where \f$r\f$ is
 * constant, which I'd rather not assume.
 */
template <typename Frame>
double dimensionful_spin_magnitude(
    const Scalar<DataVector>& ricci_scalar,
    const Scalar<DataVector>& spin_function,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const StrahlkorperTags::aliases::Jacobian<Frame>& tangents,
    const YlmSpherepack& ylm, const Scalar<DataVector>& area_element,
    const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Frame>& r_hat,
    const StrahlkorperTags::aliases::Jacobian<Frame>& jacobian,
    const StrahlkorperTags::aliases::InvHessian<Frame>& inv_hessian,
    const StrahlkorperTags::aliases::Vector<Frame>& cartesian_coords);

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
void spin_vector(const gsl::not_null<std::array<double, 3>*> result,
                 double spin_magnitude, const Scalar<DataVector>& area_element,
                 const Scalar<DataVector>& radius,
                 const tnsr::i<DataVector, 3, Frame>& r_hat,
                 const Scalar<DataVector>& ricci_scalar,
                 const Scalar<DataVector>& spin_function,
                 const Strahlkorper<Frame>& strahlkorper);

template <typename Frame>
std::array<double, 3> spin_vector(double spin_magnitude,
                                  const Scalar<DataVector>& area_element,
                                  const Scalar<DataVector>& radius,
                                  const tnsr::i<DataVector, 3, Frame>& r_hat,
                                  const Scalar<DataVector>& ricci_scalar,
                                  const Scalar<DataVector>& spin_function,
                                  const Strahlkorper<Frame>& strahlkorper);

/*!
 * \ingroup SurfacesGroup
 * \brief Irreducible mass of a 2D `Strahlkorper`.
 *
 * \details See Eqs. (15.38) \cite Hartle2003gravity. This function computes the
 * irreducible mass from the area of a horizon. Specifically, computes
 * \f$M_\mathrm{irr}=\sqrt{\frac{A}{16\pi}}\f$.
 */
double irreducible_mass(double area);

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
                          double irreducible_mass);
}  // namespace StrahlkorperGr
