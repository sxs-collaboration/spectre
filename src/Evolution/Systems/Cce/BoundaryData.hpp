// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/BoundaryDataTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"

/// \cond
class DataVector;
class ComplexDataVector;
/// \endcond

namespace Cce {

/*!
 * \brief Constructs the collocation values for \f$\cos(\phi)\f$,
 * \f$\cos(\theta)\f$, \f$\sin(\phi)\f$, and \f$\sin(\theta)\f$, returned by
 * `not_null` pointer in that order.
 *
 * \details These are needed for coordinate transformations from the input
 * Cartesian-like coordinates.
 */
void trigonometric_functions_on_swsh_collocation(
    gsl::not_null<Scalar<DataVector>*> cos_phi,
    gsl::not_null<Scalar<DataVector>*> cos_theta,
    gsl::not_null<Scalar<DataVector>*> sin_phi,
    gsl::not_null<Scalar<DataVector>*> sin_theta, size_t l_max) noexcept;

/*!
 * \brief Creates both the Jacobian and inverse Jacobian between Cartesian and
 * spherical coordinates, and the coordinates themselves
 *
 * \details The `cartesian_to_spherical_jacobian` is
 * \f$dx^i/d\tilde{x}^{\tilde j}\f$,
 * where the Cartesian components are in order \f$x^i = \{x, y, z\}\f$
 * and the spherical coordinates are
 * \f$\tilde{x}^{\tilde j} = \{r, \theta, \phi\}\f$.
 * The Cartesian coordinates given are the standard unit sphere coordinates:
 *
 * \f{align*}{
 *  x &= \cos(\phi) \sin(\theta)\\
 *  y &= \sin(\phi) \sin(\theta)\\
 *  z &= \cos(\theta)
 * \f}
 *
 * \note These Jacobians are adjusted to improve regularity near the pole, in
 * particular the \f$\partial \phi / \partial x^i\f$ components have been scaled
 * by \f$\sin \theta\f$ (omitting a \f$1/\sin(\theta)\f$) and the
 * \f$\partial x^i/\partial \phi\f$ components have been scaled by
 * \f$1/\sin(\theta)\f$ (omitting a \f$\sin(\theta)\f$). The reason is that in
 * most careful calculations, these problematic sin factors can actually be
 * omitted because they cancel. In cases where they are actually required, they
 * must be put in by hand.
 */
void cartesian_to_spherical_coordinates_and_jacobians(
    gsl::not_null<tnsr::I<DataVector, 3>*> unit_cartesian_coords,
    gsl::not_null<SphericaliCartesianJ*> cartesian_to_spherical_jacobian,
    gsl::not_null<CartesianiSphericalJ*>
        inverse_cartesian_to_spherical_jacobian,
    const Scalar<DataVector>& cos_phi, const Scalar<DataVector>& cos_theta,
    const Scalar<DataVector>& sin_phi, const Scalar<DataVector>& sin_theta,
    double extraction_radius) noexcept;

/*
 * \brief Compute \f$g_{i j}\f$, \f$g^{i j}\f$, \f$\partial_i g_{j k}\f$, and
 * \f$\partial_t g_{i j}\f$ from input libsharp-compatible modal spatial
 * metric quantities.
 *
 * \details This function interpolates the modes of
 * input \f$g_{ij}\f$, \f$\partial_r g_{i j}\f$, and \f$\partial_r g_{i j}\f$ to
 * the libsharp-compatible grid. This function then applies the necessary
 * jacobian factors and angular derivatives to determine the full \f$\partial_i
 * g_{j k}\f$.
 */
void cartesian_spatial_metric_and_derivatives_from_modes(
    gsl::not_null<tnsr::ii<DataVector, 3>*> cartesian_spatial_metric,
    gsl::not_null<tnsr::II<DataVector, 3>*> inverse_cartesian_spatial_metric,
    gsl::not_null<tnsr::ijj<DataVector, 3>*> d_cartesian_spatial_metric,
    gsl::not_null<tnsr::ii<DataVector, 3>*> dt_cartesian_spatial_metric,
    gsl::not_null<Scalar<SpinWeighted<ComplexModalVector, 0>>*>
        interpolation_modal_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        interpolation_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> eth_buffer,
    const tnsr::ii<ComplexModalVector, 3>& spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dr_spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dt_spatial_metric_coefficients,
    const CartesianiSphericalJ& inverse_cartesian_to_spherical_jacobian,
    size_t l_max) noexcept;

/*!
 * \brief Compute \f$\beta^{i}\f$, \f$\partial_i \beta^{j}\f$, and
 * \f$\partial_t \beta^i\f$ from input libsharp-compatible modal spatial
 * metric quantities.
 *
 * \details This function interpolates the modes of
 * input \f$\beta^i\f$, \f$\partial_r \beta^i\f$, and \f$\partial_r \beta^i\f$
 * to the libsharp-compatible grid. This function then applies the necessary
 * jacobian factors and angular derivatives to determine the full \f$\partial_i
 * \beta^i\f$.
 */
void cartesian_shift_and_derivatives_from_modes(
    gsl::not_null<tnsr::I<DataVector, 3>*> cartesian_shift,
    gsl::not_null<tnsr::iJ<DataVector, 3>*> d_cartesian_shift,
    gsl::not_null<tnsr::I<DataVector, 3>*> dt_cartesian_shift,
    gsl::not_null<Scalar<SpinWeighted<ComplexModalVector, 0>>*>
        interpolation_modal_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        interpolation_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> eth_buffer,
    const tnsr::I<ComplexModalVector, 3>& shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dr_shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dt_shift_coefficients,
    const CartesianiSphericalJ& inverse_cartesian_to_spherical_jacobian,
    size_t l_max) noexcept;

/*!
 * \brief Compute \f$\alpha\f$, \f$\partial_i \alpha\f$, and
 * \f$\partial_t \beta^i\f$ from input libsharp-compatible modal spatial
 * metric quantities.
 *
 * \details This function interpolates the modes of input \f$\alpha\f$,
 * \f$\partial_r \alpha\f$, and \f$\partial_r \alpha\f$ to the
 * libsharp-compatible grid. This function then applies the necessary jacobian
 * factors and angular derivatives to determine the full \f$\partial_i
 * \alpha\f$.
 */
void cartesian_lapse_and_derivatives_from_modes(
    gsl::not_null<Scalar<DataVector>*> cartesian_lapse,
    gsl::not_null<tnsr::i<DataVector, 3>*> d_cartesian_lapse,
    gsl::not_null<Scalar<DataVector>*> dt_cartesian_lapse,
    gsl::not_null<Scalar<SpinWeighted<ComplexModalVector, 0>>*>
        interpolation_modal_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        interpolation_buffer,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> eth_buffer,
    const Scalar<ComplexModalVector>& lapse_coefficients,
    const Scalar<ComplexModalVector>& dr_lapse_coefficients,
    const Scalar<ComplexModalVector>& dt_lapse_coefficients,
    const CartesianiSphericalJ& inverse_cartesian_to_spherical_jacobian,
    size_t l_max) noexcept;

/*!
 * \brief Computes the spacetime metric and its first derivative in the
 * intermediate radial null coordinates
 *
 * \details These components are obtained by the steps in
 * Section II-A of \cite Barkett2019uae, which is based on the computation from
 * Section 4.3 of \cite Bishop1998uk. The most direct comparison is to be made
 * with equation (31) of \cite Barkett2019uae, which gives the null metric
 * components explicitly. The time derivative is then (using notation from
 * equation (31)  of \cite Barkett2019uae):
 *
 * \f{align}{
 * \partial_{\bar u} g_{\bar u \bar \lambda} =
 * \partial_{\bar u} g_{\bar \lambda \bar \lambda} =
 * \partial_{\bar u} g_{\bar \lambda \bar A} &= 0 \\
 * \partial_{\bar u} g_{\bar u \bar u} &=
 * \partial_{\breve t} g_{\breve t \breve t} \\
 * \partial_{\bar u} g_{\bar u \bar A} &=
 * \frac{\partial \breve x^{\breve i}}{\partial \bar x^{\bar A}}\\
 * g_{\breve i \breve t}
 * \partial_{\bar u} g_{\bar A \bar B}
 * &= \frac{\partial \breve x^{\breve i}}{\partial \bar x^{\bar A}}
 * \frac{\partial \breve x^{\breve j}}{\partial \bar x^{\bar B}}
 * g_{\breve i \breve j}
 * \f}
 */
void null_metric_and_derivative(
    gsl::not_null<tnsr::aa<DataVector, 3, Frame::RadialNull>*> du_null_metric,
    gsl::not_null<tnsr::aa<DataVector, 3, Frame::RadialNull>*> null_metric,
    const SphericaliCartesianJ& cartesian_to_spherical_jacobian,
    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const tnsr::aa<DataVector, 3>& spacetime_metric) noexcept;

/*!
 * \brief Computes the spatial unit normal vector \f$s^i\f$ to the spherical
 * worldtube surface and its first time derivative.
 *
 * \details Refer to equation (20) of \cite Barkett2019uae for the expression of
 * the spatial unit normal vector, and equation (23) of \cite Barkett2019uae for
 * the first time derivative. Refer to \cite Bishop1998uk for more exposition
 * about the overall construction of the coordinate transformations used for the
 * intermediate null coordinates.
 */
void worldtube_normal_and_derivatives(
    gsl::not_null<tnsr::I<DataVector, 3>*> worldtube_normal,
    gsl::not_null<tnsr::I<DataVector, 3>*> dt_worldtube_normal,
    const Scalar<DataVector>& cos_phi, const Scalar<DataVector>& cos_theta,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const Scalar<DataVector>& sin_phi, const Scalar<DataVector>& sin_theta,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric) noexcept;

/*!
 * \brief Computes the null 4-vector \f$l^\mu\f$ on the worldtube surface that
 * is to be used as the CCE hypersurface generator, and the first time
 * derivative \f$\partial_u l^\mu\f$.
 *
 * \details For mathematical description of our choice of the null generator,
 * refer to equation (22) of \cite Barkett2019uae, and for the first time
 * derivative see (25) of \cite Barkett2019uae.  Refer to \cite Bishop1998uk for
 * more exposition about the overall construction of the coordinate
 * transformations used for the intermediate null coordinates.
 */
void null_vector_l_and_derivatives(
    gsl::not_null<tnsr::A<DataVector, 3>*> du_null_l,
    gsl::not_null<tnsr::A<DataVector, 3>*> null_l,
    const tnsr::I<DataVector, 3>& dt_worldtube_normal,
    const Scalar<DataVector>& dt_lapse,
    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const tnsr::I<DataVector, 3>& dt_shift, const Scalar<DataVector>& lapse,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& worldtube_normal) noexcept;

/*!
 * \brief Computes the partial derivative of the spacetime metric and inverse
 * spacetime metric in the intermediate null radial coordinates with respect to
 * the null generator \f$l^\mu\f$
 *
 * \details For full expressions of the \f$l^\mu \partial_\mu g_{a b}\f$ and
 * \f$l^\mu \partial_\mu g^{a b}\f$ computed in this function, see equation (31)
 * and (32) of \cite Barkett2019uae.  Refer to \cite Bishop1998uk for more
 * exposition about the overall construction of the coordinate transformations
 * used for the intermediate null coordinates.
 */
void dlambda_null_metric_and_inverse(
    gsl::not_null<tnsr::aa<DataVector, 3, Frame::RadialNull>*>
        dlambda_null_metric,
    gsl::not_null<tnsr::AA<DataVector, 3, Frame::RadialNull>*>
        dlambda_inverse_null_metric,
    const AngulariCartesianA& angular_d_null_l,
    const SphericaliCartesianJ& cartesian_to_spherical_jacobian,
    const tnsr::iaa<DataVector, 3>& phi,
    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const tnsr::A<DataVector, 3>& du_null_l,
    const tnsr::AA<DataVector, 3, Frame::RadialNull>& inverse_null_metric,
    const tnsr::A<DataVector, 3>& null_l,
    const tnsr::aa<DataVector, 3>& spacetime_metric) noexcept;

/*!
 * \brief Computes the Bondi radius of the worldtube.
 *
 * \details Note that unlike the Cauchy coordinate radius, the Bondi radius is
 * not constant over the worldtube. Instead, it is obtained by the determinant
 * of the angular part of the metric in the intermediate null coordinates (see
 * \cite Barkett2019uae).
 *
 * \f[
 *  r = \left(\frac{\det g_{A B}}{ q_{A B}}\right)^{1/4},
 * \f]
 *
 * where \f$q_{A B}\f$ is the unit sphere metric.
 */
void bondi_r(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> bondi_r,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& null_metric) noexcept;

/*!
 * \brief Computes the full 4-dimensional partial of the Bondi radius with
 * respect to the intermediate null coordinates.
 *
 * \details The expression evaluated is obtained from differentiating the
 * determinant equation for `bondi_r`, from (35) of \cite Barkett2019uae :
 *
 * \f[
 * \partial_\alpha r = \frac{r}{4} \left(g^{A B} \partial_\alpha g_{A B}
 * - \frac{\partial_\alpha \det q_{A B}}{\det q_{A B}}\right)
 * \f]
 *
 * Note that for the angular derivatives, we just numerically differentiate
 * using the utilities in `Spectral::Swsh::angular_derivative()`. For the time
 * and radial derivatives, the second term in the above equation vanishes.
 */
void d_bondi_r(
    gsl::not_null<tnsr::a<DataVector, 3, Frame::RadialNull>*> d_bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& dlambda_null_metric,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& du_null_metric,
    const tnsr::AA<DataVector, 3, Frame::RadialNull>& inverse_null_metric,
    size_t l_max) noexcept;

/*!
 * \brief Compute the complex angular dyads used to define the spin-weighted
 * scalars in the CCE system.
 *
 * \details We use the typically chosen angular dyads in CCE
 * \cite Barkett2019uae \cite Bishop1997ik :
 *
 * \f{align*}{
 * q_A &= \{-1, -i \sin(\theta)\}\\
 * q^A &= \left\{-1, -i \frac{1}{\sin \theta}\right\}
 * \f}
 *
 * However, to maintain regularity and for compatibility with the more regular
 * Jacobians from `Cce::cartesian_to_spherical_coordinates_and_jacobians()`, in
 * the code we omit the factors of \f$\sin \theta\f$ from the above equations.
 */
void dyads(
    gsl::not_null<tnsr::i<ComplexDataVector, 2, Frame::RadialNull>*> down_dyad,
    gsl::not_null<tnsr::I<ComplexDataVector, 2, Frame::RadialNull>*>
        up_dyad) noexcept;

/*!
 * \brief Compute the \f$\beta\f$ (lapse) function for the CCE Bondi-like
 * metric.
 *
 * \details The Bondi-like metric has \f$g^{u r} = - e^{2 \beta}\f$, and the
 * value of \f$\beta\f$ is obtained from the intermediate null metric by (see
 * equation (51) of \cite Barkett2019uae) using:
 *
 * \f[
 * \beta = -\frac{1}{2} \ln \partial_{\lambda} r
 * \f]
 */
void beta_worldtube_data(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> beta,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r) noexcept;

/*!
 * \brief Compute the \f$U\f$ (shift) function for the CCE Bondi-like metric.
 *
 * \details The Bondi-like metric has \f$g^{r A} = -e^{-2 \beta} U^A\f$, and the
 * spin-weighted vector \f$U = U^A q_A\f$. The value of \f$U^A\f$ can be
 * computed from the intermediate null metric quantities (see equation (54) of
 * \cite Barkett2019uae) using:
 *
 * \f[
 * U = -(\partial_\lambda r g^{\lambda A} + \partial_B r g^{A B}) q_A
 * / \partial_\lambda r \f]
 *
 */
void bondi_u_worldtube_data(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> bondi_u,
    const tnsr::i<ComplexDataVector, 2, Frame::RadialNull>& dyad,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r,
    const tnsr::AA<DataVector, 3, Frame::RadialNull>&
        inverse_null_metric) noexcept;

/*!
 * \brief Compute the \f$W\f$ (mass aspect) function for the CCE Bondi-like
 * metric.
 *
 * \details The Bondi-like metric has \f$g^{rr} = e^{-2 \beta}(1 + r W)\f$. The
 * value of \f$W\f$ can be computed from the null metric quantities (see
 * equation (55) of \cite Barkett2019uae) using:
 *
 * \f[
 * W = \frac{1}{r} \left(-1
 * + \frac{g^{\lambda \lambda} (\partial_\lambda r)^2
 * + 2 \partial_\lambda r \left(\partial_A r g^{\lambda A}
 * - \partial_u r\right) + \partial_A r \partial_B r g^{A B}}
 * {\partial_\lambda r}\right) \f]
 */
void bondi_w_worldtube_data(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> bondi_w,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r,
    const tnsr::AA<DataVector, 3, Frame::RadialNull>& inverse_null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r) noexcept;

/*!
 * \brief Compute the \f$J\f$ (intuitively similar to the transverse-traceless
 * part of the angular metric) function for the CCE Bondi-like metric.
 *
 * \details The Bondi-like metric has \f$J = \frac{1}{2 r^2} q^A q^B g_{A B}\f$.
 * This expression holds both for the right-hand side in the Bondi coordinates
 * and for the right-hand side in the intermediate null coordinates (see
 * equation (45) of \cite Barkett2019uae).
 */
void bondi_j_worldtube_data(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> bondi_j,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad) noexcept;

/*!
 * \brief Compute the radial derivative of the angular metric spin-weighted
 * scalar \f$\partial_r J\f$ in the CCE Bondi-like metric.
 *
 * \details The radial derivative of the angular spin-weighted scalar \f$J\f$
 * can be computed from the null metric components by (c.f. equation (47) of
 * \cite Barkett2019uae):
 *
 * \f[
 * \partial_r J = \frac{\partial_\lambda J}{\partial_\lambda r} =
 *  \frac{q^A q^B \partial_\lambda g_{A B} / (2 r^2)
 * - 2 \partial_\lambda r J / r}{\partial_\lambda r}
 * \f]
 */
void dr_bondi_j(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> dr_bondi_j,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        denominator_buffer,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& dlambda_null_metric,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad) noexcept;

/*!
 * \brief Compute the second derivative of the Bondi radius with respect to the
 * intermediate null coordinate radius \f$\partial_\lambda^2 r\f$.
 *
 * \details To determine this second derivative quantity without resorting to
 * depending on second-derivative metric inputs, we need to take advantage of
 * one of the Einstein field equations. Combining equations (53) and (52) of
 * \cite Barkett2019uae, we have:
 *
 * \f[
 * \partial_\lambda^2 r = \frac{-r}{4} \left(
 * \partial_\lambda J \partial_\lambda \bar J - (\partial_\lambda K)^2\right)
 * \f],
 *
 * where the first derivative of \f$K\f$ can be obtained from \f$K = \sqrt{1 + J
 * \bar J}\f$ and the first derivative of \f$J\f$ can be obtained from (47) of
 * \cite Barkett2019uae
 */
void d2lambda_bondi_r(
    gsl::not_null<Scalar<DataVector>*> d2lambda_bondi_r,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dr_bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r) noexcept;

/*!
 * \brief Compute the Bondi metric contribution \f$Q\f$ (radial derivative of
 * shift).
 *
 * \details The definition of \f$Q\f$ in terms of the Bondi metric components is
 *
 * \f[
 *  Q = q^A e^{-2 \beta} g_{A B} \partial_r U^B.
 * \f]
 *
 * $Q$ can be derived from the intermediate null metric quantities via (see
 * equations (56) and (57) of \cite Barkett2019uae)
 *
 * \f[
 * \partial_\lambda U = - \left(\partial_\lambda g^{\lambda A}
 * + \frac{\partial_A \partial_\lambda r}{\partial_\lambda r} g^{A B}
 * + \frac{\partial_B r}{\partial_\lambda r} \partial_\lambda g^{A B}\right) q_A
 * + 2 \partial_\lambda \beta (U + g^{\lambda A} q_A)
 * \f]
 *
 * and
 *
 * \f[
 * Q = r^2 (J \partial_\lambda \bar U + K \partial_\lambda U)
 * \f]
 *
 * also provided is \f$\partial_r U\f$, which is separately useful to cache for
 * other intermediate steps in the CCE computation.
 */
void bondi_q_worldtube_data(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> bondi_q,
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> dr_bondi_u,
    const Scalar<DataVector>& d2lambda_r,
    const tnsr::AA<DataVector, 3, Frame::RadialNull>&
        dlambda_inverse_null_metric,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r,
    const tnsr::i<ComplexDataVector, 2, Frame::RadialNull>& dyad,
    const tnsr::i<DataVector, 2, Frame::RadialNull>& angular_d_dlambda_r,
    const tnsr::AA<DataVector, 3, Frame::RadialNull>& inverse_null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_u) noexcept;

/*!
 * \brief Compute the Bondi metric contribution \f$(\partial_u J)_{y} \equiv
 * H\f$ (the retarded time derivative evaluated at fixed $y$ coordinate) on the
 * worldtube boundary.
 *
 * \details The numerical time derivative (along the worldtube, rather than
 * along the surface of constant Bondi \f$r\f$) is computed by (see equation
 * (48) of \cite Barkett2019uae)
 *
 * \f[
 * (\partial_u J)_y = \frac{1}{2 r^2} q^A q^B \partial_u g_{A B}
 * - \frac{2 \partial_u r}{r} J
 * \f]
 *
 * \note There is the regrettable notation difference with the primary reference
 * for these formulas \cite Barkett2019uae in that we denote with \f$H\f$ the
 * time derivative at constant numerical radius, where \cite Barkett2019uae uses
 * \f$H\f$ to denote the time derivative at constant Bondi radius.
 */
void bondi_h_worldtube_data(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> bondi_h,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& du_null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad) noexcept;

/*!
 * \brief Compute the Bondi metric contribution \f$(\partial_u J)_r\f$ (the
 * retarded time derivative at fixed coordinate $r$) on the worldtube boundary.
 *
 * \details The numerical time derivative (along the surface of constant r, not
 * along the worldtube) is computed by (see equation (50) of
 * \cite Barkett2019uae)
 *
 * \f[
 * \partial_u J = \frac{1}{2 r^2} q^A q^B \left(\partial_u g_{A B}
 * - \frac{ \partial_u r}{ \partial_\lambda r} \partial_\lambda g_{A B}\right)
 * \f]
 *
 * \note There is the regrettable notation difference with the primary reference
 * for these formulas \cite Barkett2019uae in that we denote with \f$H\f$ the
 * time derivative at constant numerical radius, where \cite Barkett2019uae uses
 * \f$H\f$ to denote the time derivative at constant Bondi radius.
 */
void du_j_worldtube_data(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> du_bondi_j,
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& bondi_j,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& du_null_metric,
    const tnsr::aa<DataVector, 3, Frame::RadialNull>& dlambda_null_metric,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r,
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad) noexcept;
}  // namespace Cce
