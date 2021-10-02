// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/BoundaryDataTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpacetimeMetric.hpp"

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
    gsl::not_null<Scalar<DataVector>*> sin_theta, size_t l_max);

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
    double extraction_radius);

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
    size_t l_max);

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
    size_t l_max);

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
    size_t l_max);

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
    const tnsr::aa<DataVector, 3>& spacetime_metric);

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
    const tnsr::II<DataVector, 3>& inverse_spatial_metric);

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
    const tnsr::I<DataVector, 3>& worldtube_normal);

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
    const tnsr::aa<DataVector, 3>& spacetime_metric);

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
void bondi_r(gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> bondi_r,
             const tnsr::aa<DataVector, 3, Frame::RadialNull>& null_metric);

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
    size_t l_max);

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
    gsl::not_null<tnsr::I<ComplexDataVector, 2, Frame::RadialNull>*> up_dyad);

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
    const tnsr::a<DataVector, 3, Frame::RadialNull>& d_bondi_r);

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
    const tnsr::AA<DataVector, 3, Frame::RadialNull>& inverse_null_metric);

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
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r);

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
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad);

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
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad);

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
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& bondi_r);

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
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& bondi_u);

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
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad);

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
    const tnsr::I<ComplexDataVector, 2, Frame::RadialNull>& dyad);

namespace Tags {
/// The collection of tags mutated by `create_bondi_boundary_data`
template <template <typename> class BoundaryPrefix>
using characteristic_worldtube_boundary_tags = db::wrap_tags_in<
    BoundaryPrefix,
    tmpl::list<Tags::BondiBeta, Tags::BondiU, Tags::Dr<Tags::BondiU>,
               Tags::BondiQ, Tags::BondiW, Tags::BondiJ, Tags::Dr<Tags::BondiJ>,
               Tags::BondiH, Tags::Du<Tags::BondiJ>, Tags::BondiR,
               Tags::Du<Tags::BondiR>, Tags::DuRDividedByR>>;
}  // namespace Tags

namespace detail {
// the common step between the modal input and the Generalized harmonic input
// that performs the final gauge processing to Bondi scalars and places them in
// the Variables.
template <typename BoundaryTagList, typename BufferTagList,
          typename ComplexBufferTagList>
void create_bondi_boundary_data(
    const gsl::not_null<Variables<BoundaryTagList>*> bondi_boundary_data,
    const gsl::not_null<Variables<BufferTagList>*> computation_variables,
    const gsl::not_null<Variables<ComplexBufferTagList>*> derivative_buffers,
    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const tnsr::iaa<DataVector, 3>& phi,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::A<DataVector, 3>& null_l,
    const tnsr::A<DataVector, 3>& du_null_l,
    const SphericaliCartesianJ& cartesian_to_spherical_jacobian,
    const size_t l_max, const double extraction_radius) {
  const size_t size = Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // unfortunately, because the dyads are not themselves spin-weighted, they
  // need a separate Variables
  Variables<tmpl::list<Tags::detail::DownDyad, Tags::detail::UpDyad>>
      dyad_variables{size};

  auto& null_metric =
      get<gr::Tags::SpacetimeMetric<3, Frame::RadialNull, DataVector>>(
          *computation_variables);
  auto& du_null_metric = get<
      ::Tags::dt<gr::Tags::SpacetimeMetric<3, Frame::RadialNull, DataVector>>>(
      *computation_variables);
  null_metric_and_derivative(
      make_not_null(&du_null_metric), make_not_null(&null_metric),
      cartesian_to_spherical_jacobian, dt_spacetime_metric, spacetime_metric);

  auto& inverse_null_metric =
      get<gr::Tags::InverseSpacetimeMetric<3, Frame::RadialNull, DataVector>>(
          *computation_variables);

  // the below scaling process is used to reduce accumulation of numerical
  // error in the determinant evaluation

  // buffer reuse because the scaled null metric is only needed until the
  // `determinant_and_inverse` call
  auto& scaled_null_metric =
      get<gr::Tags::InverseSpacetimeMetric<3, Frame::RadialNull, DataVector>>(
          *computation_variables);
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = i; j < 4; ++j) {
      if (i > 1 and j > 1) {
        scaled_null_metric.get(i, j) =
            null_metric.get(i, j) / square(extraction_radius);
      } else if (i > 1 or j > 1) {
        scaled_null_metric.get(i, j) =
            null_metric.get(i, j) / extraction_radius;
      } else {
        scaled_null_metric.get(i, j) = null_metric.get(i, j);
      }
    }
  }
  // Allocation
  const auto scaled_inverse_null_metric =
      determinant_and_inverse(scaled_null_metric).second;
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = i; j < 4; ++j) {
      if (i > 1 and j > 1) {
        inverse_null_metric.get(i, j) =
            scaled_inverse_null_metric.get(i, j) / square(extraction_radius);
      } else if (i > 1 or j > 1) {
        inverse_null_metric.get(i, j) =
            scaled_inverse_null_metric.get(i, j) / extraction_radius;
      } else {
        inverse_null_metric.get(i, j) = scaled_inverse_null_metric.get(i, j);
      }
    }
  }

  auto& angular_d_null_l =
      get<Tags::detail::AngularDNullL>(*computation_variables);
  auto& buffer_for_derivatives =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                   std::integral_constant<int, 0>>>(
          *derivative_buffers));
  auto& eth_buffer =
      get(get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                   std::integral_constant<int, 1>>>(
          *derivative_buffers));
  for (size_t a = 0; a < 4; ++a) {
    buffer_for_derivatives.data() =
        std::complex<double>(1.0, 0.0) * null_l.get(a);
    Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
        l_max, 1, make_not_null(&eth_buffer), buffer_for_derivatives);
    angular_d_null_l.get(0, a) = -real(eth_buffer.data());
    angular_d_null_l.get(1, a) = -imag(eth_buffer.data());
  }

  auto& dlambda_null_metric = get<Tags::detail::DLambda<
      gr::Tags::SpacetimeMetric<3, Frame::RadialNull, DataVector>>>(
      *computation_variables);
  auto& dlambda_inverse_null_metric = get<Tags::detail::DLambda<
      gr::Tags::InverseSpacetimeMetric<3, Frame::RadialNull, DataVector>>>(
      *computation_variables);
  dlambda_null_metric_and_inverse(
      make_not_null(&dlambda_null_metric),
      make_not_null(&dlambda_inverse_null_metric), angular_d_null_l,
      cartesian_to_spherical_jacobian, phi, dt_spacetime_metric, du_null_l,
      inverse_null_metric, null_l, spacetime_metric);

  auto& r = get<Tags::BoundaryValue<Tags::BondiR>>(*bondi_boundary_data);
  bondi_r(make_not_null(&r), null_metric);

  auto& d_r =
      get<::Tags::spacetime_deriv<Tags::detail::RealBondiR, tmpl::size_t<3>,
                                  Frame::RadialNull>>(*computation_variables);
  d_bondi_r(make_not_null(&d_r), r, dlambda_null_metric, du_null_metric,
            inverse_null_metric, l_max);
  get(get<Tags::BoundaryValue<Tags::DuRDividedByR>>(*bondi_boundary_data))
      .data() =
      std::complex<double>(1.0, 0.0) * get<0>(d_r) / get(r).data();
  get(get<Tags::BoundaryValue<Tags::Du<Tags::BondiR>>>(*bondi_boundary_data))
      .data() = std::complex<double>(1.0, 0.0) * get<0>(d_r);

  auto& down_dyad = get<Tags::detail::DownDyad>(dyad_variables);
  auto& up_dyad = get<Tags::detail::UpDyad>(dyad_variables);
  dyads(make_not_null(&down_dyad), make_not_null(&up_dyad));

  beta_worldtube_data(make_not_null(&get<Tags::BoundaryValue<Tags::BondiBeta>>(
                          *bondi_boundary_data)),
                      d_r);

  auto& bondi_u = get<Tags::BoundaryValue<Tags::BondiU>>(*bondi_boundary_data);
  bondi_u_worldtube_data(make_not_null(&bondi_u), down_dyad, d_r,
                         inverse_null_metric);

  bondi_w_worldtube_data(make_not_null(&get<Tags::BoundaryValue<Tags::BondiW>>(
                             *bondi_boundary_data)),
                         d_r, inverse_null_metric, r);

  auto& bondi_j =
      get<Tags::BoundaryValue<Tags::BondiJ>>(*bondi_boundary_data);
  bondi_j_worldtube_data(make_not_null(&bondi_j), null_metric, r, up_dyad);

  auto& dr_j =
      get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(*bondi_boundary_data);
  auto& denominator_buffer =
      get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          *derivative_buffers);
  dr_bondi_j(make_not_null(&get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(
                 *bondi_boundary_data)),
             make_not_null(&denominator_buffer), dlambda_null_metric, d_r,
             bondi_j, r, up_dyad);

  auto& d2lambda_r = get<
      Tags::detail::DLambda<Tags::detail::DLambda<Tags::detail::RealBondiR>>>(
      *computation_variables);
  d2lambda_bondi_r(make_not_null(&d2lambda_r), d_r, dr_j, bondi_j, r);

  auto& angular_d_dlambda_r =
      get<::Tags::deriv<Tags::detail::DLambda<Tags::detail::RealBondiR>,
                        tmpl::size_t<2>, Frame::RadialNull>>(
          *computation_variables);
  buffer_for_derivatives.data() = std::complex<double>(1.0, 0.0) * get<1>(d_r);
  Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
      l_max, 1, make_not_null(&eth_buffer), buffer_for_derivatives);
  angular_d_dlambda_r.get(0) = -real(eth_buffer.data());
  angular_d_dlambda_r.get(1) = -imag(eth_buffer.data());

  bondi_q_worldtube_data(
      make_not_null(
          &get<Tags::BoundaryValue<Tags::BondiQ>>(*bondi_boundary_data)),
      make_not_null(&get<Tags::BoundaryValue<Tags::Dr<Tags::BondiU>>>(
          *bondi_boundary_data)),
      d2lambda_r, dlambda_inverse_null_metric, d_r, down_dyad,
      angular_d_dlambda_r, inverse_null_metric, bondi_j, r, bondi_u);

  bondi_h_worldtube_data(make_not_null(&get<Tags::BoundaryValue<Tags::BondiH>>(
                             *bondi_boundary_data)),
                         d_r, bondi_j, du_null_metric, r, up_dyad);

  du_j_worldtube_data(
      make_not_null(&get<Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>>(
          *bondi_boundary_data)),
      d_r, bondi_j, du_null_metric, dlambda_null_metric, r, up_dyad);
}
}  // namespace detail

/*!
 * \brief Process the worldtube data from generalized harmonic quantities
 *  to desired Bondi quantities, placing the result in the passed
 * `Variables`.
 *
 * \details
 * The mathematics are a bit complicated for all of the coordinate
 * transformations that are necessary to obtain the Bondi gauge quantities.
 * For full mathematical details, see the documentation for functions in
 * `BoundaryData.hpp` and \cite Barkett2019uae \cite Bishop1998uk.
 *
 * This function takes as input the full set of Generalized harmonic metric data
 * on a two-dimensional surface of constant \f$r\f$ and \f$t\f$ in numerical
 * coordinates.
 *
 * Sufficient tags to provide full worldtube boundary data at a particular
 * time are set in `bondi_boundary_data`. In particular, the set of tags in
 * `Tags::characteristic_worldtube_boundary_tags` in the provided `Variables`
 * are assigned to the worldtube boundary values associated with the input
 * metric components.
 *
 * The majority of the mathematical transformations are implemented as a set of
 * individual cascaded functions below. The details of the manipulations that
 * are performed to the input data may be found in the individual functions
 * themselves, which are called in the following order:
 * - `trigonometric_functions_on_swsh_collocation()`
 * - `gr::shift()`
 * - `gr::lapse()`
 * - `worldtube_normal_and_derivatives()`
 * - `gr::spacetime_normal_vector()`
 * - `GeneralizedHarmonic::time_deriv_of_lapse()`
 * - `GeneralizedHarmonic::time_deriv_of_shift()`
 * - `null_vector_l_and_derivatives()`
 * - `cartesian_to_spherical_coordinates_and_jacobians()`
 * - `null_metric_and_derivative()`
 * - `dlambda_null_metric_and_inverse()`
 * - `bondi_r()`
 * - `d_bondi_r()`
 * - `dyads()`
 * - `beta_worldtube_data()`
 * - `bondi_u_worldtube_data()`
 * - `bondi_w_worldtube_data()`
 * - `bondi_j_worldtube_data()`
 * - `dr_bondi_j()`
 * - `d2lambda_bondi_r()`
 * - `bondi_q_worldtube_data()`
 * - `bondi_h_worldtube_data()`
 * - `du_j_worldtube_data()`
 */
template <typename BoundaryTagList>
void create_bondi_boundary_data(
    const gsl::not_null<Variables<BoundaryTagList>*> bondi_boundary_data,
    const tnsr::iaa<DataVector, 3>& phi, const tnsr::aa<DataVector, 3>& pi,
    const tnsr::aa<DataVector, 3>& spacetime_metric,
    const double extraction_radius, const size_t l_max) {
  const size_t size = Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  // Most allocations required for the full boundary computation are merged into
  // a single, large Variables allocation. There remain a handful of cases in
  // the computational functions called where an intermediate quantity that is
  // not re-used is allocated rather than taking a buffer. These cases are
  // marked with code comments 'Allocation'; In the future, if allocations are
  // identified as a point to optimize, those buffers may be allocated here and
  // passed as function arguments
  Variables<tmpl::list<
      Tags::detail::CosPhi, Tags::detail::CosTheta, Tags::detail::SinPhi,
      Tags::detail::SinTheta, Tags::detail::CartesianCoordinates,
      Tags::detail::CartesianToSphericalJacobian,
      Tags::detail::InverseCartesianToSphericalJacobian,
      gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<3, ::Frame::Inertial, DataVector>,
      gr::Tags::Shift<3, ::Frame::Inertial, DataVector>,
      ::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>,
      gr::Tags::Lapse<DataVector>, ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>,
      Tags::detail::WorldtubeNormal, ::Tags::dt<Tags::detail::WorldtubeNormal>,
      gr::Tags::SpacetimeNormalVector<3, ::Frame::Inertial, DataVector>,
      Tags::detail::NullL, ::Tags::dt<Tags::detail::NullL>,
      // for the detail function called at the end
      gr::Tags::SpacetimeMetric<3, Frame::RadialNull, DataVector>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<3, Frame::RadialNull, DataVector>>,
      gr::Tags::InverseSpacetimeMetric<3, Frame::RadialNull, DataVector>,
      Tags::detail::AngularDNullL,
      Tags::detail::DLambda<
          gr::Tags::SpacetimeMetric<3, Frame::RadialNull, DataVector>>,
      Tags::detail::DLambda<
          gr::Tags::InverseSpacetimeMetric<3, Frame::RadialNull, DataVector>>,
      ::Tags::spacetime_deriv<Tags::detail::RealBondiR, tmpl::size_t<3>,
                              Frame::RadialNull>,
      Tags::detail::DLambda<Tags::detail::DLambda<Tags::detail::RealBondiR>>,
      ::Tags::deriv<Tags::detail::DLambda<Tags::detail::RealBondiR>,
                    tmpl::size_t<2>, Frame::RadialNull>>>
      computation_variables{size};

  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 1>>>>
      derivative_buffers{size};

  auto& cos_phi = get<Tags::detail::CosPhi>(computation_variables);
  auto& cos_theta = get<Tags::detail::CosTheta>(computation_variables);
  auto& sin_phi = get<Tags::detail::SinPhi>(computation_variables);
  auto& sin_theta = get<Tags::detail::SinTheta>(computation_variables);
  trigonometric_functions_on_swsh_collocation(
      make_not_null(&cos_phi), make_not_null(&cos_theta),
      make_not_null(&sin_phi), make_not_null(&sin_theta), l_max);

  // NOTE: to handle the singular values of polar coordinates, the phi
  // components of all tensors are scaled according to their sin(theta)
  // prefactors.
  // so, any down-index component get<2>(A) represents 1/sin(theta) A_\phi,
  // and any up-index component get<2>(A) represents sin(theta) A^\phi.
  // This holds for Jacobians, and so direct application of the Jacobians
  // brings the factors through.
  auto& cartesian_coords =
      get<Tags::detail::CartesianCoordinates>(computation_variables);
  auto& cartesian_to_spherical_jacobian =
      get<Tags::detail::CartesianToSphericalJacobian>(computation_variables);
  auto& inverse_cartesian_to_spherical_jacobian =
      get<Tags::detail::InverseCartesianToSphericalJacobian>(
          computation_variables);
  cartesian_to_spherical_coordinates_and_jacobians(
      make_not_null(&cartesian_coords),
      make_not_null(&cartesian_to_spherical_jacobian),
      make_not_null(&inverse_cartesian_to_spherical_jacobian), cos_phi,
      cos_theta, sin_phi, sin_theta, extraction_radius);

  auto& spatial_metric =
      get<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>(
          computation_variables);
  gr::spatial_metric(make_not_null(&spatial_metric), spacetime_metric);

  auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<3, ::Frame::Inertial, DataVector>>(
          computation_variables);
  // Allocation
  inverse_spatial_metric = determinant_and_inverse(spatial_metric).second;

  auto& shift = get<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>(
      computation_variables);
  gr::shift(make_not_null(&shift), spacetime_metric, inverse_spatial_metric);

  auto& lapse = get<gr::Tags::Lapse<DataVector>>(computation_variables);
  gr::lapse(make_not_null(&lapse), shift, spacetime_metric);

  auto& dt_spacetime_metric = get<
      ::Tags::dt<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>>(
      computation_variables);

  GeneralizedHarmonic::time_derivative_of_spacetime_metric(
      make_not_null(&dt_spacetime_metric), lapse, shift, pi, phi);

  auto& dt_worldtube_normal =
      get<::Tags::dt<Tags::detail::WorldtubeNormal>>(computation_variables);
  auto& worldtube_normal =
      get<Tags::detail::WorldtubeNormal>(computation_variables);
  worldtube_normal_and_derivatives(
      make_not_null(&worldtube_normal), make_not_null(&dt_worldtube_normal),
      cos_phi, cos_theta, spacetime_metric, dt_spacetime_metric, sin_phi,
      sin_theta, inverse_spatial_metric);
  auto& spacetime_unit_normal =
      get<gr::Tags::SpacetimeNormalVector<3, ::Frame::Inertial, DataVector>>(
          computation_variables);
  gr::spacetime_normal_vector(make_not_null(&spacetime_unit_normal), lapse,
                              shift);
  auto& dt_lapse =
      get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(computation_variables);
  GeneralizedHarmonic::time_deriv_of_lapse(
      make_not_null(&dt_lapse), lapse, shift, spacetime_unit_normal, phi, pi);
  auto& dt_shift =
      get<::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>(
          computation_variables);
  GeneralizedHarmonic::time_deriv_of_shift(make_not_null(&dt_shift), lapse,
                                           shift, inverse_spatial_metric,
                                           spacetime_unit_normal, phi, pi);

  auto& du_null_l = get<::Tags::dt<Tags::detail::NullL>>(computation_variables);
  auto& null_l = get<Tags::detail::NullL>(computation_variables);
  null_vector_l_and_derivatives(make_not_null(&du_null_l),
                                make_not_null(&null_l), dt_worldtube_normal,
                                dt_lapse, dt_spacetime_metric, dt_shift, lapse,
                                spacetime_metric, shift, worldtube_normal);

  // pass to the next step that is common between the 'modal' input and 'GH'
  // input strategies
  detail::create_bondi_boundary_data(
      bondi_boundary_data, make_not_null(&computation_variables),
      make_not_null(&derivative_buffers), dt_spacetime_metric, phi,
      spacetime_metric, null_l, du_null_l, cartesian_to_spherical_jacobian,
      l_max, extraction_radius);
}

/*!
 * \brief Process the worldtube data from modal metric components and
 * derivatives to desired Bondi quantities, placing the result in the passed
 * `Variables`.
 *
 * \details
 * The mathematics are a bit complicated for all of the coordinate
 * transformations that are necessary to obtain the Bondi gauge quantities.
 * For full mathematical details, see the documentation for functions in
 * `BoundaryData.hpp` and \cite Barkett2019uae \cite Bishop1998uk.
 *
 * This function takes as input the full set of ADM metric data and its radial
 * and time derivatives on a two-dimensional surface of constant \f$r\f$ and
 * \f$t\f$ in numerical coordinates. This data must be provided as spherical
 * harmonic coefficients in the libsharp format. This data is provided in nine
 * `Tensor`s.
 *
 * Sufficient tags to provide full worldtube boundary data at a particular
 * time are set in `bondi_boundary_data`. In particular, the set of tags in
 * `Tags::characteristic_worldtube_boundary_tags` in the provided `Variables`
 * are assigned to the worldtube boundary values associated with the input
 * metric components.
 *
 * The majority of the mathematical transformations are implemented as a set of
 * individual cascaded functions below. The details of the manipulations that
 * are performed to the input data may be found in the individual functions
 * themselves, which are called in the following order:
 * - `trigonometric_functions_on_swsh_collocation()`
 * - `cartesian_to_spherical_coordinates_and_jacobians()`
 * - `cartesian_spatial_metric_and_derivatives_from_modes()`
 * - `cartesian_shift_and_derivatives_from_modes()`
 * - `cartesian_lapse_and_derivatives_from_modes()`
 * - `GeneralizedHarmonic::phi()`
 * - `gr::time_derivative_of_spacetime_metric`
 * - `gr::spacetime_metric`
 * - `generalized_harmonic_quantities()`
 * - `worldtube_normal_and_derivatives()`
 * - `null_vector_l_and_derivatives()`
 * - `null_metric_and_derivative()`
 * - `dlambda_null_metric_and_inverse()`
 * - `bondi_r()`
 * - `d_bondi_r()`
 * - `dyads()`
 * - `beta_worldtube_data()`
 * - `bondi_u_worldtube_data()`
 * - `bondi_w_worldtube_data()`
 * - `bondi_j_worldtube_data()`
 * - `dr_bondi_j()`
 * - `d2lambda_bondi_r()`
 * - `bondi_q_worldtube_data()`
 * - `bondi_h_worldtube_data()`
 * - `du_j_worldtube_data()`
 */
template <typename BoundaryTagList>
void create_bondi_boundary_data(
    const gsl::not_null<Variables<BoundaryTagList>*> bondi_boundary_data,
    const tnsr::ii<ComplexModalVector, 3>& spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dt_spatial_metric_coefficients,
    const tnsr::ii<ComplexModalVector, 3>& dr_spatial_metric_coefficients,
    const tnsr::I<ComplexModalVector, 3>& shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dt_shift_coefficients,
    const tnsr::I<ComplexModalVector, 3>& dr_shift_coefficients,
    const Scalar<ComplexModalVector>& lapse_coefficients,
    const Scalar<ComplexModalVector>& dt_lapse_coefficients,
    const Scalar<ComplexModalVector>& dr_lapse_coefficients,
    const double extraction_radius, const size_t l_max) {
  const size_t size = Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  // Most allocations required for the full boundary computation are merged into
  // a single, large Variables allocation. There remain a handful of cases in
  // the computational functions called where an intermediate quantity that is
  // not re-used is allocated rather than taking a buffer. These cases are
  // marked with code comments 'Allocation'; In the future, if allocations are
  // identified as a point to optimize, those buffers may be allocated here and
  // passed as function arguments
  Variables<tmpl::list<
      Tags::detail::CosPhi, Tags::detail::CosTheta, Tags::detail::SinPhi,
      Tags::detail::SinTheta, Tags::detail::CartesianCoordinates,
      Tags::detail::CartesianToSphericalJacobian,
      Tags::detail::InverseCartesianToSphericalJacobian,
      gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>,
      gr::Tags::InverseSpatialMetric<3, ::Frame::Inertial, DataVector>,
      ::Tags::deriv<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, ::Frame::Inertial>,
      ::Tags::dt<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>,
      gr::Tags::Shift<3, ::Frame::Inertial, DataVector>,
      ::Tags::deriv<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, ::Frame::Inertial>,
      ::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>,
      gr::Tags::Lapse<DataVector>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    ::Frame::Inertial>,
      ::Tags::dt<gr::Tags::Lapse<DataVector>>,
      gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>,
      GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>,
      Tags::detail::WorldtubeNormal, ::Tags::dt<Tags::detail::WorldtubeNormal>,
      Tags::detail::NullL, ::Tags::dt<Tags::detail::NullL>,
      // for the detail function called at the end
      gr::Tags::SpacetimeMetric<3, Frame::RadialNull, DataVector>,
      ::Tags::dt<gr::Tags::SpacetimeMetric<3, Frame::RadialNull, DataVector>>,
      gr::Tags::InverseSpacetimeMetric<3, Frame::RadialNull, DataVector>,
      Tags::detail::AngularDNullL,
      Tags::detail::DLambda<
          gr::Tags::SpacetimeMetric<3, Frame::RadialNull, DataVector>>,
      Tags::detail::DLambda<
          gr::Tags::InverseSpacetimeMetric<3, Frame::RadialNull, DataVector>>,
      ::Tags::spacetime_deriv<Tags::detail::RealBondiR, tmpl::size_t<3>,
                              Frame::RadialNull>,
      Tags::detail::DLambda<Tags::detail::DLambda<Tags::detail::RealBondiR>>,
      ::Tags::deriv<Tags::detail::DLambda<Tags::detail::RealBondiR>,
                    tmpl::size_t<2>, Frame::RadialNull>>>
      computation_variables{size};

  Variables<
      tmpl::list<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 0>>,
                 ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 1>>>>
      derivative_buffers{size};
  auto& cos_phi = get<Tags::detail::CosPhi>(computation_variables);
  auto& cos_theta = get<Tags::detail::CosTheta>(computation_variables);
  auto& sin_phi = get<Tags::detail::SinPhi>(computation_variables);
  auto& sin_theta = get<Tags::detail::SinTheta>(computation_variables);
  trigonometric_functions_on_swsh_collocation(
      make_not_null(&cos_phi), make_not_null(&cos_theta),
      make_not_null(&sin_phi), make_not_null(&sin_theta), l_max);

  // NOTE: to handle the singular values of polar coordinates, the phi
  // components of all tensors are scaled according to their sin(theta)
  // prefactors.
  // so, any down-index component get<2>(A) represents 1/sin(theta) A_\phi,
  // and any up-index component get<2>(A) represents sin(theta) A^\phi.
  // This holds for Jacobians, and so direct application of the Jacobians
  // brings the factors through.
  auto& cartesian_coords =
      get<Tags::detail::CartesianCoordinates>(computation_variables);
  auto& cartesian_to_spherical_jacobian =
      get<Tags::detail::CartesianToSphericalJacobian>(computation_variables);
  auto& inverse_cartesian_to_spherical_jacobian =
      get<Tags::detail::InverseCartesianToSphericalJacobian>(
          computation_variables);
  cartesian_to_spherical_coordinates_and_jacobians(
      make_not_null(&cartesian_coords),
      make_not_null(&cartesian_to_spherical_jacobian),
      make_not_null(&inverse_cartesian_to_spherical_jacobian), cos_phi,
      cos_theta, sin_phi, sin_theta, extraction_radius);

  auto& cartesian_spatial_metric =
      get<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>(
          computation_variables);
  auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<3, ::Frame::Inertial, DataVector>>(
          computation_variables);
  auto& d_cartesian_spatial_metric = get<
      ::Tags::deriv<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, ::Frame::Inertial>>(computation_variables);
  auto& dt_cartesian_spatial_metric = get<
      ::Tags::dt<gr::Tags::SpatialMetric<3, ::Frame::Inertial, DataVector>>>(
      computation_variables);
  auto& interpolation_buffer =
      get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                               std::integral_constant<int, 0>>>(
          derivative_buffers);
  Scalar<SpinWeighted<ComplexModalVector, 0>> interpolation_modal_buffer{size};
  auto& eth_buffer =
      get<::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                               std::integral_constant<int, 1>>>(
          derivative_buffers);
  cartesian_spatial_metric_and_derivatives_from_modes(
      make_not_null(&cartesian_spatial_metric),
      make_not_null(&inverse_spatial_metric),
      make_not_null(&d_cartesian_spatial_metric),
      make_not_null(&dt_cartesian_spatial_metric),
      make_not_null(&interpolation_modal_buffer),
      make_not_null(&interpolation_buffer), make_not_null(&eth_buffer),
      spatial_metric_coefficients, dr_spatial_metric_coefficients,
      dt_spatial_metric_coefficients, inverse_cartesian_to_spherical_jacobian,
      l_max);

  auto& cartesian_shift =
      get<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>(
          computation_variables);
  auto& d_cartesian_shift =
      get<::Tags::deriv<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>,
                        tmpl::size_t<3>, ::Frame::Inertial>>(
          computation_variables);
  auto& dt_cartesian_shift =
      get<::Tags::dt<gr::Tags::Shift<3, ::Frame::Inertial, DataVector>>>(
          computation_variables);

  cartesian_shift_and_derivatives_from_modes(
      make_not_null(&cartesian_shift), make_not_null(&d_cartesian_shift),
      make_not_null(&dt_cartesian_shift),
      make_not_null(&interpolation_modal_buffer),
      make_not_null(&interpolation_buffer), make_not_null(&eth_buffer),
      shift_coefficients, dr_shift_coefficients, dt_shift_coefficients,
      inverse_cartesian_to_spherical_jacobian, l_max);

  auto& cartesian_lapse =
      get<gr::Tags::Lapse<DataVector>>(computation_variables);
  auto& d_cartesian_lapse =
      get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                        ::Frame::Inertial>>(computation_variables);
  auto& dt_cartesian_lapse =
      get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(computation_variables);
  cartesian_lapse_and_derivatives_from_modes(
      make_not_null(&cartesian_lapse), make_not_null(&d_cartesian_lapse),
      make_not_null(&dt_cartesian_lapse),
      make_not_null(&interpolation_modal_buffer),
      make_not_null(&interpolation_buffer), make_not_null(&eth_buffer),
      lapse_coefficients, dr_lapse_coefficients, dt_lapse_coefficients,
      inverse_cartesian_to_spherical_jacobian, l_max);

  auto& phi = get<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(
      computation_variables);
  auto& dt_spacetime_metric = get<
      ::Tags::dt<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>>(
      computation_variables);
  auto& spacetime_metric =
      get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>(
          computation_variables);
  GeneralizedHarmonic::phi(
      make_not_null(&phi), cartesian_lapse, d_cartesian_lapse, cartesian_shift,
      d_cartesian_shift, cartesian_spatial_metric, d_cartesian_spatial_metric);
  gr::time_derivative_of_spacetime_metric(
      make_not_null(&dt_spacetime_metric), cartesian_lapse, dt_cartesian_lapse,
      cartesian_shift, dt_cartesian_shift, cartesian_spatial_metric,
      dt_cartesian_spatial_metric);
  gr::spacetime_metric(make_not_null(&spacetime_metric), cartesian_lapse,
                       cartesian_shift, cartesian_spatial_metric);

  auto& dt_worldtube_normal =
      get<::Tags::dt<Tags::detail::WorldtubeNormal>>(computation_variables);
  auto& worldtube_normal =
      get<Tags::detail::WorldtubeNormal>(computation_variables);
  worldtube_normal_and_derivatives(
      make_not_null(&worldtube_normal), make_not_null(&dt_worldtube_normal),
      cos_phi, cos_theta, spacetime_metric, dt_spacetime_metric, sin_phi,
      sin_theta, inverse_spatial_metric);

  auto& du_null_l = get<::Tags::dt<Tags::detail::NullL>>(computation_variables);
  auto& null_l = get<Tags::detail::NullL>(computation_variables);
  null_vector_l_and_derivatives(
      make_not_null(&du_null_l), make_not_null(&null_l), dt_worldtube_normal,
      dt_cartesian_lapse, dt_spacetime_metric, dt_cartesian_shift,
      cartesian_lapse, spacetime_metric, cartesian_shift, worldtube_normal);

  // pass to the next step that is common between the 'modal' input and 'GH'
  // input strategies
  detail::create_bondi_boundary_data(
      bondi_boundary_data, make_not_null(&computation_variables),
      make_not_null(&derivative_buffers), dt_spacetime_metric, phi,
      spacetime_metric, null_l, du_null_l, cartesian_to_spherical_jacobian,
      l_max, extraction_radius);
}
}  // namespace Cce
