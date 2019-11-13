// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/BoundaryDataTags.hpp"

/// \cond
class DataVector;
class ComplexDataVector;
/// \endcond

namespace Cce {
// tensor aliases for brevity
using jacobian_tensor = Tensor<
    DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
    index_list<SpatialIndex<3, UpLo::Lo, ::Frame::Spherical<::Frame::Inertial>>,
               SpatialIndex<3, UpLo::Up, ::Frame::Inertial>>>;

using inverse_jacobian_tensor =
    Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
           index_list<SpatialIndex<3, UpLo::Lo, ::Frame::Inertial>,
                      SpatialIndex<3, UpLo::Up,
                                   ::Frame::Spherical<::Frame::Inertial>>>>;

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
    gsl::not_null<jacobian_tensor*> cartesian_to_spherical_jacobian,
    gsl::not_null<inverse_jacobian_tensor*>
        inverse_cartesian_to_spherical_jacobian,
    const Scalar<DataVector>& cos_phi, const Scalar<DataVector>& cos_theta,
    const Scalar<DataVector>& sin_phi, const Scalar<DataVector>& sin_theta,
    double extraction_radius) noexcept;

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
    const jacobian_tensor& cartesian_to_spherical_jacobian,
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
    const tnsr::iA<DataVector, 3>& angular_d_null_l,
    const jacobian_tensor& cartesian_to_spherical_jacobian,
    const tnsr::iaa<DataVector, 3>& phi,
    const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const tnsr::A<DataVector, 3>& du_null_l,
    const tnsr::AA<DataVector, 3>& inverse_null_metric,
    const tnsr::A<DataVector, 3>& null_l,
    const tnsr::aa<DataVector, 3>& spacetime_metric) noexcept;

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

}  // namespace Cce
