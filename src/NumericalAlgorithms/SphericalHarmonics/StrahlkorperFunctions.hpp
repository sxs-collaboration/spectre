// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <deque>
#include <utility>

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TagsTypeAliases.hpp"

/// \cond
class DataVector;
namespace ylm {
template <typename Fr>
class Strahlkorper;
}  // namespace ylm
template <typename X, typename Symm, typename IndexList>
class Tensor;
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

/// \ingroup SurfacesGroup
/// Contains functions that depend on a Strahlkorper but not on a metric.
namespace ylm {
/// @{
/*!
 * \f$(\theta,\phi)\f$ on the Strahlkorper surface.
 * Doesn't depend on the shape of the surface.
 *
 * We need to choose upper vs lower indices for theta_phi; it doesn't
 * matter because these are coordinates and not geometric objects, so
 * we choose lower indices arbitrarily.
 */
template <typename Fr>
tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>> theta_phi(
    const Strahlkorper<Fr>& strahlkorper);

template <typename Fr>
void theta_phi(
    const gsl::not_null<tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>*>
        theta_phi,
    const Strahlkorper<Fr>& strahlkorper);
/// @}

/// @{
/*!
 * `r_hat(i)` is \f$\hat{r}_i = x_i/\sqrt{x^2+y^2+z^2}\f$ on the
 * Strahlkorper surface.  Doesn't depend on the shape of the surface.
 *
 * We need to choose upper vs lower indices for rhat; it doesn't
 * matter because rhat is a quantity defined with a Euclidean metric,
 * so we choose the lower index arbitrarily.
 */
template <typename Fr>
tnsr::i<DataVector, 3, Fr> rhat(
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi);

template <typename Fr>
void rhat(const gsl::not_null<tnsr::i<DataVector, 3, Fr>*> r_hat,
          const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi);
/// @}

/// @{
/*!
 * `Jacobian(i,0)` is \f$\frac{1}{r}\partial x^i/\partial\theta\f$,
 * and `Jacobian(i,1)`
 * is \f$\frac{1}{r\sin\theta}\partial x^i/\partial\phi\f$.
 * Here \f$r\f$ means \f$\sqrt{x^2+y^2+z^2}\f$.
 * `Jacobian` doesn't depend on the shape of the surface.
 */
template <typename Fr>
ylm::Tags::aliases::Jacobian<Fr> jacobian(
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi);

template <typename Fr>
void jacobian(const gsl::not_null<ylm::Tags::aliases::Jacobian<Fr>*> jac,
              const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi);
/// @}

/// @{
/*!
 * `InvJacobian(0,i)` is \f$r\partial\theta/\partial x^i\f$,
 * and `InvJacobian(1,i)` is \f$r\sin\theta\partial\phi/\partial x^i\f$.
 * Here \f$r\f$ means \f$\sqrt{x^2+y^2+z^2}\f$.
 * `InvJacobian` doesn't depend on the shape of the surface.
 */
template <typename Fr>
ylm::Tags::aliases::InvJacobian<Fr> inv_jacobian(
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi);

template <typename Fr>
void inv_jacobian(
    const gsl::not_null<ylm::Tags::aliases::InvJacobian<Fr>*> inv_jac,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi);
/// @}

/// @{
/*!
 * `InvHessian(k,i,j)` is \f$r\partial (J^{-1}){}^k_j/\partial x^i\f$,
 * where \f$(J^{-1}){}^k_j\f$ is the inverse Jacobian.
 * Here \f$r\f$ means \f$\sqrt{x^2+y^2+z^2}\f$.
 * `InvHessian` is not symmetric because the Jacobians are Pfaffian.
 * `InvHessian` doesn't depend on the shape of the surface.
 */
template <typename Fr>
ylm::Tags::aliases::InvHessian<Fr> inv_hessian(
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi);

template <typename Fr>
void inv_hessian(
    const gsl::not_null<ylm::Tags::aliases::InvHessian<Fr>*> inv_hess,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi);
/// @}

/// @{
/*!
 * (Euclidean) distance \f$r_{\rm surf}(\theta,\phi)\f$ from the
 * expansion center to each point of the Strahlkorper surface.
 */
template <typename Fr>
Scalar<DataVector> radius(const Strahlkorper<Fr>& strahlkorper);

template <typename Fr>
void radius(const gsl::not_null<Scalar<DataVector>*> result,
            const Strahlkorper<Fr>& strahlkorper);
/// @}

/// @{
/*!
 * `cartesian_coords(i)` is \f$x_{\rm surf}^i\f$, the vector of \f$(x,y,z)\f$
 * coordinates of each point on the Strahlkorper surface.
 *
 * \param strahlkorper The Strahlkorper surface.
 * \param radius The radius as a function of angle, as returned by
 * `ylm::radius`.
 * \param r_hat The Euclidean radial unit vector as returned by
 * `ylm::rhat`.
 */
template <typename Fr>
tnsr::I<DataVector, 3, Fr> cartesian_coords(
    const Strahlkorper<Fr>& strahlkorper, const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Fr>& r_hat);

/*!
 * \param coords The returned Cartesian coordinates.
 * \param strahlkorper The Strahlkorper surface.
 * \param radius The radius as a function of angle, as returned by
 * `ylm::radius`.
 * \param r_hat The Euclidean radial unit vector as returned by
 * `ylm::rhat`.
 */
template <typename Fr>
void cartesian_coords(const gsl::not_null<tnsr::I<DataVector, 3, Fr>*> coords,
                      const Strahlkorper<Fr>& strahlkorper,
                      const Scalar<DataVector>& radius,
                      const tnsr::i<DataVector, 3, Fr>& r_hat);

/*!
 * This overload computes `radius`, `theta_phi`, and `r_hat` internally.
 * Use the other overloads if you already have these quantities.
 *
 * \param strahlkorper The Strahlkorper surface.
 */
template <typename Fr>
tnsr::I<DataVector, 3, Fr> cartesian_coords(
    const Strahlkorper<Fr>& strahlkorper);
/// @}

/// @{
/*!
 * `dx_scalar(i)` is \f$\partial f/\partial x^i\f$ evaluated on the
 * surface.  Here \f$f=f(r,\theta,\phi)=f(\theta,\phi)\f$ is some
 * scalar function independent of the radial coordinate. \f$f\f$ is
 * considered a function of Cartesian coordinates
 * \f$f=f(\theta(x,y,z),\phi(x,y,z))\f$ for this operation.
 *
 * \param scalar The scalar to be differentiated.
 * \param strahlkorper The Strahlkorper surface.
 * \param radius_of_strahlkorper The radius of the Strahlkorper at each
 * point, as returned by `ylm::radius`.
 * \param inv_jac The inverse Jacobian as returned by
 * `ylm::inv_jacobian`
 */
template <typename Fr>
tnsr::i<DataVector, 3, Fr> cartesian_derivs_of_scalar(
    const Scalar<DataVector>& scalar, const Strahlkorper<Fr>& strahlkorper,
    const Scalar<DataVector>& radius_of_strahlkorper,
    const ylm::Tags::aliases::InvJacobian<Fr>& inv_jac);

/*!
 * \param dx_scalar The returned derivatives of the scalar.
 * \param scalar The scalar to be differentiated.
 * \param strahlkorper The Strahlkorper surface.
 * \param radius_of_strahlkorper The radius of the Strahlkorper at each
 * point, as returned by `ylm::radius`.
 * \param inv_jac The inverse Jacobian as returned by
 * `ylm::inv_jacobian`
 */
template <typename Fr>
void cartesian_derivs_of_scalar(
    const gsl::not_null<tnsr::i<DataVector, 3, Fr>*> dx_scalar,
    const Scalar<DataVector>& scalar, const Strahlkorper<Fr>& strahlkorper,
    const Scalar<DataVector>& radius_of_strahlkorper,
    const ylm::Tags::aliases::InvJacobian<Fr>& inv_jac);
/// @}

/// @{
/*!
 * `d2x_scalar(i,j)` is \f$\partial^2 f/\partial x^i\partial x^j\f$
 * evaluated on the surface. Here
 * \f$f=f(r,\theta,\phi)=f(\theta,\phi)\f$ is some scalar function
 * independent of the radial coordinate. \f$f\f$ is considered a
 * function of Cartesian coordinates
 * \f$f=f(\theta(x,y,z),\phi(x,y,z))\f$ for this operation.
 *
 * \param scalar The scalar to be differentiated.
 * \param strahlkorper The Strahlkorper surface.
 * \param radius_of_strahlkorper The radius of the Strahlkorper at each
 * point, as returned by `ylm::radius`.
 * \param inv_jac The inverse Jacobian as returned by
 * `ylm::inv_jacobian`
 * \param inv_hess The inverse Hessian as returned by
 * `ylm::inv_hessian.
 */
template <typename Fr>
tnsr::ii<DataVector, 3, Fr> cartesian_second_derivs_of_scalar(
    const Scalar<DataVector>& scalar, const Strahlkorper<Fr>& strahlkorper,
    const Scalar<DataVector>& radius_of_strahlkorper,
    const ylm::Tags::aliases::InvJacobian<Fr>& inv_jac,
    const ylm::Tags::aliases::InvHessian<Fr>& inv_hess);

/*!
 * \param d2x_scalar The returned 2nd derivatives of the scalar.
 * \param scalar The scalar to be differentiated.
 * \param strahlkorper The Strahlkorper surface.
 * \param radius_of_strahlkorper The radius of the Strahlkorper at each
 * point, as returned by `ylm::radius`.
 * \param inv_jac The inverse Jacobian as returned by
 * `ylm::inv_jacobian`
 * \param inv_hess The inverse Hessian as returned by
 * `ylm::inv_hessian.
 */
template <typename Fr>
void cartesian_second_derivs_of_scalar(
    const gsl::not_null<tnsr::ii<DataVector, 3, Fr>*> d2x_scalar,
    const Scalar<DataVector>& scalar, const Strahlkorper<Fr>& strahlkorper,
    const Scalar<DataVector>& radius_of_strahlkorper,
    const ylm::Tags::aliases::InvJacobian<Fr>& inv_jac,
    const ylm::Tags::aliases::InvHessian<Fr>& inv_hess);
/// @}

/// @{
/*!
 * \f$\nabla^2 f\f$, the flat Laplacian of a scalar \f$f\f$ on the surface.
 * This is \f$\eta^{ij}\partial^2 f/\partial x^i\partial x^j\f$,
 * where \f$f=f(r,\theta,\phi)=f(\theta,\phi)\f$ is some scalar function
 * independent of the radial coordinate. \f$f\f$ is considered a
 * function of Cartesian coordinates
 * \f$f=f(\theta(x,y,z),\phi(x,y,z))\f$ for this operation.
 *
 */
template <typename Fr>
Scalar<DataVector> laplacian_of_scalar(
    const Scalar<DataVector>& scalar, const Strahlkorper<Fr>& strahlkorper,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi);

template <typename Fr>
void laplacian_of_scalar(
    const gsl::not_null<Scalar<DataVector>*> laplacian,
    const Scalar<DataVector>& scalar, const Strahlkorper<Fr>& strahlkorper,
    const tnsr::i<DataVector, 2, ::Frame::Spherical<Fr>>& theta_phi);
/// @}

/// @{
/*!
 * `tangents(i,j)` is \f$\partial x_{\rm surf}^i/\partial q^j\f$,
 * where \f$x_{\rm surf}^i\f$ are the Cartesian coordinates of the
 * surface (i.e. `cartesian_coords`) and are considered functions of
 * \f$(\theta,\phi)\f$.
 *
 * \f$\partial/\partial q^0\f$ means
 * \f$\partial/\partial\theta\f$; and \f$\partial/\partial q^1\f$
 * means \f$\csc\theta\,\,\partial/\partial\phi\f$.  Note that the
 * vectors `tangents(i,0)` and `tangents(i,1)` are orthogonal to the
 * `normal_one_form` \f$s_i\f$, i.e.
 * \f$s_i \partial x_{\rm surf}^i/\partial q^j = 0\f$; this statement
 * is independent of a metric.  Also, `tangents(i,0)` and
 * `tangents(i,1)` are not necessarily orthogonal to each other,
 * since orthogonality between 2 vectors (as opposed to a vector and
 * a one-form) is metric-dependent.
 *
 * \param strahlkorper The Strahlkorper surface.
 * \param radius The radius of the Strahlkorper at each
 * point, as returned by `ylm::radius`.
 * \param r_hat The radial unit vector as returned by
 * `ylm::rhat`.
 * \param jac The jacobian as returned by `ylm::jacobian`.
 */
template <typename Fr>
ylm::Tags::aliases::Jacobian<Fr> tangents(
    const Strahlkorper<Fr>& strahlkorper, const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, 3, Fr>& r_hat,
    const ylm::Tags::aliases::Jacobian<Fr>& jac);

/*!
 * \param result The computed tangent vectors.
 * \param strahlkorper The Strahlkorper surface.
 * \param radius The radius of the Strahlkorper at each
 * point, as returned by `ylm::radius`.
 * \param r_hat The radial unit vector as returned by
 * `ylm::rhat`.
 * \param jac The jacobian as returned by `ylm::jacobian`.
 */
template <typename Fr>
void tangents(const gsl::not_null<ylm::Tags::aliases::Jacobian<Fr>*> result,
              const Strahlkorper<Fr>& strahlkorper,
              const Scalar<DataVector>& radius,
              const tnsr::i<DataVector, 3, Fr>& r_hat,
              const ylm::Tags::aliases::Jacobian<Fr>& jac);
/// @}

/// @{
/*!
 * `normal_one_form(i)` is \f$s_i\f$, the (unnormalized) normal one-form
 * to the surface, expressed in Cartesian components.
 * This is computed by \f$x_i/r-\partial r_{\rm surf}/\partial x^i\f$,
 * where \f$x_i/r\f$ is `Rhat` and
 * \f$\partial r_{\rm surf}/\partial x^i\f$ is `DxRadius`.
 * See Eq. (8) of \cite Baumgarte1996hh.
 * Note on the word "normal": \f$s_i\f$ points in the correct direction
 * (it is "normal" to the surface), but it does not have unit length
 * (it is not "normalized"; normalization requires a metric).
 *
 * \param dx_radius The Cartesian derivatives of the radius, as
 * returned by ylm::cartesian_derivs_of_scalar with
 * `ylm::radius` passed in as the scalar.
 * \param r_hat The radial unit vector as returned by
 * `ylm::rhat`.
 */
template <typename Fr>
tnsr::i<DataVector, 3, Fr> normal_one_form(
    const tnsr::i<DataVector, 3, Fr>& dx_radius,
    const tnsr::i<DataVector, 3, Fr>& r_hat);

/*!
 * \param one_form The returned normal one form.
 * \param dx_radius The Cartesian derivatives of the radius, as
 * returned by ylm::cartesian_derivs_of_scalar with
 * `ylm::radius` passed in as the scalar.
 * \param r_hat The radial unit vector as returned by
 * `ylm::rhat`.
 */
template <typename Fr>
void normal_one_form(const gsl::not_null<tnsr::i<DataVector, 3, Fr>*> one_form,
                     const tnsr::i<DataVector, 3, Fr>& dx_radius,
                     const tnsr::i<DataVector, 3, Fr>& r_hat);

/// @{
/*!
 * The linear least squares fit of the polynomial of order 3
 * given a `std::vector` of `Strahlkorper`s to their \f$Y_l^m\f$ coefficients.
 * Assumes the the \f$l_{\max}\f$ and \f$m_{\max}\f$ of each `Strahlkorper`
 * are the same, and the returned vector consists of \f$2l_{\max}m_{\max}\f$
 * (the number of \f$Y_l^m\f$ coefficients) `std::array<double, 4>`s, each of
 * which consists of the four coefficients that define the best fit cubic to
 * each \f$Y_l^m\f$ coefficient of the `Strahlkorper` as a function of time.
 *
 * \param times The time corresponding to each `Strahlkorper` to be fit to.
 * \param strahlkorpers The `Strahlkorper` surfaces which consists of a set
 * of \f$Y_l^m\f$ coefficients corresponding to the shape of the `Strahlkorper`
 * at a particular time.
 */
template <typename Fr>
std::vector<std::array<double, 4>> fit_ylm_coeffs(
    const DataVector& times,
    const std::vector<Strahlkorper<Fr>>& strahlkorpers);

/*!
 * \brief Compute the time derivative of a Strahlkorper from a number of
 * previous Strahlkorpers
 *
 * \details Does simple 1D FD with non-uniform spacing using
 * `fd::non_uniform_1d_weights`.
 * \param time_deriv Strahlkorper whose coefficients are the time derivative of
 * `previous_strahlkorpers`' coefficients.
 * \param previous_strahlkorpers All previous Strahlkorpers and the times they
 * are at. They are expected to have the most recent Strahlkorper in the front
 * and the Strahlkorper furthest in the past in the back of the deque.
 */
template <typename Frame>
void time_deriv_of_strahlkorper(
    gsl::not_null<Strahlkorper<Frame>*> time_deriv,
    const std::deque<std::pair<double, Strahlkorper<Frame>>>&
        previous_strahlkorpers);
}  // namespace ylm
