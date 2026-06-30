// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Cce::InitializeJ::CauchySecondOrder_detail {

/*!
 * \brief Solve the H-hypersurface equation at the worldtube for
 * \f$\partial_y^2 J\f$.
 *
 * \details Evaluating the H-hypersurface equation at \f$y=-1\f$ and isolating
 * the contributions coming from the swsh Jacobians at the worldtube produces
 * the complex identity
 *
 * \f[
 *   c_1 \, \partial_y^2 J + c_2 \, \partial_y^2 \bar J = -c_3,
 * \f]
 *
 * where \f$c_1\f$, \f$c_2\f$, and \f$c_3\f$ depend only on quantities known at
 * the worldtube. The conjugate of this identity eliminates
 * \f$\partial_y^2 \bar J\f$ and yields \f$\partial_y^2 J\f$.
 *
 * The whole computation reuses the volume CCE machinery rather than any
 * hand-written angular-Jacobian expressions. The worldtube right-minus-left
 * side of the H-hypersurface equation is assembled as a function
 * \f$F(\partial_y^2 J)\f$: the angular derivatives come from
 * `Spectral::Swsh::angular_derivatives` and are converted from the numerical
 * (constant \f$y\f$) to the physical (constant \f$r\f$) coordinate with
 * `Cce::ApplySwshJacobianInplace`, and the hypersurface right-hand sides use
 * the same `Cce::ComputeBondiIntegrand` specializations as the evolution, with
 * \f$\partial_y^2 J\f$ threaded through every dependence. Because \f$F\f$ is
 * affine in \f$(\partial_y^2 J, \partial_y^2 \bar J)\f$, evaluating it at
 * \f$\partial_y^2 J = 0, 1, i\f$ fixes \f$c_3 = F(0)\f$ and the linear
 * coefficients \f$c_1\f$, \f$c_2\f$ exactly.
 *
 * The caller supplies the physical worldtube data; `compute_dy_dy_j` converts
 * it to the numerical (constant \f$y\f$) coordinate that
 * `evaluate_worldtube_h_residual` works in, using the worldtube Jacobian
 * \f$\partial_y J = (R / 2) \partial_r J\f$,
 *
 * \f{align*}{
 *   \partial_y J &= \tfrac{1}{2} R \, \partial_r J, \\
 *   \breve{H} &= H + \partial_u R \, \partial_r J, \\
 *   \partial_y \breve{H} &= \tfrac{1}{2}\left(\partial_u R \, \partial_r J
 *                          + R \, \partial_{\breve u} \partial_r J\right),
 * \f}
 *
 * where \f$\breve{H} = \partial_{\breve u} J = (\partial_u J)_y\f$ is the
 * numerical-coordinate \f$H\f$. The time derivative
 * enters only through the primitive radial quantity
 * `du_dr_j` \f$= \partial_{\breve u} \partial_r J\f$.
 *
 * \note The coordinate held fixed by a time derivative is written two
 * equivalent ways: a breve accent marks "at constant numerical coordinate
 * \f$y\f$" (following the worldtube), while an unaccented symbol means "at
 * constant Bondi \f$r\f$"; `BoundaryData.hpp` writes the same distinction with
 * an explicit \f$(\,\cdot\,)_y\f$ / \f$(\,\cdot\,)_r\f$ subscript. So, for
 * \f$H\f$,
 * - \f$\breve{H} = (\partial_u J)_y\f$ (constant \f$y\f$, "numerical")
 *   \f$\;\leftrightarrow\;\f$ `Cce::Tags::BondiH` (`= ::Tags::dt<BondiJ>`),
 * - \f$H = (\partial_u J)_r\f$ (constant Bondi \f$r\f$)
 *   \f$\;\leftrightarrow\;\f$ `Cce::Tags::Du<BondiJ>`.
 *
 * The tag \f$\breve{H}\f$ drops the accent because the evolution only ever uses
 * the constant-\f$y\f$ \f$H\f$, so there is nothing to distinguish it from.
 *
 * \see evaluate_worldtube_h_residual for the function \f$F\f$ that is probed.
 */
void compute_dy_dy_j(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> dy_dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& q,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& du_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& du_dr_j,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max);

/*!
 * \brief The worldtube residual \f$F(\partial_y^2 J)\f$ of the H-hypersurface
 * equation whose root `compute_dy_dy_j` returns.
 *
 * \details Returns the worldtube (\f$y = -1\f$) right-minus-left side of the
 * H-hypersurface equation,
 *
 * \f[
 *   F(\partial_y^2 J) = \mathrm{rhs}(H) - \mathrm{lhs}(H),
 * \f]
 *
 * evaluated with the supplied trial value `dy_dy_j_value` for
 * \f$\partial_y^2 J\f$ (with \f$\partial_y^2 \bar J\f$ taken to be its complex
 * conjugate). All worldtube inputs are supplied in the numerical (constant
 * \f$y\f$) coordinate: `dy_j` \f$= \partial_y J\f$, `h`
 * \f$= \breve{H} = (\partial_u J)_y\f$, and `dy_h`
 * \f$= \partial_y \breve{H}\f$; no physical radial-derivative quantity is
 * passed. Every angular derivative is converted from the numerical to the
 * physical coordinate with `Cce::ApplySwshJacobianInplace`, and every
 * hypersurface right-hand side comes from the `Cce::ComputeBondiIntegrand`
 * specializations, so the residual is assembled entirely from the same volume
 * machinery the evolution uses. `compute_dy_dy_j` returns the value of
 * \f$\partial_y^2 J\f$ for which this residual vanishes, so feeding that value
 * back in reproduces the H-hypersurface equation to machine precision.
 */
Scalar<SpinWeighted<ComplexDataVector, 2>> evaluate_worldtube_h_residual(
    const ComplexDataVector& dy_dy_j_value,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& j,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta,
    const Scalar<SpinWeighted<ComplexDataVector, 1>>& q,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_j,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& h,
    const Scalar<SpinWeighted<ComplexDataVector, 2>>& dy_h,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& r, size_t l_max);

}  // namespace Cce::InitializeJ::CauchySecondOrder_detail
