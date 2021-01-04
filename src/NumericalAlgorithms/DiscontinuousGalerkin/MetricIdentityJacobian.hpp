// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace dg {
/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Compute the Jacobian determinant times the inverse Jacobian so that
 * the result is divergence-free.
 *
 * The metric identities are given by
 *
 * \f{align*}{
 *
 * \partial_{\hat{\imath}}\left(J\frac{\partial\xi^{\hat{\imath}}}
 * {\partial x^i}\right)=0.
 *
 * \f}
 *
 * We want to compute \f$J\partial\xi^{\hat{\imath}}/\partial x^i\f$ in such a
 * way that the above metric identity is satisfied numerically/discretely. We
 * refer to the inverse Jacobian computed this way as the "metric
 * indentity-satisfying inverse Jacobian".
 *
 * The discretized form, with the Jacobian determinant \f$J\f$ expanded, is
 * given by
 *
 * \f{align*}{
 *
 * 2\left(J\frac{\partial \xi^{\hat{\imath}}}{\partial x^i}\right)_{s}
 *  &=\epsilon_{ijk}
 *    \sum_{t_{\hat{1}}} \epsilon^{\hat{\imath}\hat{1}\hat{k}}
 *    D^{(\hat{1})}_{s t_1}
 *    \left(x^j\frac{\partial x^k}{\partial \xi^{\hat{k}}}\right)_{t}
 *    \notag \\
 *  &+\epsilon_{ijk}
 *    \sum_{t_{\hat{2}}} \epsilon^{\hat{\imath}\hat{2}\hat{k}}
 *    D^{(\hat{2})}_{st_2}
 *    \left(x^j\frac{\partial x^k}{\partial \xi^{\hat{k}}}\right)_{t}
 *    \notag \\
 *  &+\epsilon_{ijk}
 *    \sum_{t_{\hat{3}}}\epsilon^{\hat{\imath}\hat{3}\hat{k}}
 *    D^{(\hat{3})}_{st_3}
 *    \left(x^j\frac{\partial x^k}{\partial \xi^{\hat{k}}}\right)_{t}.
 *
 * \f}
 *
 * where indices \f$s,t,t_1,t_2\f$ and \f$t_3\f$ are over grid points. \f$t_i\f$
 * are the grid points in the particular logical direction.
 *
 * In 1d we have:
 *
 * \f{align*}{
 *
 * J\frac{\partial \xi^{\hat{\imath}}}{\partial x^i}=
 * = J\frac{\partial \xi^{\hat{i}}}{\partial x^i} = 1
 *
 * \f}
 *
 * In 2d we have:
 *
 * \f{align*}{
 *
 *  J\frac{\partial \xi^{\hat{1}}}{\partial x^1}&=D^{\hat{(2)}}x^2 &
 *  J\frac{\partial \xi^{\hat{2}}}{\partial x^1}&=-D^{\hat{(1)}}x^2 \\
 *  J\frac{\partial \xi^{\hat{1}}}{\partial x^2}&=-D^{\hat{(2)}}x^1 &
 *  J\frac{\partial \xi^{\hat{2}}}{\partial x^2}&=D^{\hat{(1)}}x^1 \\
 *
 * \f}
 *
 * In 3d we have:
 *
 * \f{align*}{
 *
 * 2J\frac{\partial \xi^{\hat{1}}}{\partial x^1}&=
 * D^{(\hat{2})}\left(x^2\frac{\partial x^3}{\partial \xi^{\hat{3}}}\right)
 * -D^{(\hat{2})}\left(x^3\frac{\partial x^2}{\partial \xi^{\hat{3}}}\right)
 * +D^{(\hat{3})}\left(x^3\frac{\partial x^2}{\partial \xi^{\hat{2}}}\right)
 * -D^{(\hat{3})}\left(x^2\frac{\partial x^3}{\partial \xi^{\hat{2}}}\right)\\
 * 2J\frac{\partial \xi^{\hat{2}}}{\partial x^1}&=
 * D^{(\hat{1})}\left(x^3\frac{\partial x^2}{\partial\xi^{\hat{3}}}\right)
 * -D^{(\hat{1})}\left(x^2\frac{\partial x^3}{\partial\xi^{\hat{3}}}\right)
 * +D^{(\hat{3})}\left(x^2\frac{\partial x^3}{\partial\xi^{\hat{1}}}\right)
 * -D^{(\hat{3})}\left(x^3\frac{\partial x^2}{\partial\xi^{\hat{1}}}\right) \\
 * 2J\frac{\partial \xi^{\hat{3}}}{\partial x^1}&=
 * D^{(\hat{1})}\left(x^2\frac{\partial x^3}{\partial\xi^{\hat{2}}}\right)
 * -D^{(\hat{1})}\left(x^3\frac{\partial x^2}{\partial\xi^{\hat{2}}}\right)
 * +D^{(\hat{2})}\left(x^3\frac{\partial x^2}{\partial\xi^{\hat{1}}}\right)
 * -D^{(\hat{2})}\left(x^2\frac{\partial x^3}{\partial\xi^{\hat{1}}}\right)\\
 * 2J\frac{\partial \xi^{\hat{1}}}{\partial x^2}&=
 * D^{(\hat{2})}\left(x^3\frac{\partial x^1}{\partial \xi^{\hat{3}}}\right)
 * -D^{(\hat{2})}\left(x^1\frac{\partial x^3}{\partial \xi^{\hat{3}}}\right)
 * +D^{(\hat{3})}\left(x^1\frac{\partial x^3}{\partial \xi^{\hat{2}}}\right)
 * -D^{(\hat{3})}\left(x^3\frac{\partial x^1}{\partial \xi^{\hat{2}}}\right) \\
 * 2J\frac{\partial \xi^{\hat{2}}}{\partial x^2}&=
 * D^{(\hat{1})}\left(x^1\frac{\partial x^3}{\partial \xi^{\hat{3}}}\right)
 * -D^{(\hat{1})}\left(x^3\frac{\partial x^1}{\partial \xi^{\hat{3}}}\right)
 * +D^{(\hat{3})}\left(x^3\frac{\partial x^1}{\partial \xi^{\hat{1}}}\right)
 * -D^{(\hat{3})}\left(x^1\frac{\partial x^3}{\partial \xi^{\hat{1}}}\right)\\
 * 2J\frac{\partial \xi^{\hat{3}}}{\partial x^2}&=
 * D^{(\hat{1})}\left(x^3\frac{\partial x^1}{\partial \xi^{\hat{2}}}\right)
 * -D^{(\hat{1})}\left(x^1\frac{\partial x^3}{\partial \xi^{\hat{2}}}\right)
 * +D^{(\hat{2})}\left(x^1\frac{\partial x^3}{\partial \xi^{\hat{1}}}\right)
 * -D^{(\hat{2})}\left(x^3\frac{\partial x^1}{\partial \xi^{\hat{1}}}\right) \\
 * 2J\frac{\partial \xi^{\hat{1}}}{\partial x^3}&=
 * D^{(\hat{2})}\left(x^1\frac{\partial x^2}{\partial \xi^{\hat{3}}}\right)
 * -D^{(\hat{2})}\left(x^2\frac{\partial x^1}{\partial \xi^{\hat{3}}}\right)
 * +D^{(\hat{3})}\left(x^2\frac{\partial x^1}{\partial \xi^{\hat{2}}}\right)
 * -D^{(\hat{3})}\left(x^1\frac{\partial x^2}{\partial \xi^{\hat{2}}}\right) \\
 * 2J\frac{\partial \xi^{\hat{2}}}{\partial x^3}&=
 * D^{(\hat{1})}\left(x^2\frac{\partial x^1}{\partial \xi^{\hat{3}}}\right)
 * -D^{(\hat{1})}\left(x^1\frac{\partial x^2}{\partial \xi^{\hat{3}}}\right)
 * +D^{(\hat{3})}\left(x^1\frac{\partial x^2}{\partial \xi^{\hat{1}}}\right)
 * -D^{(\hat{3})}\left(x^2\frac{\partial x^1}{\partial \xi^{\hat{1}}}\right) \\
 * 2J\frac{\partial \xi^{\hat{3}}}{\partial x^3}&=
 * D^{(\hat{1})}\left(x^1\frac{\partial x^2}{\partial \xi^{\hat{2}}}\right)
 * -D^{(\hat{1})}\left(x^2\frac{\partial x^1}{\partial \xi^{\hat{2}}}\right)
 * +D^{(\hat{2})}\left(x^2\frac{\partial x^1}{\partial \xi^{\hat{1}}}\right)
 * -D^{(\hat{2})}\left(x^1\frac{\partial x^2}{\partial \xi^{\hat{1}}}\right)
 *
 * \f}
 *
 * Again, this ensures that the metric identities are satisfied discretely. That
 * is,
 *
 * \f{align*}{
 *
 * \partial_{\hat{\imath}}\left(J\frac{\partial\xi^{\hat{\imath}}}
 * {\partial x^i}\right)=0
 *
 * \f}
 *
 * numerically.
 *
 * The reason for calculating \f$J\partial\xi^{\hat{\imath}}/\partial x^i\f$ in
 * this manner is because in the weak form of DG (and most spectral-type methods
 * can be recast as DG) we effectively evaluate
 *
 * \f{align*}{
 *
 * \partial_{\hat{\imath}}\left(J\frac{\partial\xi^{\hat{\imath}}}
 * {\partial x^i} F^i\right),
 *
 * \f}
 *
 * which should be identically zero if \f$F^i\f$ is constant. This feature of a
 * scheme is referred to as free-stream preserving. Note that another way to
 * achieve free-stream preservation is to subtract off the metric identity
 * violations. That is,
 *
 * \f{align*}{
 *
 * \partial_{\hat{\imath}}\left(J\frac{\partial\xi^{\hat{\imath}}}
 * {\partial x^i} F^i\right) -
 * F^i \partial_{\hat{\imath}}\left(J\frac{\partial\xi^{\hat{\imath}}}
 * {\partial x^i}\right).
 *
 * \f}
 *
 * The subtraction technique is most commonly used in finite difference codes.
 */
template <size_t Dim>
void metric_identity_det_jac_times_inv_jac(
    gsl::not_null<
        InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>*>
        det_jac_times_inverse_jacobian,
    const Mesh<Dim>& mesh,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
    const Jacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>&
        jacobian) noexcept;

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Compute the Jacobian, inverse Jacobian, and determinant of the
 * Jacobian so that they satisfy the metric identities.
 *
 * Uses `dg::metric_identity_jacobian()` to compute the determinant of the
 * Jacobian times the inverse Jacobian. By taking the determinant of this
 * product, we can isolate \f$J\f$, the determinant of the Jacobian. In \f$d\f$
 * dimensions, we have:
 *
 * \f{align}{
 *
 *  \mathrm{det}\left(J\frac{\partial \xi^{\hat{\imath}}}{\partial x^i}\right)
 *  = J^{d-1}.
 *
 * \f}
 *
 * We assume the determinant of the Jacobian is positive, which means logical
 * and inertial coordinates have the same handedness. With this assumption, we
 * have
 *
 * \f{align}{
 *
 *  J = \sqrt[(d-1)]{\mathrm{det}\left(J\frac{\partial
 *      \xi^{\hat{\imath}}}{\partial x^i}\right)}
 *
 * \f}
 *
 * We can now compute the inverse Jacobian using:
 *
 * \f{align}{
 *
 * \frac{\partial \xi^{\hat{\imath}}}{\partial x^i}=
 *  \frac{1}{J}\left(J\frac{\partial \xi^{\hat{\imath}}}{\partial x^i}\right)
 *
 * \f}
 *
 * This guarantees that multiplying the determinant of the Jacobian by the
 * inverse Jacobian gives a result that satisfies the metric identities. We also
 * compute the Jacobian by inverting the inverse Jacobian, which guarantees they
 * are (numerical) inverses of each other.
 *
 * \warning on entry `jacobian` must be the Jacobian to use for computing the
 * determinant of the Jacobian times the inverse Jacobian so that it satisfies
 * the metric identities. The `jacobian` can be computed analytically or
 * numerically, either is fine. On output the `jacobian` is the inverse of the
 * inverse Jacobian that satisfies the metric identities.
 */
template <size_t Dim>
void metric_identity_jacobian_quantities(
    gsl::not_null<
        InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>*>
        det_jac_times_inverse_jacobian,
    gsl::not_null<
        InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>*>
        inverse_jacobian,
    gsl::not_null<Jacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>*>
        jacobian,
    gsl::not_null<Scalar<DataVector>*> det_jacobian, const Mesh<Dim>& mesh,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) noexcept;
}  // namespace dg
