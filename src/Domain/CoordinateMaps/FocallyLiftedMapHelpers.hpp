// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// Holds helper functions for use with
/// domain::CoordinateMaps::FocallyLiftedMap.
namespace domain::CoordinateMaps::FocallyLiftedMapHelpers {

/*!
 * \brief Finds how long to extend a line segment to have it intersect
 * a point on a 2-sphere.
 *
 * \details Consider a 2-sphere with center \f$C^i\f$ and radius \f$R\f$, and
 * and let \f$P^i\f$ and \f$x_0^i\f$ be two arbitrary 3D points.
 *
 * Consider the line passing through \f$P^i\f$ and \f$x_0^i\f$.
 * If this line intersects the sphere at a point \f$x_1^i\f$, then we can write
 *
 * \f[
 *  x_1^i = P^i + (x_0^i-P^i) \lambda,
 * \f]
 *
 * where \f$\lambda\f$ is a scale factor.
 *
 * `scale_factor` computes and returns \f$\lambda\f$.
 *
 * ### Even more detail:
 *
 * To solve for \f$\lambda\f$, we note that \f$x_1^i\f$ is on the surface of
 * the sphere, so
 *
 * \f[
 *  |x_1^i-C^i|^2 = R^2,
 * \f]
 *
 * (where \f$|A^i|^2\f$ means \f$\delta_{ij} A^i A^j\f$),
 *
 *  or equivalently
 *
 * \f[
 *  | P^i-C^i + (x_0^i-P^i)\lambda |^2 = R^2.
 * \f]
 *
 * This is a quadratic equation for \f$\lambda\f$
 * and it generally has more than one real root.
 * It takes the usual form \f$a\lambda^2+b\lambda+c=0\f$,
 * with
 *
 * \f{align*}
 *  a &= |x_0^i-P^i|^2,\\
 *  b &= 2(x_0^i-P^i)(P^j-C^j)\delta_{ij},\\
 *  c &= |P^i-C^i|^2 - R^2,
 * \f}
 *
 * So how do we choose between multiple roots?  Some of the maps that
 * use `scale_factor` assume that *for all points*, \f$x_0^i\f$ is
 * between \f$P^i\f$ and \f$x_1^i\f$.  Those maps should set the parameter
 * `src_is_between_proj_and_target` to true. Other maps assume that
 * *for all points*, \f$x^i\f$ is always between \f$x_0^i\f$
 * and \f$P^i\f$. Those maps should set the parameter
 * `src_is_between_proj_and_target` to false.
 *
 * \warning If we ever add maps where
 * `src_is_between_proj_and_target` can change from point to point,
 * the logic of `scale_factor` needs to be changed.
 *
 * In the arguments to the function below, `src_point` is  \f$x_0^i\f$,
 * `proj_center` is \f$P^i\f$, `sphere_center` is \f$C^i\f$, and
 * `radius` is \f$R\f$.
 *
 */
template <typename T>
void scale_factor(const gsl::not_null<tt::remove_cvref_wrap_t<T>*>& result,
                  const std::array<T, 3>& src_point,
                  const std::array<double, 3>& proj_center,
                  const std::array<double, 3>& sphere_center, double radius,
                  bool src_is_between_proj_and_target);

/*!
 *  Solves a problem of the same form as `scale_factor`, but is used
 *  only by the inverse function to compute \f$\tilde{\lambda}\f$ and
 *  \f$\bar{\lambda}\f$. `try_scale_factor` is used in two contexts:
 *
 *  `try_scale_factor` is used to determine \f$\bar{\lambda}\f$
 *  given \f$x^i\f$. \f$\bar{\lambda}\f$ is defined by
 *  \f{align*} x_1^i = P^i + (x^i - P^i) \bar{\lambda}.\f}
 *
 *  `try_scale_factor` is used by the `lambda_tilde` functions of some
 *  of the `InnerMap` classes (namely those `InnerMap` classes where
 *  \f$x_0^i\f$ is a spherical surface) to solve for
 *  \f$\tilde{\lambda}\f$ given\f$x^i\f$.  \f$\tilde{\lambda}\f$
 *  is defined by
 *  \f{align*} x_0^i = P^i + (x^i - P^i) \tilde{\lambda}.\f}
 *
 *  In both of these contexts, the input parameter `src_point` is
 *  \f$x^i\f$, a point that is supposed to be in the range of the
 *  `FocallyLiftedMap`. Because the inverse function can be and is
 *  called for an arbitrary \f$x^i\f$ that might not be in the range
 *  of the `FocallyLiftedMap`, `try_scale_factor` returns a
 *  std::optional, with a default-constructed std::optional if the roots it
 *  finds are not as expected (i.e. if the inverse map was called for
 *  a point not in the range of the map).
 *
 *  Because `try_scale_factor` can be called in different situations,
 *  it has additional boolean arguments `pick_larger_root` and
 *  `pick_root_greater_than_one` that allow the caller to choose which
 *  root to return.
 *
 *  Furthermore, to reduce roundoff errors near
 *  \f$\tilde{\lambda}=1\f$, the default behavior is to solve the
 *  quadratic equation for \f$\tilde{\lambda}-1\f$ (and then add
 *  \f$1\f$ to the solution). If instead one wants to solve the
 *  quadratic equation directly for \f$\tilde{\lambda}\f$ so as to
 *  obtain slightly different roundoff behavior, then one should
 *  specify the argument `solve_for_root_minus_one` to be `false`.
 *
 * `try_scale_factor` is not templated
 *  on type because it is used only by the inverse function, which
 *  works only on doubles.
 *
 */
std::optional<double> try_scale_factor(
    const std::array<double, 3>& src_point,
    const std::array<double, 3>& proj_center,
    const std::array<double, 3>& sphere_center, double radius,
    bool pick_larger_root, bool pick_root_greater_than_one,
    bool solve_for_root_minus_one = true);

/*!
 * Computes \f$\partial \lambda/\partial x_0^i\f$, where \f$\lambda\f$
 * is the quantity returned by `scale_factor` and `x_0` is `src_point` in
 * the `scale_factor` function.
 *
 * The formula (see `FocallyLiftedMap`) is
 * \f{align*}
 * \frac{\partial\lambda}{\partial x_0^j} &=
 * \lambda^2 \frac{C_j - x_1^j}{|x_1^i - P^i|^2
 * + (x_1^i - P^i)(P_i - C_i)}.
 * \f}
 *
 * Note that it takes `intersection_point` and not `src_point` as a
 * parameter.
 *
 * In the arguments to the function below, `intersection_point` is \f$x_1\f$,
 * `proj_center` is \f$P^i\f$, `sphere_center` is \f$C^i\f$, and
 * `radius` is \f$R\f$.
 */
template <typename T>
void d_scale_factor_d_src_point(
    const gsl::not_null<std::array<tt::remove_cvref_wrap_t<T>, 3>*>& result,
    const std::array<T, 3>& intersection_point,
    const std::array<double, 3>& proj_center,
    const std::array<double, 3>& sphere_center, const T& lambda);

}  // namespace domain::CoordinateMaps::FocallyLiftedMapHelpers
