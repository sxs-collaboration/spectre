// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg {
namespace detail {
template <size_t Dim>
void lift_boundary_terms_gauss_points_impl(
    gsl::not_null<double*> volume_dt_vars, size_t num_independent_components,
    const Mesh<Dim>& volume_mesh, size_t dimension,
    const Scalar<DataVector>& volume_det_inv_jacobian, size_t num_boundary_pts,
    const gsl::span<const double>& boundary_corrections,
    const DataVector& boundary_lifting_term,
    const Scalar<DataVector>& magnitude_of_face_normal,
    const Scalar<DataVector>& face_det_jacobian);

template <size_t Dim>
void lift_boundary_terms_gauss_points_impl(
    gsl::not_null<double*> volume_dt_vars, size_t num_independent_components,
    const Mesh<Dim>& volume_mesh, size_t dimension,
    const Scalar<DataVector>& volume_det_inv_jacobian, size_t num_boundary_pts,
    const gsl::span<const double>& upper_boundary_corrections,
    const DataVector& upper_boundary_lifting_term,
    const Scalar<DataVector>& upper_magnitude_of_face_normal,
    const Scalar<DataVector>& upper_face_det_jacobian,
    const gsl::span<const double>& lower_boundary_corrections,
    const DataVector& lower_boundary_lifting_term,
    const Scalar<DataVector>& lower_magnitude_of_face_normal,
    const Scalar<DataVector>& lower_face_det_jacobian);
}  // namespace detail

/*!
 * \brief Lift the boundary corrections to the volume time derivatives in the
 * specified direction.
 *
 * The general lifting term (for the \f$\xi\f$-dimension) is:
 *
 * \f{align*}{
 * \partial_t u_{\alpha\breve{\imath}\breve{\jmath}\breve{k}}=\cdots
 * -\frac{\ell_{\breve{\imath}}\left(\xi=1\right)}
 *   {w_{\breve{\imath}}J_{\breve{\imath}\breve{\jmath}\breve{k}}}
 *   \left[J\sqrt{
 *   \frac{\partial\xi}{\partial x^i} \gamma^{ij}
 *   \frac{\partial\xi}{\partial x^j}}
 *   \left(G_{\alpha} + D_{\alpha}\right)
 *   \right]_{\breve{\jmath}\breve{k}}\left(\xi=1\right),
 * \f}
 *
 * where \f$\breve{\imath}\f$, \f$\breve{\jmath}\f$, and \f$\breve{k}\f$ are
 * indices in the logical \f$\xi\f$, \f$\eta\f$, and \f$\zeta\f$ dimensions.
 * The \f$G+D\f$ terms correspond to the `boundary_corrections` function
 * argument, and the function Spectral::boundary_lifting_term() is used to
 * compute and cache the terms from the lifting terms
 * \f$\ell_{\breve{\imath}}(\xi=\pm1)/w_{\breve{\imath}}\f$.
 *
 * \note that normal vectors are pointing out of the element.
 */
template <size_t Dim, typename DtTagsList, typename BoundaryCorrectionTagsList>
void lift_boundary_terms_gauss_points(
    const gsl::not_null<Variables<DtTagsList>*> dt_vars,
    const Scalar<DataVector>& volume_det_inv_jacobian,
    const Mesh<Dim>& volume_mesh, const Direction<Dim>& direction,
    const Variables<BoundaryCorrectionTagsList>& boundary_corrections,
    const Scalar<DataVector>& magnitude_of_face_normal,
    const Scalar<DataVector>& face_det_jacobian) {
  ASSERT(std::all_of(volume_mesh.quadrature().begin(),
                     volume_mesh.quadrature().end(),
                     [](const Spectral::Quadrature quadrature) {
                       return quadrature == Spectral::Quadrature::Gauss;
                     }),
         "Must use Gauss points in all directions but got the mesh: "
             << volume_mesh);
  const Mesh<Dim - 1> boundary_mesh =
      volume_mesh.slice_away(direction.dimension());
  const Mesh<1> volume_stripe_mesh =
      volume_mesh.slice_through(direction.dimension());
  const size_t num_boundary_grid_points = boundary_mesh.number_of_grid_points();
  detail::lift_boundary_terms_gauss_points_impl(
      make_not_null(dt_vars->data()), dt_vars->number_of_independent_components,
      volume_mesh, direction.dimension(), volume_det_inv_jacobian,
      num_boundary_grid_points,
      gsl::make_span(boundary_corrections.data(), boundary_corrections.size()),
      direction.side() == Side::Upper
          ? Spectral::boundary_lifting_term(volume_stripe_mesh).second
          : Spectral::boundary_lifting_term(volume_stripe_mesh).first,
      magnitude_of_face_normal, face_det_jacobian);
}

/*!
 * \brief Lift both the upper and lower (in logical coordinates) boundary
 * corrections to the volume time derivatives in the specified logical
 * dimension.
 *
 * The upper and lower boundary corrections in the logical `dimension` are
 * lifted together in order to reduce the amount of striding through data that
 * is needed and to improve cache-friendliness.
 *
 * The general lifting term (for the \f$\xi\f$-dimension) is:
 *
 * \f{align*}{
 * \partial_t u_{\alpha\breve{\imath}\breve{\jmath}\breve{k}}=\cdots
 * -\frac{\ell_{\breve{\imath}}\left(\xi=1\right)}
 *   {w_{\breve{\imath}}J_{\breve{\imath}\breve{\jmath}\breve{k}}}
 *   \left[J\sqrt{
 *   \frac{\partial\xi}{\partial x^i} \gamma^{ij}
 *   \frac{\partial\xi}{\partial x^j}}
 *   \left(G_{\alpha} + D_{\alpha}\right)
 *   \right]_{\breve{\jmath}\breve{k}}\left(\xi=1\right)
 * - \frac{\ell_{\breve{\imath}}\left(\xi=-1\right)}
 *   {w_{\breve{\imath}}J_{\breve{\imath}\breve{\jmath}\breve{k}}}
 *   \left[J\sqrt{
 *   \frac{\partial\xi}{\partial x^i} \gamma^{ij}
 *   \frac{\partial\xi}{\partial x^j}}
 *   \left(G_{\alpha} + D_{\alpha}\right)
 *   \right]_{\breve{\jmath}\breve{k}}\left(\xi=-1\right),
 * \f}
 *
 * where \f$\breve{\imath}\f$, \f$\breve{\jmath}\f$, and \f$\breve{k}\f$ are
 * indices in the logical \f$\xi\f$, \f$\eta\f$, and \f$\zeta\f$ dimensions.
 * The \f$G+D\f$ terms correspond to the `upper_boundary_corrections` and
 * `lower_boundary_corrections` function arguments, and the function
 * Spectral::boundary_lifting_term() is used to compute and cache the terms
 * from the lifting terms
 * \f$\ell_{\breve{\imath}}(\xi=\pm1)/w_{\breve{\imath}}\f$.
 *
 * \note that normal vectors are pointing out of the element and therefore both
 * terms have the same sign.
 */
template <size_t Dim, typename DtTagsList, typename BoundaryCorrectionTagsList>
void lift_boundary_terms_gauss_points(
    const gsl::not_null<Variables<DtTagsList>*> dt_vars,
    const Scalar<DataVector>& volume_det_inv_jacobian,
    const Mesh<Dim>& volume_mesh, const size_t dimension,
    const Variables<BoundaryCorrectionTagsList>& upper_boundary_corrections,
    const Scalar<DataVector>& upper_magnitude_of_face_normal,
    const Scalar<DataVector>& upper_face_det_jacobian,
    const Variables<BoundaryCorrectionTagsList>& lower_boundary_corrections,
    const Scalar<DataVector>& lower_magnitude_of_face_normal,
    const Scalar<DataVector>& lower_face_det_jacobian) {
  ASSERT(std::all_of(volume_mesh.quadrature().begin(),
                     volume_mesh.quadrature().end(),
                     [](const Spectral::Quadrature quadrature) {
                       return quadrature == Spectral::Quadrature::Gauss;
                     }),
         "Must use Gauss points in all directions but got the mesh: "
             << volume_mesh);
  const Mesh<Dim - 1> boundary_mesh = volume_mesh.slice_away(dimension);
  const Mesh<1> volume_stripe_mesh = volume_mesh.slice_through(dimension);
  const size_t num_boundary_grid_points = boundary_mesh.number_of_grid_points();
  detail::lift_boundary_terms_gauss_points_impl(
      make_not_null(dt_vars->data()), dt_vars->number_of_independent_components,
      volume_mesh, dimension, volume_det_inv_jacobian, num_boundary_grid_points,
      gsl::make_span(upper_boundary_corrections.data(),
                     upper_boundary_corrections.size()),
      Spectral::boundary_lifting_term(volume_stripe_mesh).second,
      upper_magnitude_of_face_normal, upper_face_det_jacobian,
      gsl::make_span(lower_boundary_corrections.data(),
                     lower_boundary_corrections.size()),
      Spectral::boundary_lifting_term(volume_stripe_mesh).first,
      lower_magnitude_of_face_normal, lower_face_det_jacobian);
}
}  // namespace evolution::dg
