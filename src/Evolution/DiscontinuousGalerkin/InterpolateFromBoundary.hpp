// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg {
namespace detail {
template <size_t Dim>
void interpolate_dt_terms_gauss_points_impl(
    gsl::not_null<double*> volume_dt_vars, size_t num_independent_components,
    const Mesh<Dim>& volume_mesh, size_t dimension, size_t num_boundary_pts,
    const gsl::span<const double>& dt_corrections,
    const DataVector& boundary_interpolation_term);
}  // namespace detail

/*!
 * \brief Interpolate the Bjorhus/time derivative corrections to the volume time
 * derivatives in the specified direction.
 *
 * The general interpolation term (for the \f$+\xi\f$-dimension) is:
 *
 * \f{align*}{
 *   \partial_t u_{\alpha\breve{\imath}\breve{\jmath}\breve{k}}=\cdots
 *   +\ell^{\mathrm{Gauss-Lobatto}}_{N}
 *    \left(\xi_{\breve{\imath}}^{\mathrm{Gauss}}\right)
 *    \partial_t u^{\mathrm{BC}}_{\alpha\breve{\jmath}\breve{k}},
 * \f}
 *
 * where \f$\breve{\imath}\f$, \f$\breve{\jmath}\f$, and \f$\breve{k}\f$ are
 * indices in the logical \f$\xi\f$, \f$\eta\f$, and \f$\zeta\f$ dimensions.
 * \f$\partial_t u^{\mathrm{BC}}\f$ is the time derivative correction, and
 * the  function Spectral::boundary_interpolation_term() is used to compute and
 * cache the terms from the lifting terms.
 */
template <size_t Dim, typename DtTagsList>
void interpolate_dt_terms_gauss_points(
    const gsl::not_null<Variables<DtTagsList>*> dt_vars,
    const Mesh<Dim>& volume_mesh, const Direction<Dim>& direction,
    const Variables<DtTagsList>& dt_corrections) {
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
  detail::interpolate_dt_terms_gauss_points_impl(
      make_not_null(dt_vars->data()), dt_vars->number_of_independent_components,
      volume_mesh, direction.dimension(), num_boundary_grid_points,
      gsl::make_span(dt_corrections.data(), dt_corrections.size()),
      direction.side() == Side::Upper
          ? Spectral::boundary_interpolation_term(volume_stripe_mesh).second
          : Spectral::boundary_interpolation_term(volume_stripe_mesh).first);
}
}  // namespace evolution::dg
