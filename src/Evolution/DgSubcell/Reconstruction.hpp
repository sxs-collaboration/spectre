// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <size_t>
class Index;
template <size_t>
class Mesh;
/// \endcond

namespace evolution::dg::subcell::fd {
namespace detail {
template <size_t Dim>
void reconstruct_impl(gsl::span<double> dg_u,
                      gsl::span<const double> subcell_u_times_projected_det_jac,
                      const Mesh<Dim>& dg_mesh,
                      const Index<Dim>& subcell_extents) noexcept;
}  // namespace detail

// @{
/*!
 * \ingroup DgSubcellGroup
 * \brief reconstruct the variable `subcell_u_times_projected_det_jac` onto the
 * DG grid `dg_mesh`.
 *
 * In general we wish that the reconstruction operator is the pseudo-inverse of
 * the projection operator. On curved meshes this means we either need to
 * compute a (time-dependent) reconstruction and projection matrix on each DG
 * element, or we expand the determinant of the Jacobian on the basis, accepting
 * the aliasing errors from that. We accept the aliasing errors in favor of the
 * significantly reduced computational overhead. This means that the projection
 * and reconstruction operators are only inverses of each other if both operate
 * on \f$u J\f$ where \f$u\f$ is the variable being projected and \f$J\f$ is the
 * determinant of the Jacobian. That is, the matrices are guaranteed to satisfy
 * \f$\mathcal{R}(\mathcal{P}(u J))=u J\f$. If the mesh is regular Cartesian,
 * then this isn't an issue. Furthermore, if we reconstruct
 * \f$uJ/\mathcal{P}(J)\f$ we again recover the exact DG solution. Doing the
 * latter has the advantage that, in general, we are ideally projecting to the
 * subcells much more often than reconstructing from them (a statement that we
 * would rather use DG more than the subcells).
 */
template <size_t Dim>
DataVector reconstruct(const DataVector& subcell_u_times_projected_det_jac,
                       const Mesh<Dim>& dg_mesh,
                       const Index<Dim>& subcell_extents) noexcept;

template <size_t Dim>
void reconstruct(gsl::not_null<DataVector*> dg_u,
                 const DataVector& subcell_u_times_projected_det_jac,
                 const Mesh<Dim>& dg_mesh,
                 const Index<Dim>& subcell_extents) noexcept;

template <typename SubcellTagList, typename DgTagList, size_t Dim>
void reconstruct(const gsl::not_null<Variables<DgTagList>*> dg_u,
                 const Variables<SubcellTagList>& subcell_u,
                 const Mesh<Dim>& dg_mesh,
                 const Index<Dim>& subcell_extents) noexcept {
  if (UNLIKELY(dg_u->number_of_grid_points() !=
               dg_mesh.number_of_grid_points())) {
    dg_u->initialize(dg_mesh.number_of_grid_points(), 0.0);
  }
  detail::reconstruct_impl(
      gsl::span<double>{dg_u->data(), dg_u->size()},
      gsl::span<const double>{subcell_u.data(), subcell_u.size()}, dg_mesh,
      subcell_extents);
}

template <typename TagList, size_t Dim>
Variables<TagList> reconstruct(const Variables<TagList>& subcell_u,
                               const Mesh<Dim>& dg_mesh,
                               const Index<Dim>& subcell_extents) noexcept {
  Variables<TagList> dg_u(dg_mesh.number_of_grid_points());
  reconstruct(make_not_null(&dg_u), subcell_u, dg_mesh, subcell_extents);
  return dg_u;
}
// @}
}  // namespace evolution::dg::subcell::fd
