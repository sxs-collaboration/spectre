// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Reconstruction.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Evolution/DgSubcell/Matrices.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace evolution::dg::subcell::fd {
namespace detail {
template <size_t Dim>
void reconstruct_impl(
    gsl::span<double> dg_u,
    const gsl::span<const double> subcell_u_times_projected_det_jac,
    const Mesh<Dim>& dg_mesh, const Index<Dim>& subcell_extents) noexcept {
  const size_t number_of_components =
      dg_u.size() / dg_mesh.number_of_grid_points();
  const Matrix& recons_matrix = reconstruction_matrix(dg_mesh, subcell_extents);
  dgemm_<true>('N', 'N', recons_matrix.rows(), number_of_components,
               recons_matrix.columns(), 1.0, recons_matrix.data(),
               recons_matrix.rows(), subcell_u_times_projected_det_jac.data(),
               recons_matrix.columns(), 0.0, dg_u.data(), recons_matrix.rows());
}
}  // namespace detail

template <size_t Dim>
void reconstruct(const gsl::not_null<DataVector*> dg_u,
                 const DataVector& subcell_u_times_projected_det_jac,
                 const Mesh<Dim>& dg_mesh,
                 const Index<Dim>& subcell_extents) noexcept {
  dg_u->destructive_resize(dg_mesh.number_of_grid_points());
  detail::reconstruct_impl(
      gsl::span<double>{dg_u->data(), dg_u->size()},
      gsl::span<const double>{subcell_u_times_projected_det_jac.data(),
                              subcell_u_times_projected_det_jac.size()},
      dg_mesh, subcell_extents);
}

template <size_t Dim>
DataVector reconstruct(const DataVector& subcell_u_times_projected_det_jac,
                       const Mesh<Dim>& dg_mesh,
                       const Index<Dim>& subcell_extents) noexcept {
  DataVector dg_u{dg_mesh.number_of_grid_points()};
  reconstruct(&dg_u, subcell_u_times_projected_det_jac, dg_mesh,
              subcell_extents);
  return dg_u;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                               \
  template DataVector reconstruct(const DataVector&, const Mesh<DIM(data)>&, \
                                  const Index<DIM(data)>&) noexcept;         \
  template void reconstruct(gsl::not_null<DataVector*>, const DataVector&,   \
                            const Mesh<DIM(data)>&,                          \
                            const Index<DIM(data)>&) noexcept;               \
  template void detail::reconstruct_impl(                                    \
      gsl::span<double> dg_u, const gsl::span<const double>,                 \
      const Mesh<DIM(data)>& dg_mesh,                                        \
      const Index<DIM(data)>& subcell_extents) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell::fd
/// \endcond
