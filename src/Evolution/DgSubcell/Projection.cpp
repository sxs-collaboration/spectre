// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Projection.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Evolution/DgSubcell/Matrices.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell::fd {
namespace detail {
template <size_t Dim>
void project_impl(gsl::span<double> subcell_u,
                  const gsl::span<const double> dg_u, const Mesh<Dim>& dg_mesh,
                  const Index<Dim>& subcell_extents) noexcept {
  const size_t number_of_components =
      dg_u.size() / dg_mesh.number_of_grid_points();
  const Matrix& proj_matrix = projection_matrix(dg_mesh, subcell_extents);
  dgemm_<true>('N', 'N', proj_matrix.rows(), number_of_components,
               proj_matrix.columns(), 1.0, proj_matrix.data(),
               proj_matrix.rows(), dg_u.data(), proj_matrix.columns(), 0.0,
               subcell_u.data(), proj_matrix.rows());
}
}  // namespace detail

template <size_t Dim>
void project(const gsl::not_null<DataVector*> subcell_u, const DataVector& dg_u,
             const Mesh<Dim>& dg_mesh,
             const Index<Dim>& subcell_extents) noexcept {
  subcell_u->destructive_resize(subcell_extents.product());
  detail::project_impl(gsl::span<double>{subcell_u->data(), subcell_u->size()},
                       gsl::span<const double>{dg_u.data(), dg_u.size()},
                       dg_mesh, subcell_extents);
}

template <size_t Dim>
DataVector project(const DataVector& dg_u, const Mesh<Dim>& dg_mesh,
                   const Index<Dim>& subcell_extents) noexcept {
  DataVector subcell_u{subcell_extents.product()};
  project(&subcell_u, dg_u, dg_mesh, subcell_extents);
  return subcell_u;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                           \
  template DataVector project(const DataVector&, const Mesh<DIM(data)>&, \
                              const Index<DIM(data)>&) noexcept;         \
  template void project(gsl::not_null<DataVector*>, const DataVector&,   \
                        const Mesh<DIM(data)>&,                          \
                        const Index<DIM(data)>&) noexcept;               \
  template void detail::project_impl(                                    \
      gsl::span<double> subcell_u, const gsl::span<const double> dg_u,   \
      const Mesh<DIM(data)>& dg_mesh,                                    \
      const Index<DIM(data)>& subcell_extents) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell::fd
