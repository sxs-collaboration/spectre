// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Projection.hpp"

#include <cstddef>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Evolution/DgSubcell/Matrices.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell::fd {
namespace detail {
template <size_t Dim>
void project_impl(gsl::span<double> subcell_u,
                  const gsl::span<const double> dg_u, const Mesh<Dim>& dg_mesh,
                  const Index<Dim>& subcell_extents) {
  const Matrix empty{};
  auto projection_mat = make_array<Dim>(std::cref(empty));
  for (size_t d = 0; d < Dim; d++) {
    gsl::at(projection_mat, d) = std::cref(
        projection_matrix(dg_mesh.slice_through(d), subcell_extents[d],
                          Spectral::Quadrature::CellCentered));
  }
  DataVector result{subcell_u.data(), subcell_u.size()};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  const DataVector u{const_cast<double*>(dg_u.data()), dg_u.size()};
  apply_matrices(make_not_null(&result), projection_mat, u, dg_mesh.extents());
}

// Project to a face of a Cartesian grid. We reconstruct on a FaceCentered basis
// in the dimension of the desired face, and on a CellCentered basis in other
// dimensions.
template <size_t Dim>
void project_to_face_impl(gsl::span<double> subcell_u,
                          const gsl::span<const double> dg_u,
                          const Mesh<Dim>& dg_mesh,
                          const Index<Dim>& subcell_extents,
                          const size_t& face_direction) {
  const Matrix empty{};
  auto projection_mat = make_array<Dim>(std::cref(empty));
  for (size_t d = 0; d < Dim; d++) {
    if (d == face_direction) {
      gsl::at(projection_mat, d) = std::cref(
          projection_matrix(dg_mesh.slice_through(d), subcell_extents[d],
                            Spectral::Quadrature::FaceCentered));
    } else {
      gsl::at(projection_mat, d) = std::cref(
          projection_matrix(dg_mesh.slice_through(d), subcell_extents[d],
                            Spectral::Quadrature::CellCentered));
    }
  }
  DataVector result{subcell_u.data(), subcell_u.size()};
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  const DataVector u{const_cast<double*>(dg_u.data()), dg_u.size()};
  apply_matrices(make_not_null(&result), projection_mat, u, dg_mesh.extents());
}
}  // namespace detail

template <size_t Dim>
void project(const gsl::not_null<DataVector*> subcell_u, const DataVector& dg_u,
             const Mesh<Dim>& dg_mesh, const Index<Dim>& subcell_extents) {
  ASSERT(dg_u.size() % dg_mesh.number_of_grid_points() == 0,
         "The vector dg_u must have size that is a multiple of the number of "
         "grid points "
             << dg_mesh.number_of_grid_points() << " but got " << dg_u.size());
  subcell_u->destructive_resize(subcell_extents.product() * dg_u.size() /
                                dg_mesh.number_of_grid_points());
  detail::project_impl(gsl::span<double>{subcell_u->data(), subcell_u->size()},
                       gsl::span<const double>{dg_u.data(), dg_u.size()},
                       dg_mesh, subcell_extents);
}

template <size_t Dim>
DataVector project(const DataVector& dg_u, const Mesh<Dim>& dg_mesh,
                   const Index<Dim>& subcell_extents) {
  ASSERT(dg_u.size() % dg_mesh.number_of_grid_points() == 0,
         "The vector dg_u must have size that is a multiple of the number of "
         "grid points "
             << dg_mesh.number_of_grid_points() << " but got " << dg_u.size());
  DataVector subcell_u{};
  project(&subcell_u, dg_u, dg_mesh, subcell_extents);
  return subcell_u;
}

template <size_t Dim>
void project_to_face(const gsl::not_null<DataVector*> subcell_u,
                     const DataVector& dg_u, const Mesh<Dim>& dg_mesh,
                     const Index<Dim>& subcell_extents,
                     const size_t& face_direction) {
  ASSERT(dg_u.size() % dg_mesh.number_of_grid_points() == 0,
         "The vector dg_u must have size that is a multiple of the number of "
         "grid points "
             << dg_mesh.number_of_grid_points() << " but got " << dg_u.size());
  subcell_u->destructive_resize(subcell_extents.product() * dg_u.size() /
                                dg_mesh.number_of_grid_points());
  detail::project_to_face_impl(
      gsl::span<double>{subcell_u->data(), subcell_u->size()},
      gsl::span<const double>{dg_u.data(), dg_u.size()}, dg_mesh,
      subcell_extents, face_direction);
}

template <size_t Dim>
DataVector project_to_face(const DataVector& dg_u, const Mesh<Dim>& dg_mesh,
                           const Index<Dim>& subcell_extents,
                           const size_t& face_direction) {
  ASSERT(dg_u.size() % dg_mesh.number_of_grid_points() == 0,
         "The vector dg_u must have size that is a multiple of the number of "
         "grid points "
             << dg_mesh.number_of_grid_points() << " but got " << dg_u.size());
  DataVector subcell_u{};
  project_to_face(&subcell_u, dg_u, dg_mesh, subcell_extents, face_direction);
  return subcell_u;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template DataVector project(const DataVector&, const Mesh<DIM(data)>&,       \
                              const Index<DIM(data)>&);                        \
  template void project(gsl::not_null<DataVector*>, const DataVector&,         \
                        const Mesh<DIM(data)>&, const Index<DIM(data)>&);      \
  template void detail::project_impl(gsl::span<double> subcell_u,              \
                                     const gsl::span<const double> dg_u,       \
                                     const Mesh<DIM(data)>& dg_mesh,           \
                                     const Index<DIM(data)>& subcell_extents); \
  template DataVector project_to_face(const DataVector&,                       \
                                      const Mesh<DIM(data)>&,                  \
                                      const Index<DIM(data)>&, const size_t&); \
  template void project_to_face(gsl::not_null<DataVector*>, const DataVector&, \
                                const Mesh<DIM(data)>&,                        \
                                const Index<DIM(data)>&, const size_t&);       \
  template void detail::project_to_face_impl(                                  \
      gsl::span<double> subcell_u, const gsl::span<const double> dg_u,         \
      const Mesh<DIM(data)>& dg_mesh, const Index<DIM(data)>& subcell_extents, \
      const size_t& face_direction);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell::fd
