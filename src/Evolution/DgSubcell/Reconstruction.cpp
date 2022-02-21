// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Reconstruction.hpp"

#include <array>
#include <cstddef>
#include <functional>

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
#include "Utilities/MakeArray.hpp"

namespace evolution::dg::subcell::fd {
namespace detail {
template <size_t Dim>
void reconstruct_impl(
    gsl::span<double> dg_u,
    const gsl::span<const double> subcell_u_times_projected_det_jac,
    const Mesh<Dim>& dg_mesh, const Index<Dim>& subcell_extents,
    const ReconstructionMethod reconstruction_method) {
  const size_t number_of_components =
      dg_u.size() / dg_mesh.number_of_grid_points();
  if (reconstruction_method == ReconstructionMethod::AllDimsAtOnce) {
    const Matrix& recons_matrix =
        reconstruction_matrix(dg_mesh, subcell_extents);
    dgemm_<true>('N', 'N', recons_matrix.rows(), number_of_components,
                 recons_matrix.columns(), 1.0, recons_matrix.data(),
                 recons_matrix.rows(), subcell_u_times_projected_det_jac.data(),
                 recons_matrix.columns(), 0.0, dg_u.data(),
                 recons_matrix.rows());
  } else {
    ASSERT(reconstruction_method == ReconstructionMethod::DimByDim,
           "reconstruction_method must be either DimByDim or AllDimsAtOnce");
    // We multiply the last dim first because projection is done with the last
    // dim last. We do this because it ensures the R(P)==I up to machine
    // precision.
    const Matrix empty{};
    auto recons_matrices = make_array<Dim>(std::cref(empty));
    for (size_t d = 0; d < Dim; d++) {
      gsl::at(recons_matrices, d) = std::cref(reconstruction_matrix(
          dg_mesh.slice_through(d), Index<1>{subcell_extents[d]}));
    }
    DataVector result{dg_u.data(), dg_u.size()};
    const DataVector u{
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<double*>(subcell_u_times_projected_det_jac.data()),
        subcell_u_times_projected_det_jac.size()};
    apply_matrices(make_not_null(&result), recons_matrices, u, subcell_extents);
  }
}
}  // namespace detail

template <size_t Dim>
void reconstruct(const gsl::not_null<DataVector*> dg_u,
                 const DataVector& subcell_u_times_projected_det_jac,
                 const Mesh<Dim>& dg_mesh, const Index<Dim>& subcell_extents,
                 const ReconstructionMethod reconstruction_method) {
  ASSERT(subcell_u_times_projected_det_jac.size() == subcell_extents.product(),
         "Incorrect subcell size of u: "
             << subcell_u_times_projected_det_jac.size() << " but should be "
             << subcell_extents.product());
  dg_u->destructive_resize(dg_mesh.number_of_grid_points());
  detail::reconstruct_impl(
      gsl::span<double>{dg_u->data(), dg_u->size()},
      gsl::span<const double>{subcell_u_times_projected_det_jac.data(),
                              subcell_u_times_projected_det_jac.size()},
      dg_mesh, subcell_extents, reconstruction_method);
}

template <size_t Dim>
DataVector reconstruct(const DataVector& subcell_u_times_projected_det_jac,
                       const Mesh<Dim>& dg_mesh,
                       const Index<Dim>& subcell_extents,
                       const ReconstructionMethod reconstruction_method) {
  DataVector dg_u{dg_mesh.number_of_grid_points()};
  reconstruct(&dg_u, subcell_u_times_projected_det_jac, dg_mesh,
              subcell_extents, reconstruction_method);
  return dg_u;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template DataVector reconstruct(const DataVector&, const Mesh<DIM(data)>&,   \
                                  const Index<DIM(data)>&,                     \
                                  ReconstructionMethod);                       \
  template void reconstruct(gsl::not_null<DataVector*>, const DataVector&,     \
                            const Mesh<DIM(data)>&, const Index<DIM(data)>&,   \
                            ReconstructionMethod);                             \
  template void detail::reconstruct_impl(                                      \
      gsl::span<double> dg_u, const gsl::span<const double>,                   \
      const Mesh<DIM(data)>& dg_mesh, const Index<DIM(data)>& subcell_extents, \
      ReconstructionMethod);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell::fd
