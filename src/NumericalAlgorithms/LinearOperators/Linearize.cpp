// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/Linearize.hpp"

#include <functional>
#include <ostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Mesh.hpp"  // IWYU pragma: keep
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

template <size_t Dim>
void linearize(const gsl::not_null<DataVector*> result, const DataVector& u,
               const Mesh<Dim>& mesh) noexcept {
  ASSERT(result->size() == u.size(),
         "The size of result passed to linearize must be equal to the size of "
         "u. result.size(): "
             << result->size() << " u.size(): " << u.size());
  const Matrix empty{};
  auto filter = make_array<Dim>(std::cref(empty));
  for (size_t d = 0; d < Dim; d++) {
    gsl::at(filter, d) =
        std::cref(Spectral::linear_filter_matrix(mesh.slice_through(d)));
  }
  apply_matrices(result, filter, u, mesh.extents());
}

template <size_t Dim>
DataVector linearize(const DataVector& u, const Mesh<Dim>& mesh) noexcept {
  DataVector result(mesh.number_of_grid_points());
  linearize(&result, u, mesh);
  return result;
}

template <size_t Dim>
void linearize(const gsl::not_null<DataVector*> result, const DataVector& u,
               const Mesh<Dim>& mesh, const size_t d) noexcept {
  ASSERT(result->size() == u.size(),
         "The size of result passed to linearize must be equal to the size of "
         "u. result.size(): "
             << result->size() << " u.size(): " << u.size());
  const Matrix empty{};
  auto filter = make_array<Dim>(std::cref(empty));
  gsl::at(filter, d) =
      std::cref(Spectral::linear_filter_matrix(mesh.slice_through(d)));
  apply_matrices(result, filter, u, mesh.extents());
}

template <size_t Dim>
DataVector linearize(const DataVector& u, const Mesh<Dim>& mesh,
                     const size_t d) noexcept {
  DataVector result(mesh.number_of_grid_points());
  linearize(&result, u, mesh, d);
  return result;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                                 \
  template void linearize<GET_DIM(data)>(const gsl::not_null<DataVector*>,     \
                                         const DataVector&,                    \
                                         const Mesh<GET_DIM(data)>&) noexcept; \
  template DataVector linearize<GET_DIM(data)>(                                \
      const DataVector&, const Mesh<GET_DIM(data)>&) noexcept;                 \
  template void linearize<GET_DIM(data)>(                                      \
      const gsl::not_null<DataVector*>, const DataVector&,                     \
      const Mesh<GET_DIM(data)>&, const size_t);                               \
  template DataVector linearize<GET_DIM(data)>(                                \
      const DataVector&, const Mesh<GET_DIM(data)>&, const size_t);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
