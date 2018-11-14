// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"

#include <ostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "Domain/Mesh.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <>
double mean_value_on_boundary(const DataVector& f, const Mesh<1>& mesh,
                              size_t d, Side side) noexcept {
  ASSERT(d == 0, "d = " << d);
  return Side::Lower == side ? f[0] : f[mesh.extents(0) - 1];
}

double mean_value_on_boundary(
    const gsl::not_null<DataVector*> /*boundary_buffer*/, const DataVector& f,
    const Mesh<1>& mesh, size_t d, Side side) noexcept {
  return mean_value_on_boundary(f, mesh, d, side);
}

double mean_value_on_boundary(
    const gsl::not_null<DataVector*> /*boundary_buffer*/,
    const gsl::span<std::pair<size_t, size_t>> /*volume_and_slice_indices*/,
    const DataVector& f, const Mesh<1>& mesh, const size_t d,
    const Side side) noexcept {
  return mean_value_on_boundary(f, mesh, d, side);
}

template <size_t Dim>
double mean_value_on_boundary(const DataVector& f, const Mesh<Dim>& mesh,
                              size_t d, Side side) noexcept {
  const Mesh<Dim - 1> mesh_on_boundary = mesh.slice_away(d);
  DataVector f_on_boundary(mesh_on_boundary.number_of_grid_points());
  return mean_value_on_boundary(&f_on_boundary, f, mesh, d, side);
}

template <size_t Dim>
double mean_value_on_boundary(const gsl::not_null<DataVector*> boundary_buffer,
                              const DataVector& f, const Mesh<Dim>& mesh,
                              size_t d, Side side) noexcept {
  ASSERT(d < Dim, "d = " << d << ", Dim = " << Dim);
  const size_t N = mesh.extents(d);
  const Mesh<Dim - 1> mesh_on_boundary = mesh.slice_away(d);
  for (SliceIterator si(mesh.extents(), d, (Side::Lower == side ? 0 : N - 1));
       si; ++si) {
    (*boundary_buffer)[si.slice_offset()] = f[si.volume_offset()];
  }
  return definite_integral(*boundary_buffer, mesh_on_boundary) /
         two_to_the(Dim - 1);
}

template <size_t Dim>
double mean_value_on_boundary(
    const gsl::not_null<DataVector*> boundary_buffer,
    const gsl::span<std::pair<size_t, size_t>> volume_and_slice_indices,
    const DataVector& f, const Mesh<Dim>& mesh, const size_t d,
    const Side /*side*/) noexcept {
  ASSERT(d < Dim, "d = " << d << ", Dim = " << Dim);
  const Mesh<Dim - 1> mesh_on_boundary = mesh.slice_away(d);
  for (const auto& volume_and_slice_index : volume_and_slice_indices) {
    (*boundary_buffer)[volume_and_slice_index.second] =
        f[volume_and_slice_index.first];
  }
  return definite_integral(*boundary_buffer, mesh_on_boundary) /
         two_to_the(Dim - 1);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                             \
  template double mean_value_on_boundary(                                \
      const gsl::not_null<DataVector*>, const DataVector&,               \
      const Mesh<DIM(data)>&, size_t, Side) noexcept;                    \
  template double mean_value_on_boundary(                                \
      const DataVector&, const Mesh<DIM(data)>&, size_t, Side) noexcept; \
  template double mean_value_on_boundary(                                \
      const gsl::not_null<DataVector*>,                                  \
      const gsl::span<std::pair<size_t, size_t>>, const DataVector&,     \
      const Mesh<DIM(data)>&, const size_t, const Side) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef DIM
#undef INSTANTIATE
