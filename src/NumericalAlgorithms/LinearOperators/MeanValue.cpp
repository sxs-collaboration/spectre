// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"

#include <ostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "Domain/Mesh.hpp"
#include "ErrorHandling/Assert.hpp"

template <>
double mean_value_on_boundary(const DataVector& f, const Mesh<1>& mesh,
                              size_t d, Side side) {
  ASSERT(d == 0, "d = " << d);
  return Side::Lower == side ? f[0] : f[mesh.extents(0) - 1];
}

template <size_t Dim>
double mean_value_on_boundary(const DataVector& f, const Mesh<Dim>& mesh,
                              size_t d, Side side) {
  ASSERT(d < Dim, "d = " << d << ", Dim = " << Dim);
  const size_t N = mesh.extents(d);
  const Mesh<Dim - 1> mesh_on_boundary = mesh.slice_away(d);
  DataVector f_on_boundary(mesh_on_boundary.number_of_grid_points());
  for (SliceIterator si(mesh.extents(), d, (Side::Lower == side ? 0 : N - 1));
       si; ++si) {
    f_on_boundary[si.slice_offset()] = f[si.volume_offset()];
  }
  return definite_integral(f_on_boundary, mesh_on_boundary) /
         two_to_the(Dim - 1);
}

template double mean_value_on_boundary(const DataVector&, const Mesh<2>&,
                                       size_t, Side);
template double mean_value_on_boundary(const DataVector&, const Mesh<3>&,
                                       size_t, Side);
