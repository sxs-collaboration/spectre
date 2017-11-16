// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "ErrorHandling/Assert.hpp"

template <>
double mean_value_on_boundary(const DataVector& f, const Index<1>& extents,
                              size_t d, Side side) {
  ASSERT(d == 0, "d = " << d);
  return Side::Lower == side ? f[0] : f[extents[0] - 1];
}

template <size_t Dim>
double mean_value_on_boundary(const DataVector& f, const Index<Dim>& extents,
                              size_t d, Side side) {
  ASSERT(d < Dim, "d = " << d << ", Dim = " << Dim);
  const size_t N = extents[d];
  const Index<Dim - 1> extents_on_boundary = extents.slice_away(d);
  DataVector f_on_boundary(extents_on_boundary.product());
  for (SliceIterator si(extents, d, (Side::Lower == side ? 0 : N - 1)); si;
       ++si) {
    f_on_boundary[si.slice_offset()] = f[si.volume_offset()];
  }
  return Basis::lgl::definite_integral(f_on_boundary, extents_on_boundary) /
         two_to_the(Dim - 1);
}

template double mean_value_on_boundary(const DataVector&, const Index<2>&,
                                       size_t, Side);
template double mean_value_on_boundary(const DataVector&, const Index<3>&,
                                       size_t, Side);
