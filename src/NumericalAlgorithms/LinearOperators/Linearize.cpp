// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/Linearize.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/StripeIterator.hpp"
#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/MakeArray.hpp"

template <size_t Dim>
DataVector linearize(const DataVector& u, const Index<Dim>& extents) {
  auto u_linearized = make_array<Dim>(DataVector(extents.product(), 0.0));

  for (size_t d = 0; d < Dim; ++d) {
    const Matrix& F = Basis::lgl::linear_filter_matrix(extents[d]);
    for (StripeIterator s(extents, d); s; ++s) {
      dgemv_('N', extents[d], extents[d], 1., F.data(), extents[d],
             0 == d ? &u[s.offset()]                              // NOLINT
                    : &gsl::at(u_linearized, d - 1)[s.offset()],  // NOLINT
             s.stride(), 0.0,
             &gsl::at(u_linearized, d).data()[s.offset()],  // NOLINT
             s.stride());
    }
  }
  return u_linearized[Dim - 1];
}

template DataVector linearize<1>(const DataVector&, const Index<1>&);
template DataVector linearize<2>(const DataVector&, const Index<2>&);
template DataVector linearize<3>(const DataVector&, const Index<3>&);

template <size_t Dim>
DataVector linearize(const DataVector& u, const Index<Dim>& extents,
                     const size_t d) {
  DataVector u_linearized(extents.product());

  const Matrix& F = Basis::lgl::linear_filter_matrix(extents[d]);
  for (StripeIterator s(extents, d); s; ++s) {
    dgemv_('N', extents[d], extents[d], 1., F.data(), extents[d],
           &u.data()[s.offset()],                              // NOLINT
           s.stride(), 0.0, &u_linearized.data()[s.offset()],  // NOLINT
           s.stride());
  }
  return u_linearized;
}

template DataVector linearize<1>(const DataVector&, const Index<1>&,
                                 const size_t);
template DataVector linearize<2>(const DataVector&, const Index<2>&,
                                 const size_t);
template DataVector linearize<3>(const DataVector&, const Index<3>&,
                                 const size_t);
