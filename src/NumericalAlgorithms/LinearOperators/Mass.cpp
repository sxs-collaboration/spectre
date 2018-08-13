// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/Mass.hpp"

#include <ostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <size_t Dim>
DataVector mass(
    const DataVector& data,
    const Jacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>& jacobian,
    const Index<Dim>& mesh) noexcept {
  ASSERT(data.size() == mesh.product(),
         "size = " << data.size() << ", num_grid_points = " << mesh.product());
  // This can be vectorized when we have implemented functionality to apply
  // diagonal matrices to DataVectors.
  auto massive_data = data;
  for (IndexIterator<Dim> index(mesh); index; ++index) {
    for (size_t d = 0; d < Dim; d++) {
      massive_data[index.collapsed_index()] *=
          Basis::lgl::quadrature_weights(mesh[d])[index()[d]] *
          jacobian.get(d, d)[index.collapsed_index()];
    }
  }
  return massive_data;
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(_, data)                                                 \
  template DataVector mass<DIM(data)>(                                         \
      const DataVector&,                                                       \
      const Jacobian<DataVector, DIM(data), Frame::Logical, Frame::Inertial>&, \
      const Index<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
/// \endcond
