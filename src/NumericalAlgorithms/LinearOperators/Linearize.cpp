// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/Linearize.hpp"

#include <functional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"  // IWYU pragma: keep
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdArrayHelpers.hpp"

template <size_t Dim>
DataVector linearize(const DataVector& u, const Index<Dim>& extents) {
  const auto linear_filter_matrices =
      map_array(extents.indices(), [](const size_t extent) noexcept {
        return std::cref(Basis::lgl::linear_filter_matrix(extent));
      });
  return apply_matrices(linear_filter_matrices, u, extents);
}

template DataVector linearize<1>(const DataVector&, const Index<1>&);
template DataVector linearize<2>(const DataVector&, const Index<2>&);
template DataVector linearize<3>(const DataVector&, const Index<3>&);

template <size_t Dim>
DataVector linearize(const DataVector& u, const Index<Dim>& extents,
                     const size_t d) {
  const Matrix empty{};
  auto filter = make_array<Dim>(std::cref(empty));
  gsl::at(filter, d) = std::cref(Basis::lgl::linear_filter_matrix(extents[d]));
  return apply_matrices(filter, u, extents);
}

template DataVector linearize<1>(const DataVector&, const Index<1>&,
                                 const size_t);
template DataVector linearize<2>(const DataVector&, const Index<2>&,
                                 const size_t);
template DataVector linearize<3>(const DataVector&, const Index<3>&,
                                 const size_t);
