// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/Linearize.hpp"

#include <functional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Mesh.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

template <size_t Dim>
DataVector linearize(const DataVector& u, const Mesh<Dim>& mesh) {
  const Matrix empty{};
  auto filter = make_array<Dim>(std::cref(empty));
  for (size_t d = 0; d < Dim; d++) {
    gsl::at(filter, d) =
        std::cref(Spectral::linear_filter_matrix(mesh.slice_through(d)));
  }
  return apply_matrices(filter, u, mesh.extents());
}

template DataVector linearize<1>(const DataVector&, const Mesh<1>&);
template DataVector linearize<2>(const DataVector&, const Mesh<2>&);
template DataVector linearize<3>(const DataVector&, const Mesh<3>&);

template <size_t Dim>
DataVector linearize(const DataVector& u, const Mesh<Dim>& mesh,
                     const size_t d) {
  const Matrix empty{};
  auto filter = make_array<Dim>(std::cref(empty));
  gsl::at(filter, d) =
      std::cref(Spectral::linear_filter_matrix(mesh.slice_through(d)));
  return apply_matrices(filter, u, mesh.extents());
}

template DataVector linearize<1>(const DataVector&, const Mesh<1>&,
                                 const size_t);
template DataVector linearize<2>(const DataVector&, const Mesh<2>&,
                                 const size_t);
template DataVector linearize<3>(const DataVector&, const Mesh<3>&,
                                 const size_t);
