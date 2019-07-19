// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IrregularInterpolant.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
// IWYU pragma: no_forward_declare Tensor

namespace {

template <size_t Dim>
Matrix interpolation_matrix(
    const Mesh<Dim>& mesh,
    const tnsr::I<DataVector, Dim, Frame::Logical>& points) noexcept;

template <>
Matrix interpolation_matrix(
    const Mesh<1>& mesh,
    const tnsr::I<DataVector, 1, Frame::Logical>& points) noexcept {
  // For 1d interpolation matrix, simply return the legendre-gauss-lobatto
  // matrix.
  return Spectral::interpolation_matrix(mesh, get<0>(points));
}

template <>
Matrix interpolation_matrix(
    const Mesh<2>& mesh,
    const tnsr::I<DataVector, 2, Frame::Logical>& points) noexcept {
  const std::array<Matrix, 2> matrices{
      {Spectral::interpolation_matrix(mesh.slice_through(0), get<0>(points)),
       Spectral::interpolation_matrix(mesh.slice_through(1), get<1>(points))}};

  const auto number_of_points = get<0>(points).size();
  Matrix result(number_of_points, mesh.number_of_grid_points());
  // First dimension of DataVector varies fastest.
  for (size_t j = 0, s = 0; j < mesh.extents(1); ++j) {
    for (size_t i = 0; i < mesh.extents(0); ++i) {
      for (size_t p = 0; p < number_of_points; ++p) {
        result(p, s) = matrices[0](p, i) * matrices[1](p, j);
      }
      ++s;
    }
  }
  return result;
}

template <>
Matrix interpolation_matrix(
    const Mesh<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::Logical>& points) noexcept {
  const std::array<Matrix, 3> matrices{
      {Spectral::interpolation_matrix(mesh.slice_through(0), get<0>(points)),
       Spectral::interpolation_matrix(mesh.slice_through(1), get<1>(points)),
       Spectral::interpolation_matrix(mesh.slice_through(2), get<2>(points))}};

  const auto number_of_points = get<0>(points).size();
  Matrix result(number_of_points, mesh.number_of_grid_points());
  // First dimension of DataVector varies fastest.
  for (size_t k = 0, s = 0; k < mesh.extents(2); ++k) {
    for (size_t j = 0; j < mesh.extents(1); ++j) {
      for (size_t i = 0; i < mesh.extents(0); ++i) {
        for (size_t p = 0; p < number_of_points; ++p) {
          result(p, s) =
              matrices[0](p, i) * matrices[1](p, j) * matrices[2](p, k);
        }
        ++s;
      }
    }
  }
  return result;
}
}  // namespace

namespace intrp {

template <size_t Dim>
Irregular<Dim>::Irregular() = default;

template <size_t Dim>
Irregular<Dim>::Irregular(
    const Mesh<Dim>& source_mesh,
    const tnsr::I<DataVector, Dim, Frame::Logical>& target_points) noexcept
    : interpolation_matrix_(interpolation_matrix(source_mesh, target_points)) {}

template <size_t Dim>
void Irregular<Dim>::pup(PUP::er& p) noexcept {
  p | interpolation_matrix_;
}

template <size_t Dim>
bool operator!=(const Irregular<Dim>& lhs, const Irregular<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

template class Irregular<1>;
template class Irregular<2>;
template class Irregular<3>;
template bool operator!=(const Irregular<1>& lhs,
                         const Irregular<1>& rhs) noexcept;
template bool operator!=(const Irregular<2>& lhs,
                         const Irregular<2>& rhs) noexcept;
template bool operator!=(const Irregular<3>& lhs,
                         const Irregular<3>& rhs) noexcept;

}  // namespace intrp
