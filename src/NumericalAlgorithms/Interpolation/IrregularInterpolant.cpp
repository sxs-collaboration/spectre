// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IrregularInterpolant.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"
// IWYU pragma: no_forward_declare Tensor

namespace {

template <size_t Dim>
Matrix interpolation_matrix(
    const Index<Dim>& mesh,
    const tnsr::I<DataVector, Dim, Frame::Logical>& points) noexcept;

template <>
Matrix interpolation_matrix(
    const Index<1>& mesh,
    const tnsr::I<DataVector, 1, Frame::Logical>& points) noexcept {
  // For 1d interpolation matrix, simply return the legendre-gauss-lobatto
  // matrix.
  return Basis::lgl::interpolation_matrix(mesh[0], get<0>(points));
}

template <>
Matrix interpolation_matrix(
    const Index<2>& mesh,
    const tnsr::I<DataVector, 2, Frame::Logical>& points) noexcept {
  const std::array<Matrix, 2> matrices{
      {Basis::lgl::interpolation_matrix(mesh[0], get<0>(points)),
       Basis::lgl::interpolation_matrix(mesh[1], get<1>(points))}};

  const auto number_of_points = get<0>(points).size();
  Matrix result(number_of_points, mesh.product());
  // First dimension of DataVector varies fastest.
  for (size_t j = 0, s = 0; j < mesh[1]; ++j) {
    for (size_t i = 0; i < mesh[0]; ++i) {
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
    const Index<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::Logical>& points) noexcept {
  const std::array<Matrix, 3> matrices{
      {Basis::lgl::interpolation_matrix(mesh[0], get<0>(points)),
       Basis::lgl::interpolation_matrix(mesh[1], get<1>(points)),
       Basis::lgl::interpolation_matrix(mesh[2], get<2>(points))}};

  const auto number_of_points = get<0>(points).size();
  Matrix result(number_of_points, mesh.product());
  // First dimension of DataVector varies fastest.
  for (size_t k = 0, s = 0; k < mesh[2]; ++k) {
    for (size_t j = 0; j < mesh[1]; ++j) {
      for (size_t i = 0; i < mesh[0]; ++i) {
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
    const Index<Dim>& source_mesh,
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
