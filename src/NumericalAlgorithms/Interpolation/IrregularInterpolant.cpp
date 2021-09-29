// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IrregularInterpolant.hpp"

#include <algorithm>
#include <array>
#include <iterator>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
// IWYU pragma: no_forward_declare Tensor

namespace {

// Just linear for now, can be extended to higher order...
// Future optimization: it might be more efficient to use Blaze's sparse
// matrices since the interpolation matrix is mostly zeros
std::vector<double> fd_stencil(const DataVector& xi_source,
                               const double xi_target) noexcept {
  ASSERT(std::is_sorted(std::begin(xi_source), std::end(xi_source)),
         "xi_source = " << xi_source);
  auto xi_u =
      std::upper_bound(std::begin(xi_source), std::end(xi_source), xi_target);
  if (std::end(xi_source) == xi_u) {
    std::advance(xi_u, -1);
  }
  if (std::begin(xi_source) == xi_u) {
    std::advance(xi_u, 1);
  }
  const auto xi_l = std::prev(xi_u);
  auto index = std::distance(std::begin(xi_source), xi_l);
  std::vector<double> result(xi_source.size(), 0.0);
  const auto result_l = std::next(std::begin(result), index);
  const auto result_u = std::next(result_l, 1);
  *result_l = (*xi_u - xi_target) / (*xi_u - *xi_l);
  *result_u = (xi_target - *xi_l) / (*xi_u - *xi_l);
  return result;
}

template <size_t Dim>
Matrix interpolation_matrix(
    const Mesh<Dim>& mesh,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& points) noexcept;

template <>
Matrix interpolation_matrix(
    const Mesh<1>& mesh,
    const tnsr::I<DataVector, 1, Frame::ElementLogical>& points) noexcept {
  if (mesh.basis()[0] == Spectral::Basis::FiniteDifference) {
    auto source_xi = logical_coordinates(mesh);
    const auto number_of_source_points = mesh.number_of_grid_points();
    const DataVector xi_source(get<0>(source_xi).data(),
                               number_of_source_points);
    const auto number_of_target_points = get<0>(points).size();
    Matrix result(number_of_target_points, mesh.number_of_grid_points());
    for (size_t p = 0; p < number_of_target_points; ++p) {
      const double xi_target = get<0>(points)[p];
      const auto stencil = fd_stencil(xi_source, xi_target);
      for (size_t i = 0; i < number_of_source_points; ++i) {
        result(p, i) = stencil[i];
      }
    }
    return result;
  }

  // Not FD, so use spectral interpolation
  return Spectral::interpolation_matrix(mesh, get<0>(points));
}

template <>
Matrix interpolation_matrix(
    const Mesh<2>& mesh,
    const tnsr::I<DataVector, 2, Frame::ElementLogical>& points) noexcept {
  const auto number_of_target_points = get<0>(points).size();
  Matrix result(number_of_target_points, mesh.number_of_grid_points());

  if (mesh.basis()[0] == Spectral::Basis::FiniteDifference) {
    ASSERT(mesh.basis()[1] == Spectral::Basis::FiniteDifference,
           "Mixed FD and DG bases are not supported. Mesh = " << mesh);
    auto source_xi = logical_coordinates(mesh);
    DataVector xi_source{get<0>(source_xi).data(), mesh.extents(0)};
    DataVector eta_source;
    if (mesh.extents(1) == mesh.extents(0)) {
      eta_source.set_data_ref(&xi_source);
    } else {
      eta_source.destructive_resize(mesh.extents(1));
      for (size_t j = 0; j < mesh.extents(1); ++j) {
        eta_source[j] = get<1>(source_xi)[j * mesh.extents(0)];
      }
    }
    for (size_t p = 0; p < number_of_target_points; ++p) {
      const double xi_target = get<0>(points)[p];
      const double eta_target = get<1>(points)[p];
      const auto xi_stencil = fd_stencil(xi_source, xi_target);
      const auto eta_stencil = fd_stencil(eta_source, eta_target);
      for (size_t j = 0, s = 0; j < mesh.extents(1); ++j) {
        for (size_t i = 0; i < mesh.extents(0); ++i) {
          result(p, s) = xi_stencil[i] * eta_stencil[j];
          ++s;
        }
      }
    }
    return result;
  }

  // Not FD, so use spectral interpolation
  const std::array<Matrix, 2> matrices{
      {Spectral::interpolation_matrix(mesh.slice_through(0), get<0>(points)),
       Spectral::interpolation_matrix(mesh.slice_through(1), get<1>(points))}};

  // First dimension of DataVector varies fastest.
  for (size_t j = 0, s = 0; j < mesh.extents(1); ++j) {
    for (size_t i = 0; i < mesh.extents(0); ++i) {
      for (size_t p = 0; p < number_of_target_points; ++p) {
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
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& points) noexcept {
  const auto number_of_target_points = get<0>(points).size();
  Matrix result(number_of_target_points, mesh.number_of_grid_points());

  if (mesh.basis()[0] == Spectral::Basis::FiniteDifference) {
    ASSERT(mesh.basis()[1] == Spectral::Basis::FiniteDifference and
               mesh.basis()[2] == Spectral::Basis::FiniteDifference,
           "Mixed FD and DG bases are not supported. Mesh = " << mesh);
    auto source_xi = logical_coordinates(mesh);
    DataVector xi_source{get<0>(source_xi).data(), mesh.extents(0)};
    DataVector eta_source;
    if (mesh.extents(1) == mesh.extents(0)) {
      eta_source.set_data_ref(&xi_source);
    } else {
      eta_source.destructive_resize(mesh.extents(1));
      for (size_t j = 0; j < mesh.extents(1); ++j) {
        eta_source[j] = get<1>(source_xi)[j * mesh.extents(0)];
      }
    }
    DataVector zeta_source;
    if (mesh.extents(2) == mesh.extents(0)) {
      zeta_source.set_data_ref(&xi_source);
    } else if (mesh.extents(2) == mesh.extents(1)) {
      zeta_source.set_data_ref(&eta_source);
    } else {
      zeta_source.destructive_resize(mesh.extents(2));
      for (size_t k = 0; k < mesh.extents(2); ++k) {
        zeta_source[k] =
            get<2>(source_xi)[k * mesh.extents(0) * mesh.extents(1)];
      }
    }
    for (size_t p = 0; p < number_of_target_points; ++p) {
      const double xi_target = get<0>(points)[p];
      const double eta_target = get<1>(points)[p];
      const double zeta_target = get<2>(points)[p];
      const auto xi_stencil = fd_stencil(xi_source, xi_target);
      const auto eta_stencil = fd_stencil(eta_source, eta_target);
      const auto zeta_stencil = fd_stencil(zeta_source, zeta_target);
      for (size_t k = 0, s = 0; k < mesh.extents(2); ++k) {
        for (size_t j = 0; j < mesh.extents(1); ++j) {
          for (size_t i = 0; i < mesh.extents(0); ++i) {
            result(p, s) = xi_stencil[i] * eta_stencil[j] * zeta_stencil[k];
            ++s;
          }
        }
      }
    }
    return result;
  }

  // Not FD, so use spectral interpolation
  const std::array<Matrix, 3> matrices{
      {Spectral::interpolation_matrix(mesh.slice_through(0), get<0>(points)),
       Spectral::interpolation_matrix(mesh.slice_through(1), get<1>(points)),
       Spectral::interpolation_matrix(mesh.slice_through(2), get<2>(points))}};

  // First dimension of DataVector varies fastest.
  for (size_t k = 0, s = 0; k < mesh.extents(2); ++k) {
    for (size_t j = 0; j < mesh.extents(1); ++j) {
      for (size_t i = 0; i < mesh.extents(0); ++i) {
        for (size_t p = 0; p < number_of_target_points; ++p) {
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
Irregular<Dim>::Irregular(const Mesh<Dim>& source_mesh,
                          const tnsr::I<DataVector, Dim, Frame::ElementLogical>&
                              target_points) noexcept
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
