// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace Spectral {
namespace detail {
constexpr size_t minimum_number_of_points(const Basis /*basis*/,
                                          const Quadrature quadrature) {
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (quadrature == Quadrature::Gauss) {
    return 1;
    // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (quadrature == Quadrature::GaussLobatto) {
    return 2;
    // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (quadrature == Quadrature::CellCentered) {
    return 1;
    // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (quadrature == Quadrature::FaceCentered) {
    return 2;
    // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (quadrature == Quadrature::Equiangular) {
    return 1;
    // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (quadrature == Quadrature::AxialSymmetry) {
    return 1;
  } else if (quadrature == Quadrature::SphericalSymmetry) {
    return 1;
  }
  return std::numeric_limits<size_t>::max();
}
}  // namespace detail

/*!
 * \brief Minimum number of possible collocation points for a quadrature type.
 *
 * \details Since Gauss-Lobatto quadrature has points on the domain boundaries
 * it must have at least two collocation points. Gauss quadrature can have only
 * one collocation point.
 *
 * \details For `CellCentered` the minimum number of points is 1, while for
 * `FaceCentered` it is 2.
 */
template <Basis basis, Quadrature quadrature>
constexpr size_t minimum_number_of_points =
    detail::minimum_number_of_points(basis, quadrature);
}  // namespace Spectral
