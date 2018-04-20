// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class template Mesh.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Spectral {
enum class Basis;
enum class Quadrature;
}  // namespace Spectral
/// \endcond

/*!
 * \ingroup DataStructuresGroup
 * \brief Holds the number of grid points, basis, and quadrature in each
 * direction of the computational grid.
 *
 * \details A mesh encapsulates all information necessary to construct the
 * placement of grid points in the computational domain. It does so through a
 * choice of basis functions, quadrature and number of points \f$N\f$ in each
 * dimension. The grid points are the associated collocation points and can be
 * computed by `Spectral::collocation_points`. This means that a simulated
 * physical field can be represented by its value on each grid point and then
 * approximated by a polynomial of degree \f$p=N-1\f$ through a linear
 * combination of Lagrange polynomials.
 *
 * \note A field represented by a `DataVector` has no meaning without an
 * accompanying `Mesh` to provide context. Only w.r.t a `Mesh` can a
 * `DataVector` of length `mesh.number_of_grid_points()` be interpreted as the
 * field values at the collocation points. These field values are identical to
 * the nodal expansion coefficients in Lagrange polynomials.
 *
 * \tparam Dim the number of dimensions of the computational grid.
 */
template <size_t Dim>
class Mesh {
 public:
  Mesh() noexcept = default;

  /*!
   * \brief Construct a computational grid with the same number of collocation
   * points in each dimension.
   *
   * \param isotropic_extents The number of collocation points in each
   * dimension.
   * \param basis The choice of spectral basis to compute the
   * collocation points
   * \param quadrature The choice of quadrature to compute
   * the collocation points
   */
  Mesh(const size_t isotropic_extents, const Spectral::Basis basis,
       const Spectral::Quadrature quadrature) noexcept
      : extents_(isotropic_extents) {
    bases_.fill(basis);
    quadratures_.fill(quadrature);
  }

  /*!
   * \brief Construct a computational grid where each dimension can have a
   * different number of collocation points.
   *
   * \param extents The number of collocation points per dimension
   * \param basis The choice of spectral basis to compute the
   * collocation points
   * \param quadrature The choice of quadrature to compute
   * the collocation points
   */
  Mesh(std::array<size_t, Dim> extents, const Spectral::Basis basis,
       const Spectral::Quadrature quadrature) noexcept
      : extents_(std::move(extents)) {
    bases_.fill(basis);
    quadratures_.fill(quadrature);
  }

  /*!
   * \brief Construct a computational grid where each dimension can have both a
   * different number and placement of collocation points.
   *
   * \param extents The number of collocation points per dimension
   * \param bases The choice of spectral bases to compute the
   * collocation points per dimension
   * \param quadratures The choice of quadratures to compute
   * the collocation points per dimension
   */
  Mesh(std::array<size_t, Dim> extents, std::array<Spectral::Basis, Dim> bases,
       std::array<Spectral::Quadrature, Dim> quadratures) noexcept
      : extents_(std::move(extents)),
        bases_(std::move(bases)),
        quadratures_(std::move(quadratures)) {}

  /*!
   * \brief The number of grid points in each dimension of the grid.
   */
  const Index<Dim>& extents() const noexcept { return extents_; }

  /*!
   * \brief The number of grid points in dimension `d` of the grid.
   */
  size_t extents(const size_t d) const noexcept { return extents_[d]; }

  /*!
   * \brief The total number of grid points in all dimensions.
   *
   * \details `DataVector`s that represent field values on the grid have this
   * many entries.
   *
   * \note A zero-dimensional mesh has one grid point, since it is the slice
   * through a one-dimensional mesh (a line).
   */
  size_t number_of_grid_points() const noexcept { return extents_.product(); }

  /*!
   * \brief The basis chosen in each dimension of the grid.
   */
  const std::array<Spectral::Basis, Dim>& basis() const noexcept {
    return bases_;
  }

  /*!
   * \brief The basis chosen in dimension `d` of the grid.
   */
  Spectral::Basis basis(const size_t d) const noexcept {
    return gsl::at(bases_, d);
  }

  /*!
   * \brief The quadrature chosen in each dimension of the grid.
   */
  const std::array<Spectral::Quadrature, Dim>& quadrature() const noexcept {
    return quadratures_;
  }

  /*!
   * \brief The quadrature chosen in dimension `d` of the grid.
   */
  Spectral::Quadrature quadrature(const size_t d) const noexcept {
    return gsl::at(quadratures_, d);
  }

  /*!
   * \brief Returns a mesh with dimension `d` removed.
   *
   * \param d The dimension to remove (zero-indexed).
   */
  // clang-tidy: incorrectly reported redundancy in template expression
  template <size_t N = Dim, Requires<(N > 0 and N == Dim)> = nullptr>  // NOLINT
  Mesh<Dim - 1> slice_away(size_t d) const noexcept;

  // clang-tidy: runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  Index<Dim> extents_{};
  std::array<Spectral::Basis, Dim> bases_{};
  std::array<Spectral::Quadrature, Dim> quadratures_{};
};

/// \cond HIDDEN_SYMBOLS
template <size_t Dim>
bool operator==(const Mesh<Dim>& lhs, const Mesh<Dim>& rhs) noexcept;

template <size_t Dim>
bool operator!=(const Mesh<Dim>& lhs, const Mesh<Dim>& rhs) noexcept;
/// \endcond
