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
#include "Utilities/TypeTraits.hpp"  // IWYU pragma: keep

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
 * obtained from Spectral::collocation_points(const Mesh<1>&):
 *
 * \snippet Test_Spectral.cpp get_points_for_mesh
 *
 * A simulated physical field can be represented by a DataVector of length
 * number_of_grid_points() that holds the field value on each point of
 * the computational grid. These values are identical to the field's nodal
 * expansion coefficients. They approximate the field by a polynomial of degree
 * \f$p=N-1\f$ through a linear combination of Lagrange polynomials.
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
   * \brief The number of grid points in dimension \p d of the grid
   * (zero-indexed).
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
   * \brief Returns the 1-dimensional index corresponding to the `Dim`
   * dimensional `index`.
   *
   * The first dimension varies fastest.
   *
   * \see collapsed_index()
   */
  size_t storage_index(const Index<Dim>& index) const noexcept {
    return collapsed_index(index, extents_);
  }

  /*!
   * \brief The basis chosen in each dimension of the grid.
   */
  const std::array<Spectral::Basis, Dim>& basis() const noexcept {
    return bases_;
  }

  /*!
   * \brief The basis chosen in dimension \p d of the grid (zero-indexed).
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
   * \brief The quadrature chosen in dimension \p d of the grid (zero-indexed).
   */
  Spectral::Quadrature quadrature(const size_t d) const noexcept {
    return gsl::at(quadratures_, d);
  }

  /*!
   * \brief Returns a Mesh with dimension \p d removed (zero-indexed).
   *
   * \see slice_through()
   */
  // clang-tidy: incorrectly reported redundancy in template expression
  template <size_t N = Dim, Requires<(N > 0 and N == Dim)> = nullptr>  // NOLINT
  Mesh<Dim - 1> slice_away(size_t d) const noexcept;

  /*!
   * \brief Returns a Mesh with the dimensions \p d, ... present (zero-indexed).
   *
   * \details Generally you use this method to obtain a lower-dimensional Mesh
   * by slicing through a subset of the dimensions. However, you can also
   * reorder dimensions using this method by slicing through the dimensions in
   * an order you choose.
   *
   * \see slice_away()
   */
  template <typename... D, Requires<(sizeof...(D) <= Dim)> = nullptr>
  Mesh<sizeof...(D)> slice_through(D... d) const noexcept {
    static_assert(cpp17::conjunction_v<tt::is_integer<D>...>,
                  "The dimensions must be integers.");
    const std::array<size_t, sizeof...(D)> dims{{static_cast<size_t>(d)...}};
    return slice_through(dims);
  }

  /*!
   * \brief Returns a Mesh with the dimensions \p dims present (zero-indexed).
   *
   * \see slice_through() The templated overload of this function
   */
  template <size_t SliceDim, Requires<(SliceDim <= Dim)> = nullptr>
  Mesh<SliceDim> slice_through(const std::array<size_t, SliceDim>& dims) const
      noexcept;

  /*!
   * \brief Returns the Meshes representing 1D slices of this Mesh.
   *
   * The is the same as the array filled with `slice_through(d)` for
   * `d` from `0` to `Dim - 1` except in dimension 0 where
   * `slice_through(d)` is not defined.
   */
  std::array<Mesh<1>, Dim> slices() const noexcept;

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

template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const Mesh<Dim>& mesh) noexcept {
  os << mesh.extents();
  return os;
}
/// \endcond
