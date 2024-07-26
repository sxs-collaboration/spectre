// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class template Mesh.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Index.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/String.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsInteger.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
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
 * \note Because we use a compact bit representation for the mesh, the number
 * of grid points/extents must be fewer than 256 per dimension.
 *
 * \tparam Dim the number of dimensions of the computational grid.
 */
template <size_t Dim>
class Mesh {
 public:
  static constexpr size_t dim = Dim;

  struct Extents {
    using type = size_t;
    static constexpr Options::String help = {
        "The number of collocation points per dimension"};
  };

  struct Basis {
    using type = Spectral::Basis;
    static constexpr Options::String help = {
        "The choice of spectral basis to compute the collocation points"};
  };

  struct Quadrature {
    using type = Spectral::Quadrature;
    static constexpr Options::String help = {
        "The choice of quadrature to compute the collocation points"};
  };

  using options = tmpl::list<Extents, Basis, Quadrature>;

  static constexpr Options::String help =
      "Holds the number of grid points, basis, and quadrature in each "
      "direction of the computational grid. "
      "A mesh encapsulates all information necessary to construct the "
      "placement of grid points in the computational domain. It does so "
      "through a choice of basis functions, quadrature and number of points "
      "in each dimension.";

  Mesh() = default;

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
   *
   * \note Because a `Mesh<0>` extends over no dimensions, it has 1 grid point
   * independent of the value of `isotropic_extents`.
   */
  Mesh(size_t isotropic_extents, Spectral::Basis basis,
       Spectral::Quadrature quadrature);

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
  Mesh(const std::array<size_t, Dim>& extents, Spectral::Basis basis,
       Spectral::Quadrature quadrature);

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
  Mesh(const std::array<size_t, Dim>& extents,
       const std::array<Spectral::Basis, Dim>& bases,
       const std::array<Spectral::Quadrature, Dim>& quadratures);

  /*!
   * \brief The number of grid points in each dimension of the grid.
   */
  Index<Dim> extents() const;

  /*!
   * \brief The number of grid points in dimension \p d of the grid
   * (zero-indexed).
   */
  size_t extents(size_t d) const;

  /*!
   * \brief The total number of grid points in all dimensions.
   *
   * \details `DataVector`s that represent field values on the grid have this
   * many entries.
   *
   * \note A zero-dimensional mesh has one grid point, since it is the slice
   * through a one-dimensional mesh (a line).
   */
  size_t number_of_grid_points() const;

  /*!
   * \brief Returns the 1-dimensional index corresponding to the `Dim`
   * dimensional `index`.
   *
   * The first dimension varies fastest.
   *
   * \see collapsed_index()
   */
  size_t storage_index(const Index<Dim>& index) const;

  /*!
   * \brief The basis chosen in each dimension of the grid.
   */
  std::array<Spectral::Basis, Dim> basis() const;

  /*!
   * \brief The basis chosen in dimension \p d of the grid (zero-indexed).
   */
  Spectral::Basis basis(size_t d) const;

  /*!
   * \brief The quadrature chosen in each dimension of the grid.
   */
  std::array<Spectral::Quadrature, Dim> quadrature() const;

  /*!
   * \brief The quadrature chosen in dimension \p d of the grid (zero-indexed).
   */
  Spectral::Quadrature quadrature(size_t d) const;

  /*!
   * \brief Returns a Mesh with dimension \p d removed (zero-indexed).
   *
   * \see slice_through()
   */
  // clang-tidy: incorrectly reported redundancy in template expression
  template <size_t N = Dim, Requires<(N > 0 and N == Dim)> = nullptr>  // NOLINT
  Mesh<Dim - 1> slice_away(size_t d) const;

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
  Mesh<sizeof...(D)> slice_through(D... d) const {
    static_assert(std::conjunction_v<tt::is_integer<D>...>,
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
  Mesh<SliceDim> slice_through(const std::array<size_t, SliceDim>& dims) const;

  /*!
   * \brief Returns the Meshes representing 1D slices of this Mesh.
   *
   * The is the same as the array filled with `slice_through(d)` for
   * `d` from `0` to `Dim - 1` except in dimension 0 where
   * `slice_through(d)` is not defined.
   */
  std::array<Mesh<1>, Dim> slices() const;

  // clang-tidy: runtime-references
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  template <size_t LocalDim>
  friend class Mesh;

  template <size_t LocalDim>
  friend bool operator==(const Mesh<LocalDim>& lhs, const Mesh<LocalDim>& rhs);

  // - 8 bits per extent = 3 * 8 = 24 bits = 3 bytes
  //   This limits us to 255 extent per direction. Since FD is 2N-1 DG, this
  //   means at most 128 DG points, which is should be fine.
  // - We encode the basis & quadrature as two sets of 4 bits in a uint8_t.
  std::array<uint8_t, (Dim > 0 ? Dim : 1)> extents_{};
  std::array<uint8_t, (Dim > 0 ? Dim : 1)> quadrature_and_basis_{};
};

/*!
 * \ingroup DataStructuresGroup
 * \brief Returns `true` if the mesh is isotropic, `false` otherwise.
 *
 * If `Dim` is zero, then `true` is always returned.
 */
template <size_t Dim>
bool is_isotropic(const Mesh<Dim>& mesh);

/// \cond HIDDEN_SYMBOLS
template <size_t Dim>
bool operator==(const Mesh<Dim>& lhs, const Mesh<Dim>& rhs);

template <size_t Dim>
bool operator!=(const Mesh<Dim>& lhs, const Mesh<Dim>& rhs);

template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const Mesh<Dim>& mesh);
/// \endcond
