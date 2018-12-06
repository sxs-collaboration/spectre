// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/LinearOperators/Transpose.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Variables

/// \cond
template <size_t Dim>
class Mesh;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace intrp {

/// \ingroup NumericalAlgorithmsGroup
/// \brief Interpolate a Variables from a Mesh onto a regular grid of points.
///
/// The target points must lie on tensor-product grid that is aligned with the
/// source Mesh. In each dimension of the target grid, however, the points can
/// be freely distributed; in particular, the grid points need not be the
/// collocation points corresponding to a particular basis and quadrature.
///
/// \note When interpolating between two different resolutions on the same type
/// of spectral grid (e.g., between two Legendre Gauss Lobatto grids with
/// different numbers of points), it may be more efficient to use a projection
/// instead of the interpolation that is done by this class. This class is
/// intended to handle interpolation onto more general (but still regular)
/// target grids.
///
/// \note The target grid need not overlap the source grid. In this case,
/// polynomial extrapolation is performed, with order set by the order of the
/// basis in the source grid. The extapolation will be correct but may suffer
/// from reduced accuracy, especially for higher-order polynomials.
template <size_t Dim>
class RegularGrid {
 public:
  /// \brief An interpolator between two overlapping meshes.
  RegularGrid(const Mesh<Dim>& source_mesh,
              const Mesh<Dim>& target_mesh) noexcept;
  /// \brief An interpolator from a Mesh to a regular grid of points.
  RegularGrid(
      const Mesh<Dim>& source_mesh,
      const std::array<DataVector, Dim>& target_1d_logical_coords) noexcept;
  RegularGrid();

  // clang-tidy: no runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

  /// \brief Interpolate Variables onto new mesh.
  //@{
  template <typename TagsList>
  void interpolate(gsl::not_null<Variables<TagsList>*> result,
                   const Variables<TagsList>& vars) const noexcept;
  template <typename TagsList>
  Variables<TagsList> interpolate(const Variables<TagsList>& vars) const
      noexcept;
  //@}

 private:
  template <size_t LocalDim>
  friend bool operator==(const RegularGrid<LocalDim>& lhs,
                         const RegularGrid<LocalDim>& rhs) noexcept;

  std::array<Matrix, Dim> interpolation_matrices_;
};

template <>
template <typename TagsList>
void RegularGrid<1>::interpolate(
    const gsl::not_null<Variables<TagsList>*> result,
    const Variables<TagsList>& vars) const noexcept {
  // For matrix multiplication of Interp . Source = Result:
  //   matrix Interp is m rows by k columns
  //   matrix Source is k rows by n columns
  //   matrix Result is m rows by n columns
  const size_t m = interpolation_matrices_[0].rows();
  const size_t k = interpolation_matrices_[0].columns();
  const size_t n = vars.number_of_independent_components;
  ASSERT(k == vars.number_of_grid_points(),
         "Number of grid points in source 'vars', "
             << vars.number_of_grid_points()
             << ",\n disagrees with the size of the source_mesh, " << k
             << ", that was passed into the constructor");
  if (result->number_of_grid_points() != m) {
    *result = Variables<TagsList>(m, 0.0);
  }
  dgemm_('n', 'n', m, n, k, 1.0, interpolation_matrices_[0].data(), m,
         vars.data(), k, 0.0, result->data(), m);
}

template <>
template <typename TagsList>
void RegularGrid<2>::interpolate(
    const gsl::not_null<Variables<TagsList>*> result,
    const Variables<TagsList>& vars) const noexcept {
  // source_mesh extents: (sx, sy)
  // target_mesh extents: (tx, ty)
  // number of variables: n
  const size_t tx = interpolation_matrices_[0].rows();
  const size_t sx = interpolation_matrices_[0].columns();
  const size_t ty = interpolation_matrices_[1].rows();
  const size_t sy = interpolation_matrices_[1].columns();
  const size_t n = vars.number_of_independent_components;
  ASSERT(sx * sy == vars.number_of_grid_points(),
         "Number of grid points in source 'vars', "
             << vars.number_of_grid_points()
             << ",\n disagrees with the size of the source_mesh, " << sx * sy
             << ", that was passed into the constructor");
  if (result->number_of_grid_points() != tx * ty) {
    *result = Variables<TagsList>(tx * ty, 0.0);
  }

  // Note: The sequence of DGEMM and transpose calls needs several temporaries.
  //       Here we loop over the variable's components (`var`), so that we
  //       operate one DataVector at a time, with temporaries being DataVectors.
  //       The algorithm _can_ be rewritten so each DGEMM call works on the
  //       entire Vars at once, but then each temporary must become a Variables.
  //       Furthermore, looping over variable components is still necessary in
  //       the forward and backward transpose operations.
  //       (Note: to move the DGEMM calls out of the loop, simply take the 4th
  //        argument (number of rol/col multiplies) from, e.g., sy -> sy * n.)
  const size_t source_size = sx * sy;
  const size_t temp_size = tx * sy;
  const size_t target_size = tx * ty;
  // By making buffer1 large enough to hold two different stages in the
  // computation, we can reuse it and avoid the need to allocate a 3rd buffer.
  DataVector buffer1(std::max(temp_size, target_size));
  DataVector buffer2(temp_size);
  for (size_t var = 0; var < n; ++var) {
    // clang-tidy: do not use pointer arithmetic
    const double* const ptr_to_var = vars.data() + var * source_size;  // NOLINT
    double* const ptr_to_result = result->data() + var * target_size;  // NOLINT

    // interpolate in x
    dgemm_('n', 'n', tx, sy, sx, 1.0, interpolation_matrices_[0].data(), tx,
           ptr_to_var, sx, 0.0, buffer1.data(), tx);

    // transpose, interpolate in y, transpose back
    // Note: the transpose allows the matrix multiply to act on contiguous data.
    raw_transpose(make_not_null(buffer2.data()), buffer1.data(), tx, sy);
    dgemm_('n', 'n', ty, tx, sy, 1.0, interpolation_matrices_[1].data(), ty,
           buffer2.data(), sy, 0.0, buffer1.data(), ty);
    raw_transpose(make_not_null(ptr_to_result), buffer1.data(), ty, tx);
  }
}

template <>
template <typename TagsList>
void RegularGrid<3>::interpolate(
    const gsl::not_null<Variables<TagsList>*> result,
    const Variables<TagsList>& vars) const noexcept {
  // source_mesh extents: (sx, sy, sz)
  // target_mesh extents: (tx, ty, tz)
  // number of variables: n
  const size_t tx = interpolation_matrices_[0].rows();
  const size_t sx = interpolation_matrices_[0].columns();
  const size_t ty = interpolation_matrices_[1].rows();
  const size_t sy = interpolation_matrices_[1].columns();
  const size_t tz = interpolation_matrices_[2].rows();
  const size_t sz = interpolation_matrices_[2].columns();
  const size_t n = vars.number_of_independent_components;
  ASSERT(sx * sy * sz == vars.number_of_grid_points(),
         "Number of grid points in source 'vars', "
             << vars.number_of_grid_points()
             << ",\n disagrees with the size of the source_mesh, "
             << sx * sy * sz << ", that was passed into the constructor");
  if (result->number_of_grid_points() != tx * ty * tz) {
    *result = Variables<TagsList>(tx * ty * tz, 0.0);
  }

  // Note: The sequence of DGEMM and transpose calls needs several temporaries.
  //       Here we loop over the variable's components (`var`), so that we
  //       operate one DataVector at a time, with temporaries being DataVectors.
  //       The algorithm _can_ be rewritten so each DGEMM call works on the
  //       entire Vars at once, but then each temporary must become a Variables.
  //       Furthermore, looping over variable components is still necessary in
  //       the forward and backward transpose operations.
  //       (Note: to move the DGEMM calls out of the loop, simply take the 4th
  //        argument (number of rol/col multiplies) from, e.g., sy -> sy * n.)
  const size_t source_size = sx * sy * sz;
  const size_t temp1_size = tx * sy * sz;
  const size_t temp2_size = tx * ty * sz;
  const size_t target_size = tx * ty * tz;
  // By making both buffers large enough to hold several different stages in the
  // computation, we can reuse them and avoid the need to allocate five buffers.
  DataVector buffer1(std::max(std::max(temp1_size, temp2_size), target_size));
  DataVector buffer2(std::max(temp1_size, temp2_size));
  for (size_t var = 0; var < n; ++var) {
    // clang-tidy: do not use pointer arithmetic
    const double* const ptr_to_var = vars.data() + var * source_size;  // NOLINT
    double* const ptr_to_result = result->data() + var * target_size;  // NOLINT

    // interpolate in x
    dgemm_('n', 'n', tx, sy * sz, sx, 1.0, interpolation_matrices_[0].data(),
           tx, ptr_to_var, sx, 0.0, buffer1.data(), tx);

    // transpose, interpolate in y
    // Note: the transpose allows the matrix multiply to act on contiguous data.
    raw_transpose(make_not_null(buffer2.data()), buffer1.data(), tx, sy * sz);
    dgemm_('n', 'n', ty, tx * sz, sy, 1.0, interpolation_matrices_[1].data(),
           ty, buffer2.data(), sy, 0.0, buffer1.data(), ty);

    // transpose, interpolate in z, transpose back
    // Note: the transpose allows the matrix multiply to act on contiguous data.
    raw_transpose(make_not_null(buffer2.data()), buffer1.data(), ty, tx * sz);
    dgemm_('n', 'n', tz, tx * ty, sz, 1.0, interpolation_matrices_[2].data(),
           tz, buffer2.data(), sz, 0.0, buffer1.data(), tz);
    raw_transpose(make_not_null(ptr_to_result), buffer1.data(), tz, tx * ty);
  }
}

template <size_t Dim>
template <typename TagsList>
Variables<TagsList> RegularGrid<Dim>::interpolate(
    const Variables<TagsList>& vars) const noexcept {
  Variables<TagsList> result;
  interpolate(make_not_null(&result), vars);
  return result;
}

template <size_t Dim>
bool operator!=(const RegularGrid<Dim>& lhs,
                const RegularGrid<Dim>& rhs) noexcept;

}  // namespace intrp
