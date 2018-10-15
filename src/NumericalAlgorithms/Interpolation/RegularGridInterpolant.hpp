// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"  // IWYU pragma: keep
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

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
/// The target points must lie on a tensor-product grid that is aligned with the
/// source Mesh. In any direction where the source and target points share an
/// identical Mesh (i.e., where the underlying 1-dimensional meshes share the
/// same extent, basis, and quadrature), the code is optimized to avoid
/// performing identity interpolations.
///
/// Note, however, that in each dimension of the target grid, the points can
/// be freely distributed; in particular, the grid points need not be the
/// collocation points corresponding to a particular basis and quadrature. Note
/// also that the target grid need not overlap the source grid. In this case,
/// polynomial extrapolation is performed, with order set by the order of the
/// basis in the source grid. The extrapolation will be correct but may suffer
/// from reduced accuracy, especially for higher-order polynomials.
template <size_t Dim>
class RegularGrid {
 public:
  /// \brief An interpolator between two regular grids.
  ///
  /// When the optional third argument is NOT passed, creates an interpolator
  /// between two regular meshes.
  ///
  /// The optional third argument allows the caller to override the distribution
  /// of grid points in any dimension(s) of the target grid. Each non-empty
  /// element of `override_target_mesh_with_1d_logical_coords` gives the logical
  /// coordinates which will override the default coordinates of `target_mesh`.
  RegularGrid(const Mesh<Dim>& source_mesh, const Mesh<Dim>& target_mesh,
              const std::array<DataVector, Dim>&
                  override_target_mesh_with_1d_logical_coords =
                      make_array<Dim>(DataVector())) noexcept;

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
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const RegularGrid<LocalDim>& lhs,
                         const RegularGrid<LocalDim>& rhs) noexcept;

  size_t number_of_target_points_{};
  Index<Dim> source_extents_;
  std::array<Matrix, Dim> interpolation_matrices_;
};

template <size_t Dim>
template <typename TagsList>
void RegularGrid<Dim>::interpolate(
    const gsl::not_null<Variables<TagsList>*> result,
    const Variables<TagsList>& vars) const noexcept {
  if (result->number_of_grid_points() != number_of_target_points_) {
    result->initialize(number_of_target_points_);
  }
  apply_matrices(result, interpolation_matrices_, vars, source_extents_);
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
