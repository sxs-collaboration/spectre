// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t VolumeDim>
class Mesh;
template <size_t VolumeDim>
class OrientationMap;
template <size_t>
class Index;
template <typename TagsList>
class Variables;
/// \endcond

/// \ingroup ComputationalDomainGroup
/// \brief Orient a sliced Mesh to the logical frame of  a neighbor element with
/// the given orientation.
template <size_t VolumeDim>
Mesh<VolumeDim - 1> orient_mesh_on_slice(
    const Mesh<VolumeDim - 1>& mesh_on_slice, size_t sliced_dim,
    const OrientationMap<VolumeDim>& orientation_of_neighbor);

/// @{
/// \ingroup ComputationalDomainGroup
/// \brief Orient a `DataVector`, `ComplexDataVector`, `std::vector<double>`, or
/// `std::vector<std::complex<double>>` to the data-storage order of a neighbor
/// element with the given orientation.
///
/// The vector may represent more than one tensor component over the grid
/// represented by `extents`.
///
/// \warning The result is *not* resized and assumes to be of the correct size
/// (`variables.size()`).
///
/// In most cases the `Variables` version of `orient_variables` should be
/// called. However, in some cases the tags and thus the type of the data being
/// sent is determined at runtime. In these cases the `std::vector` version of
/// `orient_variables` is useful. A concrete example of this is when hybridizing
/// DG with finite difference methods, where sometimes the data sent is both the
/// variables for reconstruction and the fluxes for either the DG or finite
/// difference scheme, while at other points only one of these three is sent.
template <typename VectorType, size_t VolumeDim>
void orient_variables(gsl::not_null<VectorType*> result,
                      const VectorType& variables,
                      const Index<VolumeDim>& extents,
                      const OrientationMap<VolumeDim>& orientation_of_neighbor);

template <typename VectorType, size_t VolumeDim>
VectorType orient_variables(
    const VectorType& variables, const Index<VolumeDim>& extents,
    const OrientationMap<VolumeDim>& orientation_of_neighbor);

template <typename VectorType, size_t VolumeDim>
void orient_variables_on_slice(
    gsl::not_null<VectorType*> result, const VectorType& variables_on_slice,
    const Index<VolumeDim - 1>& slice_extents, size_t sliced_dim,
    const OrientationMap<VolumeDim>& orientation_of_neighbor);

template <typename VectorType, size_t VolumeDim>
VectorType orient_variables_on_slice(
    const VectorType& variables_on_slice,
    const Index<VolumeDim - 1>& slice_extents, size_t sliced_dim,
    const OrientationMap<VolumeDim>& orientation_of_neighbor);
/// @}

/// @{
/// \ingroup ComputationalDomainGroup
/// Orient variables to the data-storage order of a neighbor element with
/// the given orientation.
template <size_t VolumeDim, typename TagsList>
Variables<TagsList> orient_variables(
    const Variables<TagsList>& variables, const Index<VolumeDim>& extents,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) {
  // Skip work (aside from a copy) if neighbor is aligned
  if (orientation_of_neighbor.is_aligned()) {
    return variables;
  }

  const size_t number_of_grid_points = extents.product();
  ASSERT(variables.number_of_grid_points() == number_of_grid_points,
         "Inconsistent `variables` and `extents`:\n"
         "  variables.number_of_grid_points() = "
             << variables.number_of_grid_points()
             << "\n"
                "  extents.product() = "
             << extents.product());
  Variables<TagsList> oriented_variables(number_of_grid_points);
  using VectorType = typename Variables<TagsList>::vector_type;
  using ValueType = typename Variables<TagsList>::value_type;
  VectorType result(oriented_variables.data(), oriented_variables.size());
  orient_variables(
      make_not_null(&result),
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      VectorType(const_cast<ValueType*>(variables.data()), variables.size()),
      extents, orientation_of_neighbor);
  return oriented_variables;
}

template <size_t VolumeDim, typename TagsList>
Variables<TagsList> orient_variables_on_slice(
    const Variables<TagsList>& variables_on_slice,
    const Index<VolumeDim - 1>& slice_extents, const size_t sliced_dim,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) {
  // Skip work (aside from a copy) if neighbor slice is aligned
  if (orientation_of_neighbor.is_aligned()) {
    return variables_on_slice;
  }

  const size_t number_of_grid_points = slice_extents.product();
  ASSERT(variables_on_slice.number_of_grid_points() == number_of_grid_points,
         "Inconsistent `variables_on_slice` and `slice_extents`:\n"
         "  variables_on_slice.number_of_grid_points() = "
             << variables_on_slice.number_of_grid_points()
             << "\n"
                "  slice_extents.product() = "
             << slice_extents.product());

  Variables<TagsList> oriented_variables(number_of_grid_points);
  using VectorType = typename Variables<TagsList>::vector_type;
  using ValueType = typename Variables<TagsList>::value_type;
  VectorType result(oriented_variables.data(), oriented_variables.size());
  orient_variables_on_slice(
      make_not_null(&result),
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      VectorType(const_cast<ValueType*>(variables_on_slice.data()),
                 variables_on_slice.size()),
      slice_extents, sliced_dim, orientation_of_neighbor);
  return oriented_variables;
}
/// @}
