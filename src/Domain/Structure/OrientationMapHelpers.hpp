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
class OrientationMap;
template <size_t>
class Index;
template <typename TagsList>
class Variables;
/// \endcond

namespace OrientationMapHelpers_detail {
template <typename T>
void orient_each_component(gsl::not_null<gsl::span<T>*> oriented_variables,
                           const gsl::span<const T>& variables, size_t num_pts,
                           const std::vector<size_t>& oriented_offset) noexcept;

template <size_t VolumeDim>
std::vector<size_t> oriented_offset(
    const Index<VolumeDim>& extents,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) noexcept;

inline std::vector<size_t> oriented_offset_on_slice(
    const Index<0>& /*slice_extents*/, const size_t /*sliced_dim*/,
    const OrientationMap<1>& /*orientation_of_neighbor*/) noexcept {
  // There is only one point on a slice of a 1D mesh
  return {0};
}

std::vector<size_t> oriented_offset_on_slice(
    const Index<1>& slice_extents, size_t sliced_dim,
    const OrientationMap<2>& orientation_of_neighbor) noexcept;

std::vector<size_t> oriented_offset_on_slice(
    const Index<2>& slice_extents, size_t sliced_dim,
    const OrientationMap<3>& orientation_of_neighbor) noexcept;

}  // namespace OrientationMapHelpers_detail

/// @{
/// \ingroup ComputationalDomainGroup
/// Orient variables to the data-storage order of a neighbor element with
/// the given orientation.
template <size_t VolumeDim, typename TagsList>
Variables<TagsList> orient_variables(
    const Variables<TagsList>& variables, const Index<VolumeDim>& extents,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) noexcept {
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
  const auto oriented_offset = OrientationMapHelpers_detail::oriented_offset(
      extents, orientation_of_neighbor);
  auto oriented_vars_view = gsl::make_span(oriented_variables);
  OrientationMapHelpers_detail::orient_each_component(
      make_not_null(&oriented_vars_view), gsl::make_span(variables),
      number_of_grid_points, oriented_offset);

  return oriented_variables;
}

template <size_t VolumeDim, typename TagsList>
Variables<TagsList> orient_variables_on_slice(
    const Variables<TagsList>& variables_on_slice,
    const Index<VolumeDim - 1>& slice_extents, const size_t sliced_dim,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) noexcept {
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
  const auto oriented_offset =
      OrientationMapHelpers_detail::oriented_offset_on_slice(
          slice_extents, sliced_dim, orientation_of_neighbor);

  auto oriented_vars_view = gsl::make_span(oriented_variables);
  OrientationMapHelpers_detail::orient_each_component(
      make_not_null(&oriented_vars_view), gsl::make_span(variables_on_slice),
      number_of_grid_points, oriented_offset);

  return oriented_variables;
}
/// @}

/// @{
/// \ingroup ComputationalDomainGroup
/// Orient data in a `std::vector<double>` representing one or more tensor
/// components.
///
/// In most cases the `Variables` version of `orient_variables` should be
/// called. However, in some cases the tags and thus the type of the data being
/// sent is determined at runtime. In these cases the `std::vector` version of
/// `orient_variables` is useful. A concrete example of this is when hybridizing
/// DG with finite difference methods, where sometimes the data sent is both the
/// variables for reconstruction and the fluxes for either the DG or finite
/// difference scheme, while at other points only one of these three is sent.
template <size_t VolumeDim>
std::vector<double> orient_variables(
    const std::vector<double>& variables, const Index<VolumeDim>& extents,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) noexcept;

template <size_t VolumeDim>
std::vector<double> orient_variables_on_slice(
    const std::vector<double>& variables_on_slice,
    const Index<VolumeDim - 1>& slice_extents, size_t sliced_dim,
    const OrientationMap<VolumeDim>& orientation_of_neighbor) noexcept;
/// @}
