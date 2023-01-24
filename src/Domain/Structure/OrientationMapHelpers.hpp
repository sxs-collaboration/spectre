// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
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

/// @{
/// \ingroup ComputationalDomainGroup
/// \brief Orient variables to the data-storage order of a neighbor element with
/// the given orientation.
///
/// \warning The result is *not* resized and assumes to be of the correct size
/// (`variables.size()`).
template <size_t VolumeDim>
void orient_variables(gsl::not_null<DataVector*> result,
                      const DataVector& variables,
                      const Index<VolumeDim>& extents,
                      const OrientationMap<VolumeDim>& orientation_of_neighbor);

template <size_t VolumeDim>
void orient_variables_on_slice(
    gsl::not_null<DataVector*> result, const DataVector& variables_on_slice,
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
  DataVector result(oriented_variables.data(), oriented_variables.size());
  orient_variables(
      make_not_null(&result),
      DataVector(const_cast<double*>(variables.data()), variables.size()),
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
  DataVector result(oriented_variables.data(), oriented_variables.size());
  orient_variables_on_slice(
      make_not_null(&result),
      DataVector(const_cast<double*>(variables_on_slice.data()),
                 variables_on_slice.size()),
      slice_extents, sliced_dim, orientation_of_neighbor);
  return oriented_variables;
}
/// @}

/// @{
/// \ingroup ComputationalDomainGroup
/// Orient data in a `std::vector<double>` or `DataVector` representing one or
/// more tensor components.
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
    const OrientationMap<VolumeDim>& orientation_of_neighbor);

template <size_t VolumeDim>
DataVector orient_variables(
    const DataVector& variables, const Index<VolumeDim>& extents,
    const OrientationMap<VolumeDim>& orientation_of_neighbor);

template <size_t VolumeDim>
std::vector<double> orient_variables_on_slice(
    const std::vector<double>& variables_on_slice,
    const Index<VolumeDim - 1>& slice_extents, size_t sliced_dim,
    const OrientationMap<VolumeDim>& orientation_of_neighbor);

template <size_t VolumeDim>
DataVector orient_variables_on_slice(
    const DataVector& variables_on_slice,
    const Index<VolumeDim - 1>& slice_extents, size_t sliced_dim,
    const OrientationMap<VolumeDim>& orientation_of_neighbor);
/// @}
