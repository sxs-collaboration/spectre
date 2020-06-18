// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoGridHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoOscillationIndicator.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "Utilities/Gsl.hpp"

namespace Limiters::Weno_detail {

// Compute the Simple WENO solution for one tensor component
//
// This interface is intended for use in limiters that check the troubled-cell
// indicator independently for each tensor component. These limiters generally
// need to limit only a subset of the tensor components.
//
// When calling `simple_weno_impl`,
// - `interpolator_buffer` may be empty
// - `modified_neighbor_solution_buffer` should contain one DataVector for each
//   neighboring element (i.e. for each entry in `neighbor_data`)
template <typename Tag, size_t VolumeDim, typename PackagedData>
void simple_weno_impl(
    const gsl::not_null<std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        intrp::RegularGrid<VolumeDim>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>*>
        interpolator_buffer,
    const gsl::not_null<std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>*>
        modified_neighbor_solution_buffer,
    const gsl::not_null<db::item_type<Tag>*> tensor,
    const double neighbor_linear_weight, const size_t tensor_storage_index,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) noexcept {
  ASSERT(
      modified_neighbor_solution_buffer->size() == neighbor_data.size(),
      "modified_neighbor_solution_buffer->size() = "
          << modified_neighbor_solution_buffer->size()
          << "\nneighbor_data.size() = " << neighbor_data.size()
          << "\nmodified_neighbor_solution_buffer was incorrectly initialized "
             "before calling simple_weno_impl.");

  // Compute the modified neighbor solutions.
  // First extrapolate neighbor data onto local grid points, then shift the
  // extrapolated data so its mean matches the local mean.
  DataVector& component_to_limit = (*tensor)[tensor_storage_index];
  const double local_mean = mean_value(component_to_limit, mesh);
  for (const auto& neighbor_and_data : neighbor_data) {
    const auto& neighbor = neighbor_and_data.first;
    const auto& data = neighbor_and_data.second;

    if (interpolator_buffer->find(neighbor) == interpolator_buffer->end()) {
      // No interpolator found => create one
      const auto& direction = neighbor.first;
      const auto& source_mesh = data.mesh;
      const auto target_1d_logical_coords =
          Weno_detail::local_grid_points_in_neighbor_logical_coords(
              mesh, source_mesh, element, direction);
      interpolator_buffer->insert(std::make_pair(
          neighbor, intrp::RegularGrid<VolumeDim>(source_mesh, mesh,
                                                  target_1d_logical_coords)));
    }

    // Avoid allocations by working directly in the preallocated buffer
    DataVector& buffer = modified_neighbor_solution_buffer->at(neighbor);

    interpolator_buffer->at(neighbor).interpolate(
        make_not_null(&buffer),
        get<Tag>(data.volume_data)[tensor_storage_index]);
    const double neighbor_mean = mean_value(buffer, mesh);
    buffer += (local_mean - neighbor_mean);
  }

  // Sum local and modified neighbor polynomials for the WENO reconstruction
  Weno_detail::reconstruct_from_weighted_sum(
      make_not_null(&component_to_limit), neighbor_linear_weight,
      Weno_detail::DerivativeWeight::PowTwoEll, mesh,
      *modified_neighbor_solution_buffer);
}

// Compute the Simple WENO solution for one tensor
//
// This interface is intended for use in limiters that check the troubled-cell
// indicator for the whole cell, and apply the limiter to all fields.
//
// When calling `simple_weno_impl`,
// - `interpolator_buffer` may be empty
// - `modified_neighbor_solution_buffer` should contain one DataVector for each
//   neighboring element (i.e. for each entry in `neighbor_data`)
template <typename Tag, size_t VolumeDim, typename PackagedData>
void simple_weno_impl(
    const gsl::not_null<std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        intrp::RegularGrid<VolumeDim>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>*>
        interpolator_buffer,
    const gsl::not_null<std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>*>
        modified_neighbor_solution_buffer,
    const gsl::not_null<db::item_type<Tag>*> tensor,
    const double neighbor_linear_weight, const Mesh<VolumeDim>& mesh,
    const Element<VolumeDim>& element,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_data) noexcept {
  for (size_t tensor_storage_index = 0; tensor_storage_index < tensor->size();
       ++tensor_storage_index) {
    simple_weno_impl<Tag>(interpolator_buffer,
                          modified_neighbor_solution_buffer, tensor,
                          neighbor_linear_weight, tensor_storage_index, mesh,
                          element, neighbor_data);
  }
}

}  // namespace Limiters::Weno_detail
