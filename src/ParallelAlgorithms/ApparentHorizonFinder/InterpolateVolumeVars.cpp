// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/ApparentHorizonFinder/InterpolateVolumeVars.hpp"

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include "DataStructures/Variables.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/ComputeVarsToInterpolateToTarget.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/HorizonAliases.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/Storage.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ah {
template <typename Fr>
bool interpolate_volume_data(
    const gsl::not_null<ah::Storage::Iteration<Fr>*> current_iteration_storage,
    const ah::Storage::VolumeVariables<Fr>& volume_vars_storage,
    const ElementId<3>& element_id) {
  ASSERT(current_iteration_storage->block_coord_holders.has_value(),
         "Block logical coordinates of horizon have not been set!");

  const auto& block_coord_holders =
      current_iteration_storage->block_coord_holders.value();
  auto& interpolated_vars = current_iteration_storage->interpolated_vars;
  auto& indices_interpolated_to_thus_far =
      current_iteration_storage->indices_interpolated_to_thus_far;
  auto& offsets =
      current_iteration_storage->offsets_of_newly_interpolated_points;
  auto& x_element_logical =
      current_iteration_storage->x_element_logical_of_newly_interpolated_points;

  // Initialize interpolated_vars if needed and reserve memory
  const size_t expected_num_points = block_coord_holders.size();
  if (interpolated_vars.number_of_grid_points() != expected_num_points) {
    interpolated_vars.initialize(expected_num_points);
    offsets.reserve(expected_num_points);
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(x_element_logical, d).reserve(expected_num_points);
    }
  }
  if (indices_interpolated_to_thus_far.size() != expected_num_points) {
    indices_interpolated_to_thus_far.resize(expected_num_points, false);
  }

  // Find points in this element
  offsets.clear();
  for (size_t d = 0; d < 3; ++d) {
    gsl::at(x_element_logical, d).clear();
  }
  for (size_t p = 0; p < expected_num_points; ++p) {
    if (indices_interpolated_to_thus_far[p]) {
      // Skip points that have already been interpolated to
      continue;
    }
    if (block_coord_holders[p]->id.get_index() != element_id.block_id()) {
      // Skip points that are not in this block
      continue;
    }
    const auto element_logical_coords =
        element_logical_coordinates(block_coord_holders[p]->data, element_id);
    if (not element_logical_coords.has_value()) {
      // Skip points that are not in this element
      continue;
    }
    // Collect points in this element
    offsets.push_back(p);
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(x_element_logical, d).push_back(element_logical_coords->get(d));
    }
  }  // for block_logical_coords

  // Return early if no points are in this element
  if (offsets.empty()) {
    return false;
  }

  // Interpolate!
  // Use non-owning wrappers around memory buffers to avoid allocations
  tnsr::I<DataVector, 3, Frame::ElementLogical> element_logical_coords{};
  for (size_t d = 0; d < 3; ++d) {
    element_logical_coords.get(d).set_data_ref(
        gsl::at(x_element_logical, d).data(),
        gsl::at(x_element_logical, d).size());
  }
  const intrp::Irregular<3> interpolator(volume_vars_storage.mesh,
                                         element_logical_coords);
  auto& interpolated_vars_buffer =
      current_iteration_storage->newly_interpolated_vars_buffer;
  Variables<ah::vars_to_interpolate_to_target<3, Fr>> local_interpolated_vars{};
  constexpr size_t num_components =
      Variables<ah::vars_to_interpolate_to_target<3, Fr>>::
          number_of_independent_components;
  interpolated_vars_buffer.resize(offsets.size() * num_components);
  local_interpolated_vars.set_data_ref(interpolated_vars_buffer.data(),
                                       interpolated_vars_buffer.size());
  interpolator.interpolate(make_not_null(&local_interpolated_vars),
                           volume_vars_storage.vars_to_interpolate_to_target);

  // Copy local results into overall result
  // Loop over each tensor
  tmpl::for_each<ah::vars_to_interpolate_to_target<3, Fr>>(
      [&]<typename Tag>(tmpl::type_<Tag>) {
        auto& individual_interpolated_var = get<Tag>(interpolated_vars);
        const auto& local_individual_interpolated_var =
            get<Tag>(local_interpolated_vars);

        // Loop over components of tensor
        for (size_t i = 0; i < individual_interpolated_var.size(); i++) {
          // Loop over number of points that were interpolated
          for (size_t p = 0; p < offsets.size(); ++p) {
            // Copy the local interpolated value into the correct
            // position in the overall tensor
            individual_interpolated_var[i][offsets[p]] =
                local_individual_interpolated_var[i][p];
            indices_interpolated_to_thus_far[offsets[p]] = true;
          }
        }
      });
  return true;
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                \
  template bool interpolate_volume_data(                                    \
      const gsl::not_null<ah::Storage::Iteration<FRAME(data)>*>             \
          current_iteration_storage,                                        \
      const ah::Storage::VolumeVariables<FRAME(data)>& volume_vars_storage, \
      const ElementId<3>& element_id);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Inertial, Frame::Distorted, Frame::Grid))

#undef INSTANTIATE
#undef FRAME
}  // namespace ah
