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
void interpolate_volume_data(
    const gsl::not_null<ah::Storage::Iteration<Fr>*> current_iteration_storage,
    const gsl::not_null<
        std::unordered_map<ElementId<3>, ah::Storage::VolumeVariables<Fr>>*>
        all_volume_variables) {
  std::vector<ElementId<3>> element_ids;
  element_ids.reserve(all_volume_variables->size());
  for (const auto& [element_id, volume_vars] : *all_volume_variables) {
    (void)volume_vars;  // Avoid clang-tidy warning
    if (not current_iteration_storage->interpolation_is_done_for_these_elements
                .contains(element_id)) {
      element_ids.push_back(element_id);
      current_iteration_storage->interpolation_is_done_for_these_elements
          .insert(element_id);
    }
  }

  ASSERT(current_iteration_storage->block_coord_holders.has_value(),
         "Block logical coordinates of horizon have not been set!");

  const auto& block_coord_holders =
      current_iteration_storage->block_coord_holders.value();
  const auto element_coord_holders =
      element_logical_coordinates(element_ids, block_coord_holders);

  for (const auto& [element_id, element_coord_holder] : element_coord_holders) {
    const auto& offsets = element_coord_holder.offsets;
    auto& volume_vars_storage = all_volume_variables->at(element_id);
    auto& interpolated_vars = current_iteration_storage->interpolated_vars;

    // Only fill once
    const size_t expected_num_points = block_coord_holders.size();
    if (interpolated_vars.number_of_grid_points() != expected_num_points) {
      interpolated_vars.initialize(expected_num_points);
    }

    const intrp::Irregular<3> interpolator(
        volume_vars_storage.mesh, element_coord_holder.element_logical_coords);

    // Vars interpolated to points within this element
    auto local_interpolated_vars = interpolator.interpolate(
        volume_vars_storage.vars_to_interpolate_to_target);

    // Loop over each tensor
    tmpl::for_each<ah::vars_to_interpolate_to_target<3, Fr>>(
        [&]<typename Tag>(tmpl::type_<Tag>) {
          auto& individual_interpolated_var = get<Tag>(interpolated_vars);
          auto& local_individual_interpolated_var =
              get<Tag>(local_interpolated_vars);

          // Loop over components of tensor
          for (size_t i = 0; i < individual_interpolated_var.size(); i++) {
            // Loop over number of points that were interpolated
            for (size_t j = 0; j < offsets.size(); j++) {
              // Copy the local interpolated value into the correct
              // position in the overall tensor
              individual_interpolated_var[i][offsets[j]] =
                  local_individual_interpolated_var[i][j];
            }
          }
        });

    for (const size_t offset : offsets) {
      current_iteration_storage->indices_interpolated_to_thus_far.insert(
          offset);
    }
  }
}

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                         \
  template void interpolate_volume_data(                             \
      const gsl::not_null<ah::Storage::Iteration<FRAME(data)>*>      \
          current_iteration_storage,                                 \
      const gsl::not_null<std::unordered_map<                        \
          ElementId<3>, ah::Storage::VolumeVariables<FRAME(data)>>*> \
          all_volume_variables);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Inertial, Frame::Distorted, Frame::Grid))

#undef INSTANTIATE
#undef FRAME
}  // namespace ah
