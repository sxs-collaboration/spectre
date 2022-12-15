// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Tags.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolationTargetDetail.hpp"
#include "ParallelAlgorithms/Interpolation/Tags.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace intrp::Actions {
namespace detail {
// This requires there to be a type alias named `interpolator_source_vars` in
// the metavariables
template <typename TemporalIdTag, typename Metavariables>
ElementVolumeData construct_element_volume_data(
    const ElementId<Metavariables::volume_dim>& element_id,
    const typename intrp::Tags::VolumeVarsInfo<Metavariables,
                                               TemporalIdTag>::Info& info) {
  std::vector<TensorComponent> components{};

  const auto& all_source_vars = info.source_vars_from_element;
  tmpl::for_each<typename Metavariables::interpolator_source_vars>(
      [&components, &all_source_vars](auto source_var_tag_v) {
        using source_var_tag =
            tmpl::type_from<std::decay_t<decltype(source_var_tag_v)>>;
        const auto& tensor = get<source_var_tag>(all_source_vars);
        for (size_t i = 0; i < tensor.size(); i++) {
          const auto& tensor_component = tensor[i];
          components.emplace_back(
              db::tag_name<source_var_tag>() + tensor.component_suffix(i),
              tensor_component);
        }
      });

  return ElementVolumeData{element_id, std::move(components), info.mesh};
}
}  // namespace detail

/*!
 * \brief Dump all volume data at all temporal IDs stored in the interpolator.
 *
 * All dumped data will be in the usual volume files, but under different
 * subfiles. There will be one subfile for every temporal ID type (as the
 * interpolator can have multiple different temporal ID types from different
 * interpolation targets). The tensors that are dumped are the ones defined in
 * the `interpolator_source_vars` type alias in the Metavariables.
 */
template <typename AllTemporalIds>
struct DumpInterpolatorVolumeData {
  using const_global_cache_tags =
      tmpl::list<intrp::Tags::DumpVolumeDataOnFailure>;

  template <typename DbTagList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (not Parallel::get<intrp::Tags::DumpVolumeDataOnFailure>(cache)) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    auto& observer_writer = *Parallel::local_branch(
        Parallel::get_parallel_component<
            observers::ObserverWriter<Metavariables>>(cache));

    const observers::ArrayComponentId array_component_id{
        std::add_pointer_t<ParallelComponent>{nullptr},
        Parallel::ArrayIndex<std::decay_t<ArrayIndex>>(array_index)};

    tmpl::for_each<AllTemporalIds>([&box, &observer_writer,
                                    &array_component_id](auto temporal_id_v) {
      using temporal_id_t =
          tmpl::type_from<std::decay_t<decltype(temporal_id_v)>>;
      const auto& volume_vars_info =
          db::get<intrp::Tags::VolumeVarsInfo<Metavariables, temporal_id_t>>(
              box);
      const std::string subfile_name{"/InterpolatorVolumeData_"s +
                                     db::tag_name<temporal_id_t>()};

      for (const auto& [temporal_id, info_map] : volume_vars_info) {
        std::vector<ElementVolumeData> element_volume_data{};
        for (const auto& [element_id, info] : info_map) {
          element_volume_data.emplace_back(
              detail::construct_element_volume_data<temporal_id_t,
                                                    Metavariables>(element_id,
                                                                   info));
        }

        Parallel::threaded_action<
            observers::ThreadedActions::ContributeVolumeDataToWriter>(
            observer_writer,
            observers::ObservationId{
                InterpolationTarget_detail::get_temporal_id_value(temporal_id),
                subfile_name},
            array_component_id, subfile_name,
            std::unordered_map<observers::ArrayComponentId,
                               std::vector<ElementVolumeData>>{
                {array_component_id, std::move(element_volume_data)}});
      }
    });

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace intrp::Actions
