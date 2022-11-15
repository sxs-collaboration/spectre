// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/Actions/ObserverRegistration.hpp"
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

  const auto& mesh = info.mesh;
  const auto& index_extents = mesh.extents();
  const auto& array_bases = mesh.basis();
  const auto& array_quadratures = mesh.quadrature();
  std::vector<size_t> extents(index_extents.begin(), index_extents.end());
  std::vector<Spectral::Basis> bases(array_bases.begin(), array_bases.end());
  std::vector<Spectral::Quadrature> quadratures(array_quadratures.begin(),
                                                array_quadratures.end());
  std::string element_name = MakeString{} << element_id;

  return ElementVolumeData{std::move(extents), std::move(components),
                           std::move(bases), std::move(quadratures),
                           std::move(element_name)};
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
  template <typename DbTagList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& observer_writer = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);
    auto& my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
    const auto& file_prefix =
        Parallel::get<observers::Tags::VolumeFileName>(cache);
    const size_t my_node =
        Parallel::my_node<size_t>(*Parallel::local_branch(my_proxy));
    const std::string filename{file_prefix + std::to_string(my_node)};

    tmpl::for_each<AllTemporalIds>([&box, &observer_writer, &my_node,
                                    &filename](auto temporal_id_v) {
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

        // To speed up writing, call this on our own node which is guaranteed to
        // exist because...we are on it...
        Parallel::threaded_action<observers::ThreadedActions::WriteVolumeData>(
            observer_writer[my_node], filename, subfile_name,
            observers::ObservationId{
                InterpolationTarget_detail::get_temporal_id_value(temporal_id),
                subfile_name},
            std::move(element_volume_data));
      }
    });

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace intrp::Actions
