// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace LinearSolver::multigrid::detail {

template <typename OptionsGroup>
struct RegisterWithVolumeObserver {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationKey>
  register_info(const db::DataBox<DbTagsList>& box,
                const ArrayIndex& /*array_index*/) noexcept {
    const std::string& level_observation_key =
        *db::get<observers::Tags::ObservationKey<Tags::MultigridLevel>>(box);
    const std::string subfile_path =
        "/" + Options::name<OptionsGroup>() + level_observation_key;
    return {observers::TypeOfObservation::Volume,
            observers::ObservationKey(subfile_path)};
  }
};

// Contribute the volume data recorded in the other actions to the observer at
// the end of a step.
template <typename FieldsTag, typename OptionsGroup, typename SourceTag>
struct ObserveVolumeData {
 private:
  using volume_data_tag = Tags::VolumeDataForOutput<OptionsGroup, FieldsTag>;
  using VolumeDataVars = typename volume_data_tag::type;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    if (not db::get<Tags::OutputVolumeData<OptionsGroup>>(box)) {
      return {std::move(box)};
    }
    const auto& volume_data = db::get<volume_data_tag>(box);
    const auto& observation_id =
        db::get<Tags::ObservationId<OptionsGroup>>(box);
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& inertial_coords =
        db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const std::string element_name = MakeString{} << element_id << '/';

    // Collect tensor components to observe
    std::vector<TensorComponent> components{};
    components.reserve(inertial_coords.size() +
                       VolumeDataVars::number_of_independent_components);
    const auto record_tensor_components = [&components, &element_name](
                                              const auto tensor_tag_v,
                                              const auto& tensor) noexcept {
      using tensor_tag = std::decay_t<decltype(tensor_tag_v)>;
      for (size_t i = 0; i < tensor.size(); ++i) {
        components.emplace_back(element_name + db::tag_name<tensor_tag>() +
                                    tensor.component_suffix(i),
                                tensor[i]);
      }
    };
    record_tensor_components(domain::Tags::Coordinates<Dim, Frame::Inertial>{},
                             inertial_coords);
    tmpl::for_each<typename VolumeDataVars::tags_list>(
        [&volume_data, &record_tensor_components](auto tag_v) noexcept {
          using tag = tmpl::type_from<decltype(tag_v)>;
          record_tensor_components(tag{}, get<tag>(volume_data));
        });

    // Contribute tensor components to observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    const auto& level_observation_key =
        *db::get<observers::Tags::ObservationKey<Tags::MultigridLevel>>(box);
    const std::string subfile_path =
        "/" + Options::name<OptionsGroup>() + level_observation_key;
    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        local_observer, observers::ObservationId(observation_id, subfile_path),
        subfile_path,
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementId<Dim>>(element_id)),
        std::move(components), mesh.extents(), mesh.basis(), mesh.quadrature());

    // Increment observation ID
    db::mutate<Tags::ObservationId<OptionsGroup>>(
        make_not_null(&box), [](const auto local_observation_id) noexcept {
          ++(*local_observation_id);
        });
    return {std::move(box)};
  }
};

}  // namespace LinearSolver::multigrid::detail
