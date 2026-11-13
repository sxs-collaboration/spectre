// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "IO/Observer/VolumeActions.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/TypeTraits.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver::Schwarz::detail {

template <typename OptionsGroup, typename ArraySectionIdTag>
struct RegisterWithVolumeObserver {
  template <typename ParallelComponent, typename DbTagsList,
            typename ArrayIndex>
  static std::pair<observers::TypeOfObservation, observers::ObservationKey>
  register_info(const db::DataBox<DbTagsList>& box,
                const ArrayIndex& /*array_index*/) {
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    const std::string subfile_path = "/" + pretty_type::name<OptionsGroup>() +
                                     section_observation_key.value_or("");
    return {observers::TypeOfObservation::Volume,
            observers::ObservationKey(subfile_path)};
  }
};

// Contribute the volume data recorded in the other actions to the observer at
// the end of a step.
template <typename FieldsTag, typename OptionsGroup, typename SubdomainOperator,
          typename ArraySectionIdTag>
struct ObserveVolumeData {
 private:
  using fields_tag = FieldsTag;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;
  static constexpr size_t Dim = SubdomainOperator::volume_dim;
  using SubdomainData =
      ElementCenteredSubdomainData<Dim, typename residual_tag::tags_list>;
  using volume_data_tag =
      Tags::VolumeDataForOutput<SubdomainData, OptionsGroup>;
  using VolumeDataVars = typename volume_data_tag::type::ElementData;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if (not db::get<LinearSolver::Tags::OutputVolumeData<OptionsGroup>>(box)) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    const auto& volume_data = db::get<volume_data_tag>(box);
    const auto& observation_id =
        db::get<LinearSolver::Tags::ObservationId<OptionsGroup>>(box);
    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    const auto& inertial_coords =
        db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    // Collect tensor components to observe
    std::vector<TensorComponent> components{};
    const auto record_tensor_components = [&components](
                                              const auto tensor_tag_v,
                                              const auto& tensor,
                                              const std::string& suffix = "") {
      using tensor_tag = std::decay_t<decltype(tensor_tag_v)>;
      using TensorType = std::decay_t<decltype(tensor)>;
      using VectorType = typename TensorType::type;
      using ValueType = typename VectorType::value_type;
      for (size_t i = 0; i < tensor.size(); ++i) {
        const std::string component_name =
            db::tag_name<tensor_tag>() + suffix + tensor.component_suffix(i);
        if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
          components.emplace_back("Re(" + component_name + ")",
                                  real(tensor[i]));
          components.emplace_back("Im(" + component_name + ")",
                                  imag(tensor[i]));
        } else {
          components.emplace_back(component_name, tensor[i]);
        }
      }
    };
    record_tensor_components(domain::Tags::Coordinates<Dim, Frame::Inertial>{},
                             inertial_coords);
    const auto& all_intruding_extents =
        db::get<Tags::IntrudingExtents<Dim, OptionsGroup>>(box);
    const VolumeDataVars zero_vars{mesh.number_of_grid_points(), 0.};
    tmpl::for_each<typename VolumeDataVars::tags_list>(
        [&volume_data, &record_tensor_components, &mesh, &all_intruding_extents,
         &zero_vars, &element, &element_id](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          record_tensor_components(tag{}, get<tag>(volume_data.element_data),
                                   "_Center");
          for (const auto& direction : element.all_boundaries()) {
            const auto direction_predicate =
                [&direction](const auto& overlap_data) {
                  return overlap_data.first.direction() == direction;
                };
            const auto num_overlaps =
                alg::count_if(volume_data.overlap_data, direction_predicate);
            if (num_overlaps == 0) {
              // No overlap data for this direction (e.g. external boundary),
              // record zero
              record_tensor_components(tag{}, get<tag>(zero_vars),
                                       "_Overlap" + get_output(direction));
            } else if (num_overlaps == 1) {
              // Overlap data from a single neighbor, record it
              const auto& overlap_data =
                  alg::find_if(volume_data.overlap_data, direction_predicate)
                      ->second;
              const auto& intruding_extents =
                  gsl::at(all_intruding_extents, direction.dimension());
              record_tensor_components(
                  tag{},
                  get<tag>(extended_overlap_data(overlap_data, mesh.extents(),
                                                 intruding_extents, direction)),
                  "_Overlap" + get_output(direction));
            } else {
              ERROR("Multiple neighbors ("
                    << num_overlaps << ") overlap with element " << element_id
                    << " in direction " << direction
                    << ", but we can record only one for volume data output.");
            }
          }
        });

    // Contribute tensor components to observer
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    const std::string subfile_path = "/" + pretty_type::name<OptionsGroup>() +
                                     section_observation_key.value_or("");
    observers::contribute_volume_data<
        not Parallel::is_nodegroup_v<ParallelComponent>>(
        cache, observers::ObservationId(observation_id, subfile_path),
        subfile_path,
        Parallel::make_array_component_id<ParallelComponent>(element_id),
        ElementVolumeData{element_id, std::move(components), mesh});

    // Increment observation ID
    db::mutate<LinearSolver::Tags::ObservationId<OptionsGroup>>(
        [](const auto local_observation_id) { ++(*local_observation_id); },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace LinearSolver::Schwarz::detail
