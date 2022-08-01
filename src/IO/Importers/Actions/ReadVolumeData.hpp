// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Importers/ObservationSelector.hpp"
#include "IO/Importers/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace importers {
template <typename Metavariables>
struct ElementDataReader;
namespace Actions {
template <typename ImporterOptionsGroup, typename FieldTagsList,
          typename ReceiveComponent>
struct ReadAllVolumeDataAndDistribute;
}  // namespace Actions
}  // namespace importers
/// \endcond

namespace importers::Tags {
/*!
 * \brief Indicates an available tensor field is selected for importing, along
 * with the name of the dataset in the volume data file.
 *
 * Set the value to a dataset name to import the `FieldTag` from that dataset,
 * or to `std::nullopt` to skip importing the `FieldTag`. The dataset name
 * excludes tensor component suffixes like "_x" or "_xy". These suffixes will be
 * added automatically. A sensible value for the dataset name is often
 * `db::tag_name<FieldTag>()`, but the user should generally be given the
 * opportunity to set the dataset name in the input file.
 */
template <typename FieldTag>
struct Selected : db::SimpleTag {
  using type = std::optional<std::string>;
};
}  // namespace importers::Tags

namespace importers::Actions {

/*!
 * \brief Read a volume data file and distribute the data to all registered
 * elements.
 *
 * Invoke this action on the elements of an array parallel component to dispatch
 * reading the volume data file specified by the options in
 * `ImporterOptionsGroup`. The tensors in `FieldTagsList` will be loaded from
 * the file and distributed to all elements that have previously registered. Use
 * `importers::Actions::RegisterWithElementDataReader` to register the elements
 * of the array parallel component in a previous phase.
 *
 * Note that the volume data file will only be read once per node, triggered by
 * the first element that invokes this action. All subsequent invocations of
 * this action on the node will do nothing. See
 * `importers::Actions::ReadAllVolumeDataAndDistribute` for details.
 *
 * The data is distributed to the elements using `Parallel::receive_data`. The
 * elements can monitor `importers::Tags::VolumeData` in their inbox to wait for
 * the data and process it once it's available. We provide the action
 * `importers::Actions::ReceiveVolumeData` that waits for the data and moves it
 * directly into the DataBox. You can also implement a specialized action that
 * might verify and post-process the data before populating the DataBox.
 *
 * \see Dev guide on \ref dev_guide_importing
 */
template <typename ImporterOptionsGroup, typename FieldTagsList>
struct ReadVolumeData {
  using const_global_cache_tags =
      tmpl::list<Tags::FileGlob<ImporterOptionsGroup>,
                 Tags::Subgroup<ImporterOptionsGroup>,
                 Tags::ObservationValue<ImporterOptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Not using `ckLocalBranch` here to make sure the simple action invocation
    // is asynchronous.
    auto& reader_component = Parallel::get_parallel_component<
        importers::ElementDataReader<Metavariables>>(cache);
    Parallel::simple_action<importers::Actions::ReadAllVolumeDataAndDistribute<
        ImporterOptionsGroup, FieldTagsList, ParallelComponent>>(
        reader_component);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Read a volume data file and distribute the data to all registered
 * elements.
 *
 * This action can be invoked on the `importers::ElementDataReader` component
 * once all elements have been registered with it. It opens the data file, reads
 * the data for each registered element and uses `Parallel::receive_data` to
 * distribute the data to the elements. The elements can monitor
 * `importers::Tags::VolumeData` in their inbox to wait for the data and process
 * it once it's available. You can use `importers::Actions::ReceiveVolumeData`
 * to wait for the data and move it directly into the DataBox, or implement a
 * specialized action that might verify and post-process the data.
 *
 * Note that instead of invoking this action directly on the
 * `importers::ElementDataReader` component you can invoke the iterable action
 * `importers::Actions::ReadVolumeData` on the elements of an array parallel
 * component.
 *
 * - The `ImporterOptionsGroup` parameter specifies the \ref OptionGroupsGroup
 * "options group" in the input file that provides the following run-time
 * options:
 *   - `importers::OptionTags::FileGlob`
 *   - `importers::OptionTags::Subgroup`
 *   - `importers::OptionTags::ObservationValue`
 * - The `FieldTagsList` parameter specifies a typelist of tensor tags that
 * can be read from the file and provided to each element. The subset of tensors
 * that will actually be read and distributed can be selected at runtime with
 * the `selected_fields` argument that is passed to this simple action. See
 * importers::Tags::Selected for details. By default, all tensors in the
 * `FieldTagsList` are selected, and read from datasets named
 * `db::tag_name<Tag>() + suffix`, where the `suffix` is empty for scalars, or
 * `"_"` followed by the `Tensor::component_name` for each independent tensor
 * component.
 * - `Parallel::receive_data` is invoked on each registered element of the
 * `ReceiveComponent` to populate `importers::Tags::VolumeData` in the element's
 * inbox with a `tuples::tagged_tuple_from_typelist<FieldTagsList>` containing
 * the tensor data for that element. The `ReceiveComponent` must the the same
 * that was encoded into the `observers::ArrayComponentId` used to register the
 * elements.
 *
 * \see Dev guide on \ref dev_guide_importing
 */
template <typename ImporterOptionsGroup, typename FieldTagsList,
          typename ReceiveComponent>
struct ReadAllVolumeDataAndDistribute {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename ArrayIndex>
  static void apply(DataBox& box, Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    tuples::tagged_tuple_from_typelist<
                        db::wrap_tags_in<Tags::Selected, FieldTagsList>>
                        selected_fields = select_all_fields(FieldTagsList{})) {
    // Only read and distribute the volume data once
    // This action will be invoked by `importers::Actions::ReadVolumeData` from
    // every element on the node, but only the first invocation reads the file
    // and distributes the data to all elements. Subsequent invocations do
    // nothing. We use the `ImporterOptionsGroup` that specifies the data file
    // to read in as the identifier for whether or not we have already read the
    // requested data. Doing this at runtime avoids having to collect all
    // data files that will be read in at compile-time to initialize a flag in
    // the DataBox for each of them.
    const auto& has_read_volume_data =
        db::get<Tags::ElementDataAlreadyRead>(box);
    const auto volume_data_id = pretty_type::get_name<ImporterOptionsGroup>();
    if (has_read_volume_data.find(volume_data_id) !=
        has_read_volume_data.end()) {
      return;
    }
    db::mutate<Tags::ElementDataAlreadyRead>(
        make_not_null(&box),
        [&volume_data_id](const gsl::not_null<std::unordered_set<std::string>*>
                              local_has_read_volume_data) {
          local_has_read_volume_data->insert(std::move(volume_data_id));
        });

    // Resolve the file glob
    const std::string& file_glob =
        Parallel::get<Tags::FileGlob<ImporterOptionsGroup>>(cache);
    const std::vector<std::string> file_paths = file_system::glob(file_glob);

    // Open every file in turn
    std::optional<size_t> prev_observation_id{};
    for (const std::string& file_name : file_paths) {
      // Open the volume data file
      h5::H5File<h5::AccessType::ReadOnly> h5file(file_name);
      constexpr size_t version_number = 0;
      const auto& volume_file = h5file.get<h5::VolumeData>(
          "/" + Parallel::get<Tags::Subgroup<ImporterOptionsGroup>>(cache),
          version_number);

      // Select observation ID
      const size_t observation_id = std::visit(
          Overloader{
              [&volume_file](const double local_obs_value) {
                return volume_file.find_observation_id(local_obs_value);
              },
              [&volume_file](const ObservationSelector local_obs_selector) {
                const std::vector<size_t> all_observation_ids =
                    volume_file.list_observation_ids();
                switch (local_obs_selector) {
                  case ObservationSelector::First:
                    return all_observation_ids.front();
                  case ObservationSelector::Last:
                    return all_observation_ids.back();
                  default:
                    ERROR("Unknown importers::ObservationSelector: "
                          << local_obs_selector);
                }
              }},
          Parallel::get<Tags::ObservationValue<ImporterOptionsGroup>>(cache));
      if (prev_observation_id.has_value() and
          prev_observation_id.value() != observation_id) {
        ERROR("Inconsistent selection of observation ID in file "
              << file_name
              << ". Make sure all files select the same observation ID.");
      }
      prev_observation_id = observation_id;

      // Read the tensor data for all elements at once, since that's how it's
      // stored in the file
      tuples::tagged_tuple_from_typelist<FieldTagsList> all_tensor_data{};
      tmpl::for_each<FieldTagsList>([&all_tensor_data, &volume_file,
                                     &observation_id,
                                     &selected_fields](auto field_tag_v) {
        using field_tag = tmpl::type_from<decltype(field_tag_v)>;
        const auto& selection = get<Tags::Selected<field_tag>>(selected_fields);
        if (not selection.has_value()) {
          return;
        }
        auto& tensor_data = get<field_tag>(all_tensor_data);
        for (size_t i = 0; i < tensor_data.size(); i++) {
          tensor_data[i] = std::get<DataVector>(
              volume_file
                  .get_tensor_component(
                      observation_id,
                      selection.value() + tensor_data.component_suffix(
                                              tensor_data.get_tensor_index(i)))
                  .data);
        }
      });
      // Retrieve the information needed to reconstruct which element the data
      // belongs to
      const auto all_grid_names = volume_file.get_grid_names(observation_id);
      const auto all_extents = volume_file.get_extents(observation_id);
      // Distribute the tensor data to the registered elements
      for (auto& [element_array_component_id, grid_name] :
           get<Tags::RegisteredElements>(box)) {
        const CkArrayIndex& raw_element_index =
            element_array_component_id.array_index();
        // Check if the parallel component of the registered element matches the
        // callback, because it's possible that elements from other components
        // with the same index are also registered.
        // Since the way the component is encoded in `ArrayComponentId` is
        // private to that class, we construct one and compare.
        if (element_array_component_id !=
            observers::ArrayComponentId(
                std::add_pointer_t<ReceiveComponent>{nullptr},
                raw_element_index)) {
          continue;
        }
        // Proceed with the registered element only if it's included in the
        // volume file. It's possible that the volume file only contains data
        // for a subset of elements, e.g., when each node of a simulation wrote
        // volume data for its elements to a separate file.
        if (std::find(all_grid_names.begin(), all_grid_names.end(),
                      grid_name) == all_grid_names.end()) {
          continue;
        }
        // Find the data offset that corresponds to this element
        const auto element_data_offset_and_length =
            h5::offset_and_length_for_grid(grid_name, all_grid_names,
                                           all_extents);
        // Extract this element's data from the read-in dataset
        tuples::tagged_tuple_from_typelist<FieldTagsList> element_data{};
        tmpl::for_each<FieldTagsList>([&element_data,
                                       &element_data_offset_and_length,
                                       &all_tensor_data,
                                       &selected_fields](auto field_tag_v) {
          using field_tag = tmpl::type_from<decltype(field_tag_v)>;
          const auto& selection =
              get<Tags::Selected<field_tag>>(selected_fields);
          if (not selection.has_value()) {
            return;
          }
          auto& element_tensor_data = get<field_tag>(element_data);
          // Iterate independent components of the tensor
          for (size_t i = 0; i < element_tensor_data.size(); i++) {
            const DataVector& data_tensor_component =
                get<field_tag>(all_tensor_data)[i];
            DataVector element_tensor_component{
                element_data_offset_and_length.second};
            // Retrieve data from slice of the contigious dataset
            for (size_t j = 0; j < element_tensor_component.size(); j++) {
              element_tensor_component[j] =
                  data_tensor_component[element_data_offset_and_length.first +
                                        j];
            }
            element_tensor_data[i] = element_tensor_component;
          }
        });
        // Pass the data to the element
        const auto element_index =
            Parallel::ArrayIndex<typename ReceiveComponent::array_index>(
                raw_element_index)
                .get_index();
        Parallel::receive_data<
            Tags::VolumeData<ImporterOptionsGroup, FieldTagsList>>(
            Parallel::get_parallel_component<ReceiveComponent>(
                cache)[element_index],
            // Using `0` for the temporal ID since we only read the volume data
            // once, so there's no need to keep track of the temporal ID.
            0_st, std::move(element_data));
      }
    }
  }

 private:
  template <typename... LocalFieldTags>
  static tuples::TaggedTuple<Tags::Selected<LocalFieldTags>...>
  select_all_fields(tmpl::list<LocalFieldTags...> /*meta*/) {
    return {db::tag_name<LocalFieldTags>()...};
  }
};

}  // namespace importers::Actions
