// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "ErrorHandling/Error.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Importers/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace importers {
/// Actions related to importers
namespace Actions {

/*!
 * \brief Invoked on the `importers::VolumeDataReader` component to store the
 * registered data.
 *
 * The `importers::Actions::RegisterWithVolumeDataReader` action, which is
 * performed on each element of an array parallel component, invokes this action
 * on the `importers::VolumeDataReader` component.
 */
struct RegisterElementWithSelf {
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      typename ArrayIndex, typename DataBox = db::DataBox<DbTagsList>,
      Requires<db::tag_is_retrievable_v<Tags::RegisteredElements, DataBox>> =
          nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ArrayComponentId& array_component_id,
                    const std::string& grid_name) noexcept {
    db::mutate<Tags::RegisteredElements>(
        make_not_null(&box),
        [&array_component_id, &grid_name](
            const gsl::not_null<db::item_type<Tags::RegisteredElements>*>
                registered_elements) noexcept {
          (*registered_elements)[array_component_id] = grid_name;
        });
  }
};

}  // namespace Actions

/// Threaded actions related to importers
namespace ThreadedActions {

/*!
 * \brief Read a volume data file and distribute the data to the registered
 * elements.
 *
 * This action can be invoked on the `importers::VolumeDataReader` component
 * once all elements have been registered with it. It opens the data file, reads
 * the data for each registered element and calls the `CallbackAction` on each
 * element providing the data.
 *
 * - The `ImporterOptionsGroup` parameter specifies the \ref OptionGroupsGroup
 * "options group" in the input file that provides the following run-time
 * options:
 *   - `importers::OptionTags::FileName`
 *   - `importers::OptionTags::Subgroup`
 *   - `importers::OptionTags::ObservationValue`
 * - The `FieldTagsList` parameter specifies a typelist of tensor tags that
 * are read from the file and provided to each element. It is assumed that the
 * tensor data is stored in datasets named `db::tag_name<Tag>() + suffix`, where
 * the `suffix` is empty for scalars or `"_"` followed by the
 * `Tensor::component_name` for each independent tensor component.
 * - The `CallbackAction` is invoked on each registered element of the
 * `CallbackComponent` with a
 * `tuples::tagged_tuple_from_typelist<FieldTagsList>` containing the
 * tensor data for that element. Use `Actions::SetData` to write the data
 * directly into their respective tags in the DataBox. The `CallbackComponent`
 * must the the same that was encoded into the `observers::ArrayComponentId`
 * used to register the elements.
 */
template <typename ImporterOptionsGroup, typename FieldTagsList,
          typename CallbackAction, typename CallbackComponent>
struct ReadVolumeData {
  using const_global_cache_tags =
      tmpl::list<Tags::FileName<ImporterOptionsGroup>,
                 Tags::Subgroup<ImporterOptionsGroup>,
                 Tags::ObservationValue<ImporterOptionsGroup>>;

  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename ArrayIndex,
            Requires<db::tag_is_retrievable_v<Tags::RegisteredElements,
                                              DataBox>> = nullptr>
  static void apply(DataBox& box,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<CmiNodeLock*> node_lock) noexcept {
    Parallel::lock(node_lock);
    {
      // The scoping is to close the file before unlocking
      h5::H5File<h5::AccessType::ReadOnly> h5file(
          Parallel::get<Tags::FileName<ImporterOptionsGroup>>(cache));
      constexpr size_t version_number = 0;
      const auto& volume_file = h5file.get<h5::VolumeData>(
          "/" + Parallel::get<Tags::Subgroup<ImporterOptionsGroup>>(cache),
          version_number);
      const auto observation_id = volume_file.find_observation_id(
          Parallel::get<Tags::ObservationValue<ImporterOptionsGroup>>(cache));
      // Read the tensor data for all elements at once, since that's how it's
      // stored in the file
      tuples::tagged_tuple_from_typelist<FieldTagsList> all_tensor_data{};
      tmpl::for_each<FieldTagsList>([
        &all_tensor_data, &volume_file, &observation_id
      ](auto field_tag_v) noexcept {
        using field_tag = tmpl::type_from<decltype(field_tag_v)>;
        auto& tensor_data = get<field_tag>(all_tensor_data);
        for (size_t i = 0; i < tensor_data.size(); i++) {
          tensor_data[i] = volume_file.get_tensor_component(
              observation_id,
              db::tag_name<field_tag>() + tensor_data.component_suffix(
                                              tensor_data.get_tensor_index(i)));
        }
      });
      // Retrieve the information needed to reconstruct which element the data
      // belongs to
      const auto all_grid_names = volume_file.get_grid_names(observation_id);
      const auto all_extents = volume_file.get_extents(observation_id);
      // Distribute the tensor data to the registered elements
      for (auto& element_and_name : get<Tags::RegisteredElements>(box)) {
        const CkArrayIndex& raw_element_index =
            element_and_name.first.array_index();
        // Check if the parallel component of the registered element matches the
        // callback, because it's possible that elements from other components
        // with the same index are also registered.
        // Since the way the component is encoded in `ArrayComponentId` is
        // private to that class, we construct one and compare.
        if (element_and_name.first !=
            observers::ArrayComponentId(
                std::add_pointer_t<CallbackComponent>{nullptr},
                raw_element_index)) {
          continue;
        }
        // Find the data offset that corresponds to this element
        const auto element_data_offset_and_length =
            h5::offset_and_length_for_grid(element_and_name.second,
                                           all_grid_names, all_extents);
        // Extract this element's data from the read-in dataset
        tuples::tagged_tuple_from_typelist<FieldTagsList> element_data{};
        tmpl::for_each<FieldTagsList>([
          &element_data, &element_data_offset_and_length, &all_tensor_data
        ](auto field_tag_v) noexcept {
          using field_tag = tmpl::type_from<decltype(field_tag_v)>;
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
        // Pass the data to the element in a simple action
        const auto element_index =
            Parallel::ArrayIndex<typename CallbackComponent::array_index>(
                raw_element_index)
                .get_index();
        Parallel::simple_action<CallbackAction>(
            Parallel::get_parallel_component<CallbackComponent>(
                cache)[element_index],
            std::move(element_data));
      }
    }
    Parallel::unlock(node_lock);
  }
};

}  // namespace ThreadedActions
}  // namespace importers
