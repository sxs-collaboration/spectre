// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Importers/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
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
      tmpl::list<Tags::FileName<ImporterOptionsGroup>,
                 Tags::Subgroup<ImporterOptionsGroup>,
                 Tags::ObservationValue<ImporterOptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    auto& local_reader_component =
        *Parallel::get_parallel_component<
             importers::ElementDataReader<Metavariables>>(cache)
             .ckLocalBranch();
    Parallel::simple_action<importers::Actions::ReadAllVolumeDataAndDistribute<
        ImporterOptionsGroup, FieldTagsList, ParallelComponent>>(
        local_reader_component);
    return {std::move(box)};
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
 *   - `importers::OptionTags::FileName`
 *   - `importers::OptionTags::Subgroup`
 *   - `importers::OptionTags::ObservationValue`
 * - The `FieldTagsList` parameter specifies a typelist of tensor tags that
 * are read from the file and provided to each element. It is assumed that the
 * tensor data is stored in datasets named `db::tag_name<Tag>() + suffix`, where
 * the `suffix` is empty for scalars or `"_"` followed by the
 * `Tensor::component_name` for each independent tensor component.
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
  template <
      typename ParallelComponent, typename DataBox, typename Metavariables,
      typename ArrayIndex,
      Requires<db::tag_is_retrievable_v<Tags::RegisteredElements, DataBox> and
               db::tag_is_retrievable_v<Tags::ElementDataAlreadyRead,
                                        DataBox>> = nullptr>
  static void apply(DataBox& box, Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) noexcept {
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
                              local_has_read_volume_data) noexcept {
          local_has_read_volume_data->insert(std::move(volume_data_id));
        });

    // Open the volume data file
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
    tmpl::for_each<FieldTagsList>([&all_tensor_data, &volume_file,
                                   &observation_id](auto field_tag_v) noexcept {
      using field_tag = tmpl::type_from<decltype(field_tag_v)>;
      auto& tensor_data = get<field_tag>(all_tensor_data);
      for (size_t i = 0; i < tensor_data.size(); i++) {
        tensor_data[i] = volume_file.get_tensor_component(
            observation_id,
            db::tag_name<field_tag>() +
                tensor_data.component_suffix(tensor_data.get_tensor_index(i)));
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
              std::add_pointer_t<ReceiveComponent>{nullptr},
              raw_element_index)) {
        continue;
      }
      // Find the data offset that corresponds to this element
      const auto element_data_offset_and_length =
          h5::offset_and_length_for_grid(element_and_name.second,
                                         all_grid_names, all_extents);
      // Extract this element's data from the read-in dataset
      tuples::tagged_tuple_from_typelist<FieldTagsList> element_data{};
      tmpl::for_each<FieldTagsList>([&element_data,
                                     &element_data_offset_and_length,
                                     &all_tensor_data](
                                        auto field_tag_v) noexcept {
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
                data_tensor_component[element_data_offset_and_length.first + j];
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
};

}  // namespace importers::Actions
