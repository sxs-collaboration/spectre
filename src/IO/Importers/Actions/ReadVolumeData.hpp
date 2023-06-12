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
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementLogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Importers/ObservationSelector.hpp"
#include "IO/Importers/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace importers {

/// \cond
template <typename Metavariables>
struct ElementDataReader;
namespace Actions {
template <size_t Dim, typename FieldTagsList, typename ReceiveComponent>
struct ReadAllVolumeDataAndDistribute;
}  // namespace Actions
/// \endcond

namespace Tags {
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
}  // namespace Tags

namespace detail {

// Read the single `tensor_name` from the `volume_file`, taking care of suffixes
// like "_x" etc for its components.
template <typename TensorType>
void read_tensor_data(const gsl::not_null<TensorType*> tensor_data,
                      const std::string& tensor_name,
                      const h5::VolumeData& volume_file,
                      const size_t observation_id) {
  for (size_t i = 0; i < tensor_data->size(); ++i) {
    (*tensor_data)[i] = std::get<DataVector>(
        volume_file
            .get_tensor_component(
                observation_id,
                tensor_name + tensor_data->component_suffix(
                                  tensor_data->get_tensor_index(i)))
            .data);
  }
}

// Read the `selected_fields` from the `volume_file`. Reads the data
// for all elements in the `volume_file` at once. Invoked lazily when data
// for an element in the volume file is needed.
template <typename FieldTagsList>
tuples::tagged_tuple_from_typelist<FieldTagsList> read_tensor_data(
    const h5::VolumeData& volume_file, const size_t observation_id,
    const tuples::tagged_tuple_from_typelist<
        db::wrap_tags_in<Tags::Selected, FieldTagsList>>& selected_fields) {
  tuples::tagged_tuple_from_typelist<FieldTagsList> all_tensor_data{};
  tmpl::for_each<FieldTagsList>([&all_tensor_data, &volume_file,
                                 &observation_id,
                                 &selected_fields](auto field_tag_v) {
    using field_tag = tmpl::type_from<decltype(field_tag_v)>;
    const auto& selection = get<Tags::Selected<field_tag>>(selected_fields);
    if (not selection.has_value()) {
      return;
    }
    read_tensor_data(make_not_null(&get<field_tag>(all_tensor_data)),
                     selection.value(), volume_file, observation_id);
  });
  return all_tensor_data;
}

// Extract this element's data from the read-in dataset
template <typename FieldTagsList>
tuples::tagged_tuple_from_typelist<FieldTagsList> extract_element_data(
    const std::pair<size_t, size_t>& element_data_offset_and_length,
    const tuples::tagged_tuple_from_typelist<FieldTagsList>& all_tensor_data,
    const tuples::tagged_tuple_from_typelist<
        db::wrap_tags_in<Tags::Selected, FieldTagsList>>& selected_fields) {
  tuples::tagged_tuple_from_typelist<FieldTagsList> element_data{};
  tmpl::for_each<FieldTagsList>(
      [&element_data, &offset = element_data_offset_and_length.first,
       &num_points = element_data_offset_and_length.second, &all_tensor_data,
       &selected_fields](auto field_tag_v) {
        using field_tag = tmpl::type_from<decltype(field_tag_v)>;
        const auto& selection = get<Tags::Selected<field_tag>>(selected_fields);
        if (not selection.has_value()) {
          return;
        }
        auto& element_tensor_data = get<field_tag>(element_data);
        // Iterate independent components of the tensor
        for (size_t i = 0; i < element_tensor_data.size(); ++i) {
          const DataVector& data_tensor_component =
              get<field_tag>(all_tensor_data)[i];
          DataVector element_tensor_component{num_points};
          // Retrieve data from slice of the contigious dataset
          for (size_t j = 0; j < element_tensor_component.size(); ++j) {
            element_tensor_component[j] = data_tensor_component[offset + j];
          }
          element_tensor_data[i] = element_tensor_component;
        }
      });
  return element_data;
}

// Check that the source and target points are the same in case interpolation is
// disabled. This is important to avoid hard-to-find bugs where data is loaded
// to the wrong coordinates. For example, if the evolution domain deforms the
// excision surfaces a bit but the initial data doesn't, then it would be wrong
// to load the initial data to the evolution grid without an interpolation.
template <size_t Dim>
void verify_inertial_coordinates(
    const std::pair<size_t, size_t>& source_element_data_offset_and_length,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& source_inertial_coords,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& target_inertial_coords,
    const std::string& grid_name) {
  for (size_t d = 0; d < Dim; ++d) {
    const DataVector& source_coord = source_inertial_coords[d];
    const DataVector& target_coord = target_inertial_coords[d];
    if (target_coord.size() != source_element_data_offset_and_length.second) {
      ERROR_NO_TRACE(
          "The source and target coordinates don't match on grid "
          << grid_name << ". The source coordinates stored in the file have "
          << source_element_data_offset_and_length.second
          << " points, but the target grid has " << target_coord.size()
          << " points. Set 'Interpolate: True' to enable interpolation between "
             "the grids.");
    }
    for (size_t j = 0; j < source_element_data_offset_and_length.second; ++j) {
      if (not equal_within_roundoff(
              target_coord[j],
              source_coord[source_element_data_offset_and_length.first + j])) {
        ERROR_NO_TRACE(
            "The source and target coordinates don't match on grid "
            << grid_name << " in dimension " << d << " at point " << j
            << " (plus offset " << source_element_data_offset_and_length.first
            << " in the data file). Source coordinate: "
            << source_coord[source_element_data_offset_and_length.first + j]
            << ", target coordinate: " << target_coord[j]
            << ". Set 'Interpolate: True' to enable interpolation between the "
               "grids.");
      }
    }
  }
}

// Interpolate only the `selected_fields` in `source_element_data` to the
// `target_logical_coords`.
template <typename FieldTagsList, size_t Dim>
void interpolate_selected_fields(
    const gsl::not_null<tuples::tagged_tuple_from_typelist<FieldTagsList>*>
        target_element_data,
    const tuples::tagged_tuple_from_typelist<FieldTagsList>&
        source_element_data,
    const Mesh<Dim>& source_mesh,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>&
        target_logical_coords,
    const std::vector<size_t>& offsets,
    const tuples::tagged_tuple_from_typelist<
        db::wrap_tags_in<Tags::Selected, FieldTagsList>>& selected_fields) {
  const intrp::Irregular<Dim> interpolator{source_mesh, target_logical_coords};
  const size_t target_num_points = target_logical_coords.begin()->size();
  ASSERT(target_num_points == offsets.size(),
         "The number of target points ("
             << target_num_points << ") must match the number of offsets ("
             << offsets.size() << ").");
  DataVector target_tensor_component_buffer{target_num_points};
  tmpl::for_each<FieldTagsList>([&source_element_data, &target_element_data,
                                 &interpolator, &target_tensor_component_buffer,
                                 &selected_fields, &offsets](auto field_tag_v) {
    using field_tag = tmpl::type_from<decltype(field_tag_v)>;
    const auto& selection = get<Tags::Selected<field_tag>>(selected_fields);
    if (not selection.has_value()) {
      return;
    }
    const auto& source_tensor_data = get<field_tag>(source_element_data);
    auto& target_tensor_data = get<field_tag>(*target_element_data);
    // Iterate independent components of the tensor
    for (size_t i = 0; i < source_tensor_data.size(); ++i) {
      const DataVector& source_tensor_component = source_tensor_data[i];
      DataVector& target_tensor_component = target_tensor_data[i];
      // Interpolate
      interpolator.interpolate(make_not_null(&target_tensor_component_buffer),
                               source_tensor_component);
      // Fill target element data at corresponding offsets
      for (size_t j = 0; j < target_tensor_component_buffer.size(); ++j) {
        target_tensor_component[offsets[j]] = target_tensor_component_buffer[j];
      }
    }
  });
}

}  // namespace detail

namespace Actions {

/*!
 * \brief Read a volume data file and distribute the data to all registered
 * elements, interpolating to the target points if needed.
 *
 * \note Use this action if you want to quickly load and distribute volume data.
 * If you need to beyond that (such as more control over input-file options),
 * write a new action and dispatch to
 * `importers::Actions::ReadAllVolumeDataAndDistribute`.
 *
 * \details Invoke this action on the elements of an array parallel component to
 * dispatch reading the volume data file specified by options placed in the
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
      tmpl::list<Tags::ImporterOptions<ImporterOptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Not using `ckLocalBranch` here to make sure the simple action invocation
    // is asynchronous.
    auto& reader_component = Parallel::get_parallel_component<
        importers::ElementDataReader<Metavariables>>(cache);
    Parallel::simple_action<importers::Actions::ReadAllVolumeDataAndDistribute<
        Dim, FieldTagsList, ParallelComponent>>(
        reader_component,
        get<Tags::ImporterOptions<ImporterOptionsGroup>>(cache), 0_st);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/*!
 * \brief Read a volume data file and distribute the data to all registered
 * elements, interpolating to the target points if needed.
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
 * component for simple use cases.
 *
 * - Pass along the following arguments to the simple action invocation:
 *   - `options`: `importers::ImporterOptions` that specify the H5 files
 *     with volume data to load.
 *   - `volume_data_id`: A number (or hash) that identifies this import
 *     operation. Will also be used to identify the loaded volume data in the
 *     inbox of the receiving elements.
 *   - `selected_fields` (optional): See below.
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
 * elements. The `volume_data_id` passed to this action is used as key.
 *
 * \par Memory consumption
 * This action runs once on every node. It reads all volume data files on the
 * node, but doesn't keep them all in memory at once. The following items
 * contribute primarily to memory consumption and can be reconsidered if we run
 * into memory issues:
 *
 * - `all_tensor_data`: All requested tensor components in the volume data file
 *   at the specified observation ID. Only data from one volume data file is
 *   held in memory at any time. Only data from files that overlap with target
 *   elements on this node are read in.
 * - `target_element_data_buffer`: Holds incomplete interpolated data for each
 *   (target) element that resides on this node. In the worst case, when all
 *   target elements need data from the last source element in the last volume
 *   data file, the memory consumption of this buffer can grow to hold all
 *   requested tensor components on all elements that reside on this node.
 *   However, elements are erased from this buffer once their interpolated data
 *   is complete (and sent to the target element), so the memory consumption
 *   should remain much lower in practice.
 *
 * \see Dev guide on \ref dev_guide_importing
 */
template <size_t Dim, typename FieldTagsList, typename ReceiveComponent>
struct ReadAllVolumeDataAndDistribute {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename ArrayIndex>
  static void apply(DataBox& box, Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ImporterOptions& options, const size_t volume_data_id,
                    tuples::tagged_tuple_from_typelist<
                        db::wrap_tags_in<Tags::Selected, FieldTagsList>>
                        selected_fields = select_all_fields(FieldTagsList{})) {
    const bool enable_interpolation =
        get<OptionTags::EnableInterpolation>(options);

    // Only read and distribute the volume data once
    // This action will be invoked by `importers::Actions::ReadVolumeData` from
    // every element on the node, but only the first invocation reads the file
    // and distributes the data to all elements. Subsequent invocations do
    // nothing. The `volume_data_id` identifies whether or not we have already
    // read the requested data. Doing this at runtime avoids having to collect
    // all data files that will be read in at compile-time to initialize a flag
    // in the DataBox for each of them.
    const auto& has_read_volume_data =
        db::get<Tags::ElementDataAlreadyRead>(box);
    if (has_read_volume_data.find(volume_data_id) !=
        has_read_volume_data.end()) {
      return;
    }
    db::mutate<Tags::ElementDataAlreadyRead>(
        [&volume_data_id](const auto local_has_read_volume_data) {
          local_has_read_volume_data->insert(volume_data_id);
        },
        make_not_null(&box));

    // This is the subset of elements that reside on this node. They have
    // registered themselves before. Our job is to fill them with volume data.
    std::unordered_set<ElementId<Dim>> target_element_ids{};
    for (const auto& target_element : get<Tags::RegisteredElements<Dim>>(box)) {
      const auto& element_array_component_id = target_element.first;
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
      const auto target_element_id =
          Parallel::ArrayIndex<typename ReceiveComponent::array_index>(
              raw_element_index)
              .get_index();
      target_element_ids.insert(target_element_id);
    }
    if (UNLIKELY(target_element_ids.empty())) {
      return;
    }

    // Temporary buffer for data on target elements. These variables get filled
    // with interpolated data while we're reading in volume files. Once data on
    // an element is complete, the data is sent to that element and removed from
    // this list.
    std::unordered_map<ElementId<Dim>,
                       tuples::tagged_tuple_from_typelist<FieldTagsList>>
        target_element_data_buffer{};
    std::unordered_map<ElementId<Dim>, std::vector<size_t>>
        all_indices_of_filled_interp_points{};

    // Resolve the file glob
    const std::string& file_glob = get<OptionTags::FileGlob>(options);
    const std::vector<std::string> file_paths = file_system::glob(file_glob);
    if (file_paths.empty()) {
      ERROR_NO_TRACE("The file glob '" << file_glob << "' matches no files.");
    }

    // Open every file in turn
    std::optional<size_t> prev_observation_id{};
    double observation_value = std::numeric_limits<double>::signaling_NaN();
    std::optional<Domain<Dim>> source_domain{};
    std::unordered_map<std::string,
                       std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
        source_domain_functions_of_time{};
    for (const std::string& file_name : file_paths) {
      // Open the volume data file
      h5::H5File<h5::AccessType::ReadOnly> h5file(file_name);
      constexpr size_t version_number = 0;
      const auto& volume_file = h5file.get<h5::VolumeData>(
          "/" + get<OptionTags::Subgroup>(options), version_number);

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
          get<OptionTags::ObservationValue>(options));
      if (prev_observation_id.has_value() and
          prev_observation_id.value() != observation_id) {
        ERROR("Inconsistent selection of observation ID in file "
              << file_name
              << ". Make sure all files select the same observation ID.");
      }
      prev_observation_id = observation_id;
      observation_value = volume_file.get_observation_value(observation_id);

      // Memory buffer for the tensor data stored in this file. The data is
      // loaded lazily when it is needed. We may find that we can skip loading
      // some files because none of their data is needed to fill the elements on
      // this node.
      std::optional<tuples::tagged_tuple_from_typelist<FieldTagsList>>
          all_tensor_data{};
      std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>
          source_inertial_coords{};

      // Retrieve the information needed to reconstruct which element the data
      // belongs to
      const auto source_grid_names = volume_file.get_grid_names(observation_id);
      const auto source_extents = volume_file.get_extents(observation_id);
      const auto source_bases = volume_file.get_bases(observation_id);
      const auto source_quadratures =
          volume_file.get_quadratures(observation_id);
      std::vector<ElementId<Dim>> source_element_ids{};
      if (enable_interpolation) {
        // Need to parse all source grid names to element IDs only if
        // interpolation is enabled
        source_element_ids.reserve(source_grid_names.size());
        for (const auto& grid_name : source_grid_names) {
          source_element_ids.push_back(ElementId<Dim>(grid_name));
        }
        // Reconstruct domain from volume data file
        const std::optional<std::vector<char>> serialized_domain =
            volume_file.get_domain(observation_id);
        if (not serialized_domain.has_value()) {
          ERROR_NO_TRACE("No serialized domain found in file '"
                         << file_name << volume_file.subfile_path()
                         << "'. The domain is needed for interpolation.");
        }
        if (source_domain.has_value()) {
#ifdef SPECTRE_DEBUG
          // Check that the domain is the same in all files (only in debug mode)
          const auto deserialized_domain =
              deserialize<Domain<Dim>>(serialized_domain->data());
          if (*source_domain != deserialized_domain) {
            ERROR_NO_TRACE(
                "The domain in all volume files must be the same. Domain in "
                "file '"
                << file_name << volume_file.subfile_path()
                << "' differs from a previously read file.");
          }
#endif
        } else {
          source_domain = deserialize<Domain<Dim>>(serialized_domain->data());
        }
        // Reconstruct functions of time from volume data file
        if (source_domain_functions_of_time.empty() and
            alg::any_of(source_domain->blocks(), [](const auto& block) {
              return block.is_time_dependent();
            })) {
          const std::optional<std::vector<char>> serialized_functions_of_time =
              volume_file.get_functions_of_time(observation_id);
          if (not serialized_functions_of_time.has_value()) {
            ERROR_NO_TRACE("No domain functions of time found in file '"
                           << file_name << volume_file.subfile_path()
                           << "'. The functions of time are needed for "
                              "interpolating with time-dependent maps.");
          }
          source_domain_functions_of_time = deserialize<std::unordered_map<
              std::string,
              std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>>(
              serialized_functions_of_time->data());
        }
      }

      // Distribute the tensor data to the registered (target) elements. We
      // erase target elements when they are complete. This allows us to search
      // only for incomplete elements in subsequent volume files, and to stop
      // early when all registered elements are complete.
      std::unordered_set<ElementId<Dim>> completed_target_elements{};
      for (const auto& target_element_id : target_element_ids) {
        const auto& target_points = get<Tags::RegisteredElements<Dim>>(box).at(
            observers::ArrayComponentId(
                std::add_pointer_t<ReceiveComponent>{nullptr},
                Parallel::ArrayIndex<ElementId<Dim>>(target_element_id)));
        const auto target_grid_name = get_output(target_element_id);

        // Proceed with the registered element only if it overlaps with the
        // volume file. It's possible that the volume file only contains data
        // for a subset of elements, e.g., when each node of a simulation wrote
        // volume data for its elements to a separate file.
        std::vector<ElementId<Dim>> overlapping_source_element_ids{};
        std::unordered_map<ElementId<Dim>, ElementLogicalCoordHolder<Dim>>
            source_element_logical_coords{};
        if (enable_interpolation) {
          // Transform the target points to block logical coords in the source
          // domain
          const auto source_block_logical_coords = block_logical_coordinates(
              *source_domain, target_points, observation_value,
              source_domain_functions_of_time);
          // Find the target points in the subset of source elements contained
          // in this volume file
          source_element_logical_coords = element_logical_coordinates(
              source_element_ids, source_block_logical_coords);
          overlapping_source_element_ids.reserve(
              source_element_logical_coords.size());
          for (const auto& source_element_id_and_coords :
               source_element_logical_coords) {
            overlapping_source_element_ids.push_back(
                source_element_id_and_coords.first);
          }
        } else {
          // When interpolation is disabled we process only volume files that
          // contain the exact element
          if (std::find(source_grid_names.begin(), source_grid_names.end(),
                        target_grid_name) == source_grid_names.end()) {
            continue;
          }
          overlapping_source_element_ids.push_back(target_element_id);
        }

        // Lazily load the tensor data from the file if needed
        if (not overlapping_source_element_ids.empty() and
            not all_tensor_data.has_value()) {
          all_tensor_data = detail::read_tensor_data<FieldTagsList>(
              volume_file, observation_id, selected_fields);
        }

        // Iterate over the source elements in this volume file that overlap
        // with the target element
        for (const auto& source_element_id : overlapping_source_element_ids) {
          const auto source_grid_name = get_output(source_element_id);
          // Find the data offset that corresponds to this element
          const auto element_data_offset_and_length =
              h5::offset_and_length_for_grid(source_grid_name,
                                             source_grid_names, source_extents);
          // Extract this element's data from the read-in dataset
          auto source_element_data =
              detail::extract_element_data<FieldTagsList>(
                  element_data_offset_and_length, *all_tensor_data,
                  selected_fields);

          if (enable_interpolation) {
            const auto source_mesh = h5::mesh_for_grid<Dim>(
                source_grid_name, source_grid_names, source_extents,
                source_bases, source_quadratures);
            const size_t target_num_points = target_points.begin()->size();

            // Get and resize target buffer
            auto& target_element_data =
                target_element_data_buffer[target_element_id];
            tmpl::for_each<FieldTagsList>([&target_element_data,
                                           &target_num_points,
                                           &selected_fields](auto field_tag_v) {
              using field_tag = tmpl::type_from<decltype(field_tag_v)>;
              if (get<Tags::Selected<field_tag>>(selected_fields).has_value()) {
                for (auto& component : get<field_tag>(target_element_data)) {
                  component.destructive_resize(target_num_points);
                }
              }
            });
            auto& indices_of_filled_interp_points =
                all_indices_of_filled_interp_points[target_element_id];

            // Interpolate!
            const auto& source_logical_coords_of_target_points =
                source_element_logical_coords.at(source_element_id);
            detail::interpolate_selected_fields<FieldTagsList>(
                make_not_null(&target_element_data), source_element_data,
                source_mesh,
                source_logical_coords_of_target_points.element_logical_coords,
                source_logical_coords_of_target_points.offsets,
                selected_fields);
            indices_of_filled_interp_points.insert(
                indices_of_filled_interp_points.end(),
                source_logical_coords_of_target_points.offsets.begin(),
                source_logical_coords_of_target_points.offsets.end());

            if (indices_of_filled_interp_points.size() == target_num_points) {
              // Pass the (interpolated) data to the element. Now it can proceed
              // in parallel with transforming the data, taking derivatives on
              // the grid, etc.
              Parallel::receive_data<Tags::VolumeData<FieldTagsList>>(
                  Parallel::get_parallel_component<ReceiveComponent>(
                      cache)[target_element_id],
                  volume_data_id, std::move(target_element_data));
              completed_target_elements.insert(target_element_id);
              target_element_data_buffer.erase(target_element_id);
              all_indices_of_filled_interp_points.erase(target_element_id);
            }
          } else {
            // Verify that the inertial coordinates of the source and target
            // elements match. To do so we retrieve the inertial coordinates
            // that are written alongside the tensor data in the file. This is
            // an important check. It avoids nasty bugs where tensor data is
            // read in to points that don't exactly match the input. Therefore
            // we DON'T restrict this check to Debug mode.
            if (not source_inertial_coords.has_value()) {
              tnsr::I<DataVector, Dim, Frame::Inertial> inertial_coords{};
              detail::read_tensor_data(make_not_null(&inertial_coords),
                                       "InertialCoordinates", volume_file,
                                       observation_id);
              source_inertial_coords = std::move(inertial_coords);
            }
            detail::verify_inertial_coordinates(
                element_data_offset_and_length, *source_inertial_coords,
                target_points, source_grid_name);
            // Pass data directly to the element when interpolation is disabled
            Parallel::receive_data<Tags::VolumeData<FieldTagsList>>(
                Parallel::get_parallel_component<ReceiveComponent>(
                    cache)[target_element_id],
                volume_data_id, std::move(source_element_data));
            completed_target_elements.insert(target_element_id);
          }
        }  // loop over overlapping source elements
      }    // loop over registered elements
      for (const auto& completed_element_id : completed_target_elements) {
        target_element_ids.erase(completed_element_id);
      }
      // Stop early when all target elements are complete
      if (target_element_ids.empty()) {
        break;
      }
    }  // loop over volume files

    // Have we completed all target elements? If we haven't, the target domain
    // probably extends outside the source domain. In that case we report the
    // coordinates that couldn't be filled.
    if (not target_element_ids.empty()) {
      std::unordered_map<ElementId<Dim>, std::vector<std::array<double, Dim>>>
          missing_coords{};
      size_t num_missing_points = 0;
      for (const auto& target_element_id : target_element_ids) {
        const auto& target_inertial_coords =
            get<Tags::RegisteredElements<Dim>>(box).at(
                observers::ArrayComponentId(
                    std::add_pointer_t<ReceiveComponent>{nullptr},
                    Parallel::ArrayIndex<ElementId<Dim>>(target_element_id)));
        const size_t target_num_points = target_inertial_coords.begin()->size();
        const auto& indices_of_filled_interp_points =
            all_indices_of_filled_interp_points[target_element_id];
        for (size_t i = 0; i < target_num_points; ++i) {
          if (alg::find(indices_of_filled_interp_points, i) ==
              indices_of_filled_interp_points.end()) {
            missing_coords[target_element_id].push_back(
                [&target_inertial_coords, &i]() {
                  std::array<double, Dim> x{};
                  for (size_t d = 0; d < Dim; ++d) {
                    x[d] = target_inertial_coords[d][i];
                  }
                  return x;
                }());
            ++num_missing_points;
          }
        }
      }
      ERROR_NO_TRACE("The following " << num_missing_points << " point(s) in "
                                      << missing_coords.size()
                                      << " element(s) could not be filled:\n"
                                      << missing_coords);
    }
  }

 private:
  template <typename... LocalFieldTags>
  static tuples::TaggedTuple<Tags::Selected<LocalFieldTags>...>
  select_all_fields(tmpl::list<LocalFieldTags...> /*meta*/) {
    return {db::tag_name<LocalFieldTags>()...};
  }
};

}  // namespace Actions
}  // namespace importers
