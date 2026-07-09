// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
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
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Exporter/Exporter.hpp"
#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "IO/Exporter/SelectObservation.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Importers/ObservationSelector.hpp"
#include "IO/Importers/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

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

// Translate the importer's observation selection into the Exporter's
// `ObservationVariant`
inline spectre::Exporter::ObservationVariant observation_variant(
    const std::variant<double, ObservationSelector>& observation_value,
    const std::optional<double>& observation_value_epsilon) {
  return std::visit(
      Overloader{[&observation_value_epsilon](const double local_obs_value)
                     -> spectre::Exporter::ObservationVariant {
                   if (observation_value_epsilon.has_value()) {
                     return spectre::Exporter::ObservationValue{
                         local_obs_value, observation_value_epsilon.value()};
                   }
                   return local_obs_value;
                 },
                 [](const ObservationSelector local_obs_selector)
                     -> spectre::Exporter::ObservationVariant {
                   switch (local_obs_selector) {
                     case ObservationSelector::First:
                       return spectre::Exporter::ObservationStep{0};
                     case ObservationSelector::Last:
                       return spectre::Exporter::ObservationStep{-1};
                     default:
                       ERROR("Unknown importers::ObservationSelector: "
                             << local_obs_selector);
                   }
                 }},
      observation_value);
}

// Read the single `tensor_name` from the `volume_file`, taking care of suffixes
// like "_x" etc for its components.
template <typename TensorType>
void read_tensor_data(const gsl::not_null<TensorType*> tensor_data,
                      const std::string& tensor_name,
                      const h5::VolumeData& volume_file,
                      const size_t observation_id) {
  for (size_t i = 0; i < tensor_data->size(); ++i) {
    const auto& tensor_component = volume_file.get_tensor_component(
        observation_id, tensor_name + tensor_data->component_suffix(
                                          tensor_data->get_tensor_index(i)));
    if (not std::holds_alternative<DataVector>(tensor_component.data)) {
      ERROR("The tensor component '"
            << tensor_component.name
            << "' is not a double-precision DataVector. Reading in "
               "single-precision volume data is not supported.");
    }
    (*tensor_data)[i] = std::get<DataVector>(tensor_component.data);
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

// Check that the inertial coordinates computed with the given domain are the
// same as the ones passed to this function.
// This is important to avoid hard-to-find bugs where data is loaded
// to the wrong coordinates. For example, if the evolution domain deforms the
// excision surfaces a bit but the initial data doesn't, then it would be wrong
// to load the initial data to the evolution grid without an interpolation.
template <size_t Dim>
void verify_inertial_coordinates(
    const Domain<Dim>& domain, const double time,
    const domain::FunctionsOfTimeMap& functions_of_time,
    const ElementId<Dim>& element_id, const Mesh<Dim>& mesh,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  const auto logical_coords = logical_coordinates(mesh);
  ElementMap<Dim, Frame::Inertial> element_map{
      element_id, domain.blocks()[element_id.block_id()]};
  const auto mapped_inertial_coords =
      element_map(logical_coords, time, functions_of_time);
  const double scale = blaze::max(get(magnitude(mapped_inertial_coords)));
  if (not equal_within_roundoff(mapped_inertial_coords, inertial_coords,
                                std::numeric_limits<double>::epsilon() * 100.0,
                                scale)) {
    DataVector diff =
        square(get<0>(inertial_coords) - get<0>(mapped_inertial_coords));
    for (size_t d = 1; d < Dim; ++d) {
      diff += square(inertial_coords.get(d) - mapped_inertial_coords.get(d));
    }
    diff = sqrt(diff);
    const double max_coord_distance = blaze::max(diff);
    CAPTURE_FOR_ERROR(element_id);
    CAPTURE_FOR_ERROR(max_coord_distance);
    CAPTURE_FOR_ERROR(scale);
    ERROR_NO_TRACE(
        "The source and target domain don't match. Set 'ElementsAreIdentical: "
        "False' to enable interpolation between the grids.");
  }
}

// Interpolate only the `selected_fields` in `source_element_data` to the
// `target_mesh` (used when elements differ only by p-refinement)
template <typename FieldTagsList, size_t Dim>
void interpolate_selected_fields(
    const gsl::not_null<tuples::tagged_tuple_from_typelist<FieldTagsList>*>
        target_element_data,
    const tuples::tagged_tuple_from_typelist<FieldTagsList>&
        source_element_data,
    const Mesh<Dim>& source_mesh, const Mesh<Dim>& target_mesh,
    const tuples::tagged_tuple_from_typelist<
        db::wrap_tags_in<Tags::Selected, FieldTagsList>>& selected_fields) {
  const intrp::RegularGrid<Dim> interpolator{source_mesh, target_mesh};
  tmpl::for_each<FieldTagsList>([&source_element_data, &target_element_data,
                                 &interpolator,
                                 &selected_fields](auto field_tag_v) {
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
      interpolator.interpolate(make_not_null(&target_tensor_component),
                               source_tensor_component);
    }
  });
}

// Scatter the slice `[start, start + num_points)` of the flat,
// component-indexed `interpolated_data` (as produced by
// `spectre::Exporter::interpolate_to_points`) into a tagged tuple of tensors
// for a single target element. The components are laid out in `FieldTagsList`
// order, skipping unselected fields, matching the order of the
// `tensor_components` passed to the interpolation.
template <typename FieldTagsList>
tuples::tagged_tuple_from_typelist<FieldTagsList> scatter_element_data(
    const std::vector<DataVector>& interpolated_data, const size_t start,
    const size_t num_points,
    const tuples::tagged_tuple_from_typelist<
        db::wrap_tags_in<Tags::Selected, FieldTagsList>>& selected_fields) {
  tuples::tagged_tuple_from_typelist<FieldTagsList> element_data{};
  size_t component_index = 0;
  tmpl::for_each<FieldTagsList>([&element_data, &interpolated_data, &start,
                                 &num_points, &selected_fields,
                                 &component_index](auto field_tag_v) {
    using field_tag = tmpl::type_from<decltype(field_tag_v)>;
    if (not get<Tags::Selected<field_tag>>(selected_fields).has_value()) {
      return;
    }
    auto& element_tensor_data = get<field_tag>(element_data);
    for (size_t i = 0; i < element_tensor_data.size(); ++i) {
      DataVector component{num_points};
      const DataVector& interpolated_component =
          interpolated_data[component_index];
      for (size_t j = 0; j < num_points; ++j) {
        component[j] = interpolated_component[start + j];
      }
      element_tensor_data[i] = std::move(component);
      ++component_index;
    }
  });
  return element_data;
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
 * that was encoded into the `Parallel::ArrayComponentId` used to register the
 * elements. The `volume_data_id` passed to this action is used as key.
 *
 * \par Memory consumption
 * This action runs once on every node. Volume data files are loaded one at a
 * time, so memory consumption does _not_ grow with the the number of source
 * files. All coordinates of elements on this node and their interpolated data
 * is held in memory at once, so memory consumption scales with the number of
 * elements on this node.
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
    const bool elements_are_identical =
        get<OptionTags::ElementsAreIdentical>(options);

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
      // Can't use Parallel::make_array_component_id here because we need the
      // original array_index type, not a CkArrayIndex.
      if (element_array_component_id !=
          Parallel::ArrayComponentId(
              std::add_pointer_t<ReceiveComponent>{nullptr},
              raw_element_index)) {
        continue;
      }
      const auto target_element_id =
          Parallel::ArrayIndex<ElementId<Dim>>(raw_element_index).get_index();
      target_element_ids.insert(target_element_id);
    }
    if (UNLIKELY(target_element_ids.empty())) {
      return;
    }

    // Resolve the file glob
    const std::string& file_glob = get<OptionTags::FileGlob>(options);
    const std::vector<std::string> file_paths = file_system::glob(file_glob);
    if (file_paths.empty()) {
      ERROR_NO_TRACE("The file glob '" << file_glob << "' matches no files.");
    }

    // Select observation to read from each file
    const spectre::Exporter::ObservationVariant observation =
        detail::observation_variant(
            get<OptionTags::ObservationValue>(options),
            get<OptionTags::ObservationValueEpsilon>(options));

    // When interpolation between the source and target grids is needed, reuse
    // spectre::Exporter::interpolate_to_points to interpolate to all target
    // points on this node at once, then scatter the results to the elements.
    if (not elements_are_identical) {
      // Gather all target points on this node into a single contiguous tensor,
      // recording the range [start, start + num_points) of each target element.
      std::vector<ElementId<Dim>> target_ids{};
      std::vector<size_t> target_starts{};
      std::vector<size_t> target_num_points{};
      std::vector<const tnsr::I<DataVector, Dim, Frame::Inertial>*>
          target_coords{};
      target_ids.reserve(target_element_ids.size());
      target_starts.reserve(target_element_ids.size());
      target_num_points.reserve(target_element_ids.size());
      target_coords.reserve(target_element_ids.size());
      size_t total_num_points = 0;
      for (const auto& target_element_id : target_element_ids) {
        const auto& target_points =
            get<Tags::RegisteredElements<Dim>>(box)
                .at(Parallel::make_array_component_id<ReceiveComponent>(
                    target_element_id))
                .first;
        target_ids.push_back(target_element_id);
        target_coords.push_back(&target_points);
        target_starts.push_back(total_num_points);
        target_num_points.push_back(target_points.begin()->size());
        total_num_points += target_num_points.back();
      }
      tnsr::I<DataVector, Dim, Frame::Inertial> all_target_points{
          total_num_points};
      for (size_t e = 0; e < target_ids.size(); ++e) {
        for (size_t d = 0; d < Dim; ++d) {
          for (size_t i = 0; i < target_num_points[e]; ++i) {
            all_target_points.get(d)[target_starts[e] + i] =
                target_coords[e]->get(d)[i];
          }
        }
      }

      // Flat list of dataset component names for the selected fields, in
      // `FieldTagsList` order. The layout matches
      // `detail::scatter_element_data`.
      std::vector<std::string> tensor_components{};
      tmpl::for_each<FieldTagsList>([&tensor_components,
                                     &selected_fields](auto field_tag_v) {
        using field_tag = tmpl::type_from<decltype(field_tag_v)>;
        const auto& selection = get<Tags::Selected<field_tag>>(selected_fields);
        if (not selection.has_value()) {
          return;
        }
        using TensorType = typename field_tag::type;
        for (size_t i = 0; i < TensorType::size(); ++i) {
          tensor_components.push_back(selection.value() +
                                      TensorType::component_suffix(i));
        }
      });

      // Interpolate all target points at once. Error if any target point lies
      // outside the source domain. This implementation opens each file in turn
      // (so it doesn't hold all files in memory at once), and it is efficient
      // about mapping points through the blocks of the source domain.
      std::vector<DataVector> interpolated_data{};
      spectre::Exporter::interpolate_to_points(
          make_not_null(&interpolated_data), file_paths,
          "/" + get<OptionTags::Subgroup>(options), observation,
          tensor_components, all_target_points,
          get<OptionTags::ExtrapolateIntoExcisions>(options),
          /*error_on_missing_points=*/true,
          get<OptionTags::NumThreads>(options));
      // The target points are no longer needed; free them before distributing
      // the (potentially large) interpolated data to the target elements.
      all_target_points = tnsr::I<DataVector, Dim, Frame::Inertial>{};

      // Distribute the interpolated data to the target elements.
      for (size_t e = 0; e < target_ids.size(); ++e) {
        auto target_element_data = detail::scatter_element_data<FieldTagsList>(
            interpolated_data, target_starts[e], target_num_points[e],
            selected_fields);
        if constexpr (Parallel::is_dg_element_collection_v<ReceiveComponent>) {
          ERROR("Can't yet do numerical initial data with nodegroups");
        } else {
          Parallel::receive_data<Tags::VolumeData<FieldTagsList>>(
              Parallel::get_parallel_component<ReceiveComponent>(
                  cache)[target_ids[e]],
              volume_data_id, std::move(target_element_data));
        }
      }
      return;
    }  // not elements_are_identical

    // Now handle identical elements:
    // The source and target elements are the same (matching domains and
    // h-refinement), so data is transferred one-to-one, interpolating only
    // between different meshes (p-refinement).
    std::optional<size_t> prev_observation_id{};
    double observation_value = std::numeric_limits<double>::signaling_NaN();
    std::optional<Domain<Dim>> source_domain{};
    domain::FunctionsOfTimeMap source_domain_functions_of_time{};
    for (const std::string& file_name : file_paths) {
      // Open the volume data file
      h5::H5File<h5::AccessType::ReadOnly> h5file(file_name);
      constexpr size_t version_number = 0;
      const auto& volume_file = h5file.get<h5::VolumeData>(
          "/" + get<OptionTags::Subgroup>(options), version_number);

      // Select observation ID
      const size_t observation_id = std::visit(
          spectre::Exporter::SelectObservation{volume_file}, observation);
      if (prev_observation_id.has_value() and
          prev_observation_id.value() != observation_id) {
        ERROR("Inconsistent selection of observation ID in file "
              << file_name
              << ". Make sure all files select the same observation ID.");
      }
      prev_observation_id = observation_id;
      observation_value = volume_file.get_observation_value(observation_id);

      // Memory buffer for the tensor data stored in this file. The data is
      // loaded lazily when it is needed, so we can skip loading files that
      // contain none of the elements on this node.
      std::optional<tuples::tagged_tuple_from_typelist<FieldTagsList>>
          all_tensor_data{};

      // Retrieve the information needed to reconstruct which element the data
      // belongs to
      const auto source_grid_names = volume_file.get_grid_names(observation_id);
      const auto source_extents = volume_file.get_extents(observation_id);
      const auto source_bases = volume_file.get_bases(observation_id);
      const auto source_quadratures =
          volume_file.get_quadratures(observation_id);
      // Reconstruct domain from volume data file
      const std::optional<std::vector<char>> serialized_domain =
          volume_file.get_domain();
      if (serialized_domain.has_value()) {
        if (source_domain.has_value()) {
#ifdef SPECTRE_DEBUG
          // Check that the domain is the same in all files (only in debug
          // mode)
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
      } else {
        Parallel::printf(
            "WARNING: No serialized domain found in file. "
            "Verification that elements in the source and target domain "
            "match will be skipped.\n");
      }
      // Reconstruct functions of time from volume data file
      if (source_domain_functions_of_time.empty() and
          source_domain.has_value() and
          alg::any_of(source_domain->blocks(), [](const auto& block) {
            return block.is_time_dependent();
          })) {
        const std::optional<std::vector<char>> serialized_functions_of_time =
            volume_file.get_functions_of_time(observation_id);
        if (not serialized_functions_of_time.has_value()) {
          ERROR_NO_TRACE("No domain functions of time found in file '"
                         << file_name << volume_file.subfile_path()
                         << "'. The functions of time are needed to verify the "
                            "inertial coordinates with time-dependent maps.");
        }
        source_domain_functions_of_time =
            deserialize<domain::FunctionsOfTimeMap>(
                serialized_functions_of_time->data());
      }

      // Transfer the data to the target elements contained in this file. We
      // erase target elements when they are complete, so subsequent files only
      // search for the remaining elements and we can stop early.
      std::unordered_set<ElementId<Dim>> completed_target_elements{};
      for (const auto& target_element_id : target_element_ids) {
        const auto& [target_points, target_mesh] =
            get<Tags::RegisteredElements<Dim>>(box).at(
                Parallel::make_array_component_id<ReceiveComponent>(
                    target_element_id));
        const auto target_grid_name = get_output(target_element_id);
        // Process this element only if it's in the file
        if (std::find(source_grid_names.begin(), source_grid_names.end(),
                      target_grid_name) == source_grid_names.end()) {
          continue;
        }

        // Lazily load the tensor data from the file
        if (not all_tensor_data.has_value()) {
          all_tensor_data = detail::read_tensor_data<FieldTagsList>(
              volume_file, observation_id, selected_fields);
        }

        const auto source_mesh = h5::mesh_for_grid<Dim>(
            target_grid_name, source_grid_names, source_extents, source_bases,
            source_quadratures);
        const auto element_data_offset_and_length =
            h5::offset_and_length_for_grid(target_grid_name, source_grid_names,
                                           source_extents);
        auto source_element_data = detail::extract_element_data<FieldTagsList>(
            element_data_offset_and_length, *all_tensor_data, selected_fields);

        // Verify that the source and target elements really are the same
        if (source_domain.has_value()) {
          detail::verify_inertial_coordinates(*source_domain, observation_value,
                                              source_domain_functions_of_time,
                                              target_element_id, target_mesh,
                                              target_points);
        }

        // Transfer the data one-to-one, interpolating only if the meshes differ
        // by p-refinement
        tuples::tagged_tuple_from_typelist<FieldTagsList> target_element_data{};
        if (source_mesh == target_mesh) {
          target_element_data = std::move(source_element_data);
        } else {
          detail::interpolate_selected_fields<FieldTagsList>(
              make_not_null(&target_element_data), source_element_data,
              source_mesh, target_mesh, selected_fields);
        }
        if constexpr (Parallel::is_dg_element_collection_v<ReceiveComponent>) {
          ERROR("Can't yet do numerical initial data with nodegroups");
        } else {
          Parallel::receive_data<Tags::VolumeData<FieldTagsList>>(
              Parallel::get_parallel_component<ReceiveComponent>(
                  cache)[target_element_id],
              volume_data_id, std::move(target_element_data));
        }
        completed_target_elements.insert(target_element_id);
      }  // loop over registered elements
      for (const auto& completed_element_id : completed_target_elements) {
        target_element_ids.erase(completed_element_id);
      }
      // Stop early when all target elements are complete
      if (target_element_ids.empty()) {
        break;
      }
    }  // loop over volume files

    // Have we completed all target elements? If we haven't, the source and
    // target domains probably don't match.
    if (not target_element_ids.empty()) {
      ERROR_NO_TRACE("The following "
                     << target_element_ids.size()
                     << " element(s) were not found in the source volume "
                        "data files:\n"
                     << target_element_ids
                     << "\nMake sure the source and target domains match "
                        "when 'ElementsAreIdentical' is enabled, or set "
                        "it to 'False' to interpolate between the grids.");
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
