// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "IO/Exporter/Exporter.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace spectre::Exporter {

/// Collect tensor component names from Tags list
template <typename Tags>
auto get_tensor_components() {
  std::vector<std::string> tensor_components{};
  tmpl::for_each<Tags>([&tensor_components](auto tag_v) {
    using tensor_tag = tmpl::type_from<decltype(tag_v)>;
    using TensorType = typename tensor_tag::type;
    const std::string tag_name = db::tag_name<tensor_tag>();
    for (size_t i = 0; i < TensorType::size(); ++i) {
      const std::string component_name =
          tag_name + TensorType::component_suffix(i);
      tensor_components.push_back(component_name);
    }
  });
  return tensor_components;
}

/// Convert tensor components to a tagged_tuple
template <typename Tags, typename DataType>
auto make_tagged_tuple(std::vector<DataType> interpolated_data) {
  tuples::tagged_tuple_from_typelist<Tags> result{};
  size_t component_index = 0;
  tmpl::for_each<Tags>(
      [&component_index, &interpolated_data, &result](auto tag_v) {
        using tensor_tag = tmpl::type_from<decltype(tag_v)>;
        using TensorType = typename tensor_tag::type;
        for (size_t i = 0; i < TensorType::size(); ++i) {
          get<tensor_tag>(result)[i] =
              std::move(interpolated_data[component_index]);
          ++component_index;
        }
      });
  return result;
}

/// @{
/*!
 * \brief Interpolates data in volume files to target points
 *
 * These are overloads of the `interpolate_to_points` function that work with
 * Tensor types and tags, rather than the raw C++ types that are used in
 * Exporter.hpp so it can be used by external programs.
 *
 * The `Tags` template parameter is a typelist of tags that should be read from
 * the volume files. The dataset names to read are constructed from the tag
 * names. Here is an example of how to use this function:
 *
 * \snippet Test_Exporter.cpp interpolate_tensors_to_points_example
 */
template <typename ResultDataType, size_t Dim>
void interpolate_to_points(
    gsl::not_null<std::vector<ResultDataType>*> result,
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    const std::string& subfile_name, const ObservationVariant& observation,
    const std::vector<std::string>& tensor_components,
    const tnsr::I<DataVector, Dim>& target_points,
    bool extrapolate_into_excisions = false,
    std::optional<size_t> num_threads = std::nullopt);

template <size_t Dim>
std::vector<DataVector> interpolate_to_points(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    const std::string& subfile_name, const ObservationVariant& observation,
    const std::vector<std::string>& tensor_components,
    const tnsr::I<DataVector, Dim>& target_points,
    bool extrapolate_into_excisions = false,
    std::optional<size_t> num_threads = std::nullopt) {
  std::vector<DataVector> interpolated_data{};
  interpolate_to_points(make_not_null(&interpolated_data), volume_files_or_glob,
                        subfile_name, observation, tensor_components,
                        target_points, extrapolate_into_excisions, num_threads);
  return interpolated_data;
}

template <typename Tags, size_t Dim>
tuples::tagged_tuple_from_typelist<Tags> interpolate_to_points(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    const std::string& subfile_name, const ObservationVariant& observation,
    const tnsr::I<DataVector, Dim>& target_points,
    bool extrapolate_into_excisions = false,
    std::optional<size_t> num_threads = std::nullopt) {
  return make_tagged_tuple<Tags>(
      interpolate_to_points(volume_files_or_glob, subfile_name, observation,
                            get_tensor_components<Tags>(), target_points,
                            extrapolate_into_excisions, num_threads));
}
/// @}

}  // namespace spectre::Exporter
