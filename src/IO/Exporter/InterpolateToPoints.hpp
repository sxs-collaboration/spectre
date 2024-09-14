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

/*!
 * \brief Interpolates data in volume files to target points
 *
 * This is an overload of the `interpolate_to_points` function that works with
 * Tensor types and tags, rather than the raw C++ types that are used in the
 * other overload so it can be used by external programs.
 *
 * The `Tags` template parameter is a typelist of tags that should be read from
 * the volume files. The dataset names to read are constructed from the tag
 * names. Here is an example of how to use this function:
 *
 * \snippet Test_Exporter.cpp interpolate_tensors_to_points_example
 */
template <typename Tags, typename DataType, size_t Dim>
tuples::tagged_tuple_from_typelist<Tags> interpolate_to_points(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    const std::string& subfile_name,
    const std::variant<ObservationId, ObservationStep>& observation,
    const tnsr::I<DataType, Dim>& target_points,
    bool extrapolate_into_excisions = false,
    std::optional<size_t> num_threads = std::nullopt) {
  // Convert target_points to an array of vectors
  // Possible performance optimization: avoid copying the data by allowing
  // interpolate_to_points to accept pointers or spans.
  const size_t num_points = target_points.begin()->size();
  std::array<std::vector<double>, Dim> target_points_array{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(target_points_array, d).resize(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      gsl::at(target_points_array, d)[i] = target_points.get(d)[i];
    }
  }
  // Collect tensor component names
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
  // Interpolate!
  const auto interpolated_data = interpolate_to_points(
      volume_files_or_glob, subfile_name, observation, tensor_components,
      target_points_array, extrapolate_into_excisions, num_threads);
  // Convert the interpolated data to a tagged_tuple
  tuples::tagged_tuple_from_typelist<Tags> result{};
  size_t component_index = 0;
  tmpl::for_each<Tags>(
      [&component_index, &interpolated_data, &result](auto tag_v) {
        using tensor_tag = tmpl::type_from<decltype(tag_v)>;
        using TensorType = typename tensor_tag::type;
        for (size_t i = 0; i < TensorType::size(); ++i) {
          const auto& component_data = interpolated_data[component_index];
          get<tensor_tag>(result)[i] = DataVector(component_data);
          ++component_index;
        }
      });
  return result;
}

}  // namespace spectre::Exporter
