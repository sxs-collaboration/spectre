// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Domain.hpp"
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
template <typename ResultDataType, size_t Dim, typename Frame>
void interpolate_to_points(
    gsl::not_null<std::vector<ResultDataType>*> result,
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    const std::string& subfile_name, const ObservationVariant& observation,
    const std::vector<std::string>& tensor_components,
    const tnsr::I<DataVector, Dim, Frame>& target_points,
    bool extrapolate_into_excisions = false,
    bool error_on_missing_points = false,
    std::optional<size_t> num_threads = std::nullopt);

template <size_t Dim, typename Frame>
std::vector<DataVector> interpolate_to_points(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    const std::string& subfile_name, const ObservationVariant& observation,
    const std::vector<std::string>& tensor_components,
    const tnsr::I<DataVector, Dim, Frame>& target_points,
    bool extrapolate_into_excisions = false,
    bool error_on_missing_points = false,
    std::optional<size_t> num_threads = std::nullopt) {
  std::vector<DataVector> interpolated_data{};
  interpolate_to_points(make_not_null(&interpolated_data), volume_files_or_glob,
                        subfile_name, observation, tensor_components,
                        target_points, extrapolate_into_excisions,
                        error_on_missing_points, num_threads);
  return interpolated_data;
}

template <typename Tags, size_t Dim, typename Frame>
tuples::tagged_tuple_from_typelist<Tags> interpolate_to_points(
    const std::variant<std::vector<std::string>, std::string>&
        volume_files_or_glob,
    const std::string& subfile_name, const ObservationVariant& observation,
    const tnsr::I<DataVector, Dim, Frame>& target_points,
    bool extrapolate_into_excisions = false,
    const bool error_on_missing_points = false,
    std::optional<size_t> num_threads = std::nullopt) {
  return make_tagged_tuple<Tags>(interpolate_to_points(
      volume_files_or_glob, subfile_name, observation,
      get_tensor_components<Tags>(), target_points, extrapolate_into_excisions,
      error_on_missing_points, num_threads));
}
/// @}

/*!
 * \brief Interpolates data in volume files to target points by reading the
 * volume data into memory
 *
 * This class reads the volume data at the requested time into memory and can
 * then interpolate it to any number of target points.
 *
 * \par Thread safety
 * Constructing the `PointwiseInterpolator` on multiple threads at once is not
 * thread safe (unless HDF5 was built with thread-safety support) because the
 * constructor opens the H5 files and reads in the data. However, once the data
 * is loaded, the `interpolate_to_points` and `interpolate_to_point` functions
 * are thread safe. This means the volume data can be loaded once in a single
 * thread and then used by multiple threads to interpolate to different points
 * in parallel.
 */
template <size_t Dim, typename Frame = ::Frame::Inertial>
struct PointwiseInterpolator {
  PointwiseInterpolator() = default;
  PointwiseInterpolator(const std::variant<std::vector<std::string>,
                                           std::string>& volume_files_or_glob,
                        const std::string& subfile_name,
                        const ObservationVariant& observation,
                        const std::vector<std::string>& tensor_components);

  /*!
   * \brief Interpolate to many points
   *
   * \param result the interpolated data at the target points. The outer vector
   * is the number of components and the inner vector is the number of target
   * points. Will be resized automatically.
   * \param target_points the points to interpolate to
   * \param extrapolate_into_excisions whether to extrapolate into excised
   * regions
   * \param error_on_missing_points whether to throw an error if any of the
   * target points are outside the domain
   * \param num_threads the number of OpenMP threads to use to parallelize the
   * interpolation over the target points. If not provided, the default number
   * of threads will be used.
   */
  void interpolate_to_points(
      gsl::not_null<std::vector<DataVector>*> result,
      const tnsr::I<DataVector, Dim, Frame>& target_points,
      bool extrapolate_into_excisions = false,
      bool error_on_missing_points = false,
      std::optional<size_t> num_threads = std::nullopt) const;

  /*!
   * \brief Interpolate to a single point
   *
   * This function is thread safe, so the interpolator can be constructed to
   * load the volume data once and then used by multiple threads to interpolate
   * to different points in parallel. Note that updating the `block_order` (if
   * provided) is not thread safe, so each thread should manage a separate
   * `block_order`.
   *
   * \param result the interpolated data at the target point. The vector is over
   * the number of components. Will be resized automatically.
   * \param target_point the point to interpolate to
   * \param block_order an optional priority order to search for the block
   * containing the target point. Will be updated when the point is found.
   * See `block_logical_coordinates_single_point` for more details.
   */
  void interpolate_to_point(gsl::not_null<std::vector<double>*> result,
                            const tnsr::I<double, Dim, Frame>& target_point,
                            std::optional<gsl::not_null<std::vector<size_t>*>>
                                block_order = std::nullopt) const;

  size_t obs_id() const { return obs_id_; }
  double time() const { return time_; }
  const Domain<Dim>& domain() const { return domain_; }
  const domain::FunctionsOfTimeMap& functions_of_time() const {
    return functions_of_time_;
  }

 private:
  // Loaded from data files
  size_t obs_id_ = std::numeric_limits<size_t>::max();
  double time_ = std::numeric_limits<double>::signaling_NaN();
  Domain<Dim> domain_;
  domain::FunctionsOfTimeMap functions_of_time_;
  // Outer vector is the source data file
  std::vector<std::vector<ElementId<Dim>>> element_ids_;
  std::vector<
      std::unordered_map<ElementId<Dim>, std::tuple<Mesh<Dim>, size_t, size_t>>>
      meshes_;
  std::vector<std::vector<DataVector>> tensor_data_;
};

}  // namespace spectre::Exporter
