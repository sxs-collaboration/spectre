// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/Exporter/SpacetimeInterpolator.hpp"

#include <array>
#include <cstddef>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "IO/Exporter/SelectObservation.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Interpolation/PolynomialInterpolation.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace spectre::Exporter {

template <size_t Dim, typename Frame>
SpacetimeInterpolator<Dim, Frame>::SpacetimeInterpolator(
    std::variant<std::vector<std::string>, std::string> volume_files_or_glob,
    std::string subfile_name, std::vector<std::string> tensor_components)
    : volume_files_or_glob_(std::move(volume_files_or_glob)),
      subfile_name_(std::move(subfile_name)),
      tensor_components_(std::move(tensor_components)) {
  const std::vector<std::string> filenames =
      std::visit(Overloader{[](const std::vector<std::string>& volume_files) {
                              return volume_files;
                            },
                            [](const std::string& volume_files_glob) {
                              return file_system::glob(volume_files_glob);
                            }},
                 volume_files_or_glob_);
  if (filenames.empty()) {
    ERROR("No volume data files found.");
  }
  const h5::H5File<h5::AccessType::ReadOnly> first_h5file(filenames.front());
  const auto& first_volfile = first_h5file.get<h5::VolumeData>(subfile_name_);
  all_observation_ids_ = first_volfile.list_observation_ids();
  all_observation_values_.resize(all_observation_ids_.size());
  std::transform(all_observation_ids_.begin(), all_observation_ids_.end(),
                 all_observation_values_.begin(),
                 [&first_volfile](const size_t obs_id) {
                   return first_volfile.get_observation_value(obs_id);
                 });
  if (all_observation_ids_.empty()) {
    ERROR("No observation IDs found in the volume data files.");
  }
  if (all_observation_ids_.size() < time_interpolation_order_ + 1) {
    ERROR(
        "The number of available observations must be at least the time "
        "interpolation order plus one (num observations >= "
        << time_interpolation_order_ + 1 << "), but "
        << all_observation_ids_.size() << " observations were found.");
  }
  auto serialized_fots = first_volfile.get_global_functions_of_time();
  if (serialized_fots.has_value()) {
    functions_of_time_ =
        deserialize<domain::FunctionsOfTimeMap>(serialized_fots->data());
  }
  first_h5file.close();
}

template <size_t Dim, typename Frame>
SpacetimeInterpolator<Dim, Frame>::SpacetimeInterpolator(
    SpacetimeInterpolator&& rhs)
    : volume_files_or_glob_(std::move(rhs.volume_files_or_glob_)),
      subfile_name_(std::move(rhs.subfile_name_)),
      tensor_components_(std::move(rhs.tensor_components_)),
      functions_of_time_(std::move(rhs.functions_of_time_)),
      all_observation_ids_(std::move(rhs.all_observation_ids_)),
      all_observation_values_(std::move(rhs.all_observation_values_)),
      time_bounds_(std::move(rhs.time_bounds_)),
      interpolators_(std::move(rhs.interpolators_)) {}

template <size_t Dim, typename Frame>
SpacetimeInterpolator<Dim, Frame>& SpacetimeInterpolator<Dim, Frame>::operator=(
    SpacetimeInterpolator&& rhs) {
  if (this != &rhs) {
    volume_files_or_glob_ = std::move(rhs.volume_files_or_glob_);
    subfile_name_ = std::move(rhs.subfile_name_);
    tensor_components_ = std::move(rhs.tensor_components_);
    functions_of_time_ = std::move(rhs.functions_of_time_);
    all_observation_ids_ = std::move(rhs.all_observation_ids_);
    all_observation_values_ = std::move(rhs.all_observation_values_);
    time_bounds_ = std::move(rhs.time_bounds_);
    interpolators_ = std::move(rhs.interpolators_);
  }
  return *this;
}

template <size_t Dim, typename Frame>
std::array<double, 2> SpacetimeInterpolator<Dim, Frame>::max_time_bounds()
    const {
  return {*(all_observation_values_.begin() + num_ghost_slices_),
          *(all_observation_values_.end() - num_ghost_slices_ - 1)};
}

template <size_t Dim, typename Frame>
void SpacetimeInterpolator<Dim, Frame>::load_time_bounds(
    std::array<double, 2> time_bounds) {
  ASSERT(time_bounds[0] < time_bounds[1],
         "The lower time bound must be smaller than the upper time bound.");
  ASSERT(time_bounds[0] >= max_time_bounds()[0] and
             time_bounds[1] <= max_time_bounds()[1],
         "The time bounds " << time_bounds
                            << " must be within the maximum time bounds "
                            << max_time_bounds() << ".");
  // Collect all observations that we need to load
  const auto lower_time =
      std::lower_bound(all_observation_values_.begin(),
                       all_observation_values_.end(), time_bounds[0],
                       std::less_equal<double>{}) -
      num_ghost_slices_ - 1;
  const auto upper_time =
      std::upper_bound(lower_time, all_observation_values_.end(),
                       time_bounds[1], std::less_equal<double>{}) +
      num_ghost_slices_ + 1;
  time_bounds_[0] = *(lower_time + num_ghost_slices_);
  time_bounds_[1] = *(upper_time - num_ghost_slices_ - 1);
  const auto lower_id =
      all_observation_ids_.begin() +
      std::distance(all_observation_values_.begin(), lower_time);
  const auto upper_id =
      all_observation_ids_.begin() +
      std::distance(all_observation_values_.begin(), upper_time);
  const auto num_obs = static_cast<size_t>(std::distance(lower_id, upper_id));
  ASSERT(num_obs >= time_interpolation_order_ + 1,
         "The number of time slices must be at least the time interpolation "
         "order plus one (NumTimeSlices >= "
             << time_interpolation_order_ + 1 << ").");

  if (interpolators_.empty()) {
    // No need to lock because we are loading interpolators for the first time
    // so no other thread can be accessing the data.
    for (auto it = lower_id; it != upper_id; ++it) {
      interpolators_.emplace_back(volume_files_or_glob_, subfile_name_,
                                  ObservationId{*it}, tensor_components_);
    }
    return;
  }

  // Drop interpolators that are not needed anymore. Do this early to free up
  // memory.
  auto first_interpolator_obs_id_it =
      std::find(all_observation_ids_.begin(), all_observation_ids_.end(),
                interpolators_.front().obs_id());
  ASSERT(first_interpolator_obs_id_it != all_observation_ids_.end(),
         "The first interpolator is not in the list of observation IDs.");
  const auto last_interpolator_obs_id_it =
      first_interpolator_obs_id_it +
      static_cast<std::ptrdiff_t>(interpolators_.size());
  const auto num_erase_front =
      std::distance(first_interpolator_obs_id_it, lower_id);
  const auto num_erase_back =
      std::distance(upper_id, last_interpolator_obs_id_it);
  if (num_erase_front == 0 and num_erase_back == 0) {
    // No need to drop or load interpolators
    return;
  }
  if (num_erase_front >= static_cast<std::ptrdiff_t>(interpolators_.size()) or
      num_erase_back >= static_cast<std::ptrdiff_t>(interpolators_.size())) {
    // We are dropping all interpolators and loading new ones. No need to lock
    // because there's no overlapping time range where old data can still be
    // used.
    interpolators_.clear();
    for (auto it = lower_id; it != upper_id; ++it) {
      interpolators_.emplace_back(volume_files_or_glob_, subfile_name_,
                                  ObservationId{*it}, tensor_components_);
    }
    return;
  }
  {
    // Lock while we are modifying the vector
    const std::unique_lock lock(interpolators_mutex_);
    if (num_erase_front > 0) {
      interpolators_.erase(interpolators_.begin(),
                           interpolators_.begin() + num_erase_front);
    }
    if (num_erase_back > 0) {
      interpolators_.erase(interpolators_.end() - num_erase_back,
                           interpolators_.end());
    }
  }

  // Load new interpolators
  std::vector<PointwiseInterpolator<Dim, ::Frame::Grid>>
      new_interpolators_front;
  std::vector<PointwiseInterpolator<Dim, ::Frame::Grid>> new_interpolators_back;
  if (num_erase_front < 0) {
    const auto num_add_front =
        std::min(static_cast<size_t>(-num_erase_front), num_obs);
    new_interpolators_front.reserve(num_add_front);
    for (auto it = lower_id;
         it != lower_id + static_cast<std::ptrdiff_t>(num_add_front); ++it) {
      new_interpolators_front.emplace_back(volume_files_or_glob_, subfile_name_,
                                           ObservationId{*it},
                                           tensor_components_);
    }
  }
  if (num_erase_back < 0) {
    const auto num_add_back =
        std::min(static_cast<size_t>(-num_erase_back),
                 num_obs - new_interpolators_front.size());
    new_interpolators_back.reserve(num_add_back);
    for (auto it = upper_id - static_cast<std::ptrdiff_t>(num_add_back);
         it != upper_id; ++it) {
      new_interpolators_back.emplace_back(volume_files_or_glob_, subfile_name_,
                                          ObservationId{*it},
                                          tensor_components_);
    }
  }
  {
    // Lock while we are modifying the vector
    const std::unique_lock lock(interpolators_mutex_);
    interpolators_.insert(
        interpolators_.begin(),
        std::make_move_iterator(new_interpolators_front.begin()),
        std::make_move_iterator(new_interpolators_front.end()));
    interpolators_.insert(
        interpolators_.end(),
        std::make_move_iterator(new_interpolators_back.begin()),
        std::make_move_iterator(new_interpolators_back.end()));
  }
}

template <size_t Dim, typename Frame>
void SpacetimeInterpolator<Dim, Frame>::interpolate_to_point(
    const gsl::not_null<std::vector<double>*> result,
    const tnsr::I<double, Dim, Frame>& target_point, const double time,
    const std::optional<gsl::not_null<std::vector<size_t>*>> block_order)
    const {
  // Lock to ensure that the `interpolators_` vector is not being modified while
  // we access it. Multiple threads can still read the data concurrently even
  // while loeading new data, just the short amount of time when the vector is
  // being modified is locked.
  const std::shared_lock lock(interpolators_mutex_);
  ASSERT(not interpolators_.empty(),
         "SpacetimeInterpolator has not been initialized.");
  // Find the time slice that contains the target time
  ASSERT(time >= time_bounds_[0] and time <= time_bounds_[1],
         "Requested time " << time << " is outside the time bounds.");
  const auto lower =
      std::lower_bound(interpolators_.begin() + num_ghost_slices_ + 1,
                       interpolators_.end() - num_ghost_slices_ - 1, time,
                       [](const auto& interpolator, const double t) {
                         return interpolator.time() <= t;
                       });
  // Transform target point to block-logical frame
  // NOTE: We assume here that the grid frame and even the block-logical frame
  // is the same in all time slices. This is true while the domain stays the
  // same (just functions of time values change to control the grid-to-inertial
  // map), but won't be true in general (e.g. with AMR regrids or ringdown
  // transitions). Then, we have to do the transformation for each time slice
  // separately (see https://arxiv.org/abs/1606.00437). This is not implemented
  // yet.
  const auto& any_interpolator = *lower;
  const auto block_logical_coords = block_logical_coordinates_single_point(
      target_point, any_interpolator.domain(), time, functions_of_time_,
      block_order);
  if (not block_logical_coords.has_value()) {
    ERROR("Point is not in any block:\n" << target_point);
  }
  // Interpolate to the grid-frame target point in each time slice
  std::array<std::vector<double>, time_interpolation_order_ + 1> data{};
  std::array<double, time_interpolation_order_ + 1> times{};
  const auto times_span = gsl::make_span(times);
  for (size_t i = 0; i < time_interpolation_order_ + 1; ++i) {
    const auto& interpolator =
        *(lower - 1 - static_cast<std::ptrdiff_t>(num_ghost_slices_) +
          static_cast<std::ptrdiff_t>(i));
    interpolator.interpolate_to_point(make_not_null(&gsl::at(data, i)),
                                      block_logical_coords.value());
    gsl::at(times, i) = interpolator.time();
  }
  // Interpolate in time
  const size_t num_components = tensor_components_.size();
  result->resize(num_components);
  std::array<double, time_interpolation_order_ + 1> values{};
  const auto values_span = gsl::make_span(values);
  double interpolation_error = 0.;
  for (size_t j = 0; j < num_components; ++j) {
    for (size_t k = 0; k < time_interpolation_order_ + 1; ++k) {
      gsl::at(values, k) = gsl::at(data, k)[j];
    }
    intrp::polynomial_interpolation<time_interpolation_order_>(
        make_not_null(&(*result)[j]), make_not_null(&interpolation_error), time,
        values_span, times_span);
  }
}

// Generate instantiations

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data) \
  template class SpacetimeInterpolator<DIM(data), FRAME(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (Frame::Inertial))

#undef INSTANTIATE
#undef DIM
#undef FRAME

}  // namespace spectre::Exporter
