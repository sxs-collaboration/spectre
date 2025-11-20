// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <shared_mutex>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "Utilities/Gsl.hpp"

namespace spectre::Exporter {

/*!
 * \brief Interpolates data in volume files in both space and time
 *
 * This class reads the volume data at multiple time steps into memory and can
 * then interpolate it to any number of target points at any time within the
 * time bounds of the loaded data.
 *
 * \details The interpolation is first done in space and then in time. For the
 * interpolation in space, the PointwiseInterpolator class is used, which does
 * spectral interpolation within the elements. For the interpolation in time, a
 * polynomial interpolation with fixed order is used (currently third order,
 * meaning four time slices are used).
 *
 * \par Coordinate frames
 * The target points are given in the `Frame` specified as template parameter
 * (defaults to `Frame::Inertial`). However, the interpolation is done in the
 * grid frame, which is time-independent and therefore best suitable for
 * interpolation in time. Details of this idea can be found in
 * \cite Bohn:2016afc .
 *
 * \par Thread safety
 * Loading volume data from H5 files on multiple threads at once is not
 * generally thread safe, so `load_time_bounds` may only be called on a single
 * thread. However, once the data is loaded, the `interpolate_to_point` function
 * is thread safe. This means the volume data can be loaded once in a single
 * thread and then used by multiple threads to interpolate to different points
 * in parallel. Also, `load_time_bounds` can be called again to load new data
 * while other threads are still accessing the old data, as long as they only
 * request data within both the old and the new time bounds.
 */
template <size_t Dim, typename Frame = ::Frame::Inertial>
struct SpacetimeInterpolator {
  SpacetimeInterpolator() = default;

  /*!
   * \brief Construct the interpolator without loading any volume data
   *
   * \param volume_files_or_glob The list of H5 files, or a glob pattern
   * \param subfile_name The name of the subfile in the H5 files containing the
   * volume data
   * \param tensor_components The tensor components to interpolate. The order of
   * the components in the vector is the order in which they will be returned.
   */
  SpacetimeInterpolator(
      std::variant<std::vector<std::string>, std::string> volume_files_or_glob,
      std::string subfile_name, std::vector<std::string> tensor_components);

  /// The maximum time bounds available in the volume data files (taking into
  /// account the number of ghost slices needed for interpolation). Only time
  /// bounds in this (inclusive) range can be requested with the
  /// `load_time_bounds()` function.
  std::array<double, 2> max_time_bounds() const;

  /*!
   * \brief Load volume data into memory such that the given time bounds are
   * covered.
   *
   * The given time bounds must be within the `max_time_bounds()` (inclusive).
   *
   * Previously loaded volume data slices outside the given time bounds will be
   * dropped and new ones loaded.
   *
   * \par Thread safety
   * This function is not generally thread safe, as it loads data from H5 files.
   * It should be called from a single thread only. It is safe to keep calling
   * interpolation routines from other threads while this function is running,
   * as long as they are requesting data within both the old and the new time
   * bounds.
   */
  void load_time_bounds(std::array<double, 2> time_bounds);

  /// The time bounds of the loaded data (inclusive). It is an error to call the
  /// interpolation routines outside these bounds.
  std::array<double, 2> time_bounds() const { return time_bounds_; }

  /// The number of loaded time slices.
  size_t num_loaded_slices() const { return interpolators_.size(); }

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
   * \param target_point the point to interpolate to.
   * \param time the time to interpolate to. Must be within the `time_bounds()`.
   * \param block_order an optional priority order to search for the block
   * containing the target point. Will be updated when the point is found.
   * See `block_logical_coordinates_single_point` for more details.
   */
  void interpolate_to_point(gsl::not_null<std::vector<double>*> result,
                            const tnsr::I<double, Dim, Frame>& target_point,
                            double time,
                            std::optional<gsl::not_null<std::vector<size_t>*>>
                                block_order = std::nullopt) const;

 private:
  std::variant<std::vector<std::string>, std::string> volume_files_or_glob_;
  std::string subfile_name_;
  std::vector<std::string> tensor_components_;
  constexpr static size_t time_interpolation_order_ = 3;  // must be odd
  constexpr static size_t num_ghost_slices_ =
      (time_interpolation_order_ + 1) / 2 - 1;
  domain::FunctionsOfTimeMap functions_of_time_;
  // Metadata collected from data files
  std::vector<size_t> all_observation_ids_;
  std::vector<double> all_observation_values_;
  // State
  std::array<double, 2> time_bounds_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  // Loaded from data files
  std::vector<PointwiseInterpolator<Dim, ::Frame::Grid>> interpolators_{};
  // Mutex to protect other threads from accessing the `interpolators_` vector
  // while it is being modified
  mutable std::shared_mutex interpolators_mutex_;  // NOLINT(spectre-mutable)
};

}  // namespace spectre::Exporter
