// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <complex>
#include <cstddef>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "ErrorHandling/Assert.hpp"
#include "ErrorHandling/Error.hpp"
#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/MakeArray.hpp"

namespace Cce {

/*!
 * \brief Stores necessary data and interpolates on to new time points at scri+.
 *
 * \details The coordinate time used for the CCE evolution is not the same as
 * the asymptotic inertial retarded time, which is determined through a separate
 * evolution equation. This class manages the scri+ data passed in (via
 * `insert_data()`) along with the set of inertial retarded times associated
 * with that data, and interpolates to a set of requested times (supplied via
 * `insert_target_time()`).
 *
 * Template parameters:
 * - `VectorTypeToInterpolate`: the vector type associated with the values to
 * interpolate.
 * - `Tag`: The tag associated with the interpolation procedure. This determines
 * the behavior of the interpolation return value. The default is just
 * interpolation, if `Tag` has prefix `::Tags::Multiplies` or `Tags::Du`, the
 * interpolator performs the additional multiplication or time derivative as a
 * step in the interpolation procedure.
 */
template <typename VectorTypeToInterpolate, typename Tag>
struct ScriPlusInterpolationManager {
 public:
  ScriPlusInterpolationManager(
      const size_t target_number_of_points, const size_t vector_size,
      std::unique_ptr<intrp::SpanInterpolator> interpolator) noexcept
      : vector_size_{vector_size},
        target_number_of_points_{target_number_of_points},
        interpolator_{std::move(interpolator)} {}

  /// \brief provide data to the interpolation manager.
  ///
  /// \details `u_bondi` is a vector of inertial times, and `to_interpolate` is
  /// the vector of values that will be interpolated to target times.
  void insert_data(const DataVector& u_bondi,
                   const VectorTypeToInterpolate& to_interpolate) noexcept {
    ASSERT(to_interpolate.size() == vector_size_,
           "Inserted data must be of size specified at construction: "
               << vector_size_
               << " and provided data is of size: " << to_interpolate.size());
    u_bondi_values_.push_back(u_bondi);
    to_interpolate_values_.push_back(to_interpolate);
    u_bondi_ranges_.emplace_back(min(u_bondi), max(u_bondi));
  }

  /// \brief Request a target time to be interpolated to when enough data has
  /// been accumulated.
  ///
  /// For optimization, we assume that these are inserted in ascending order.
  void insert_target_time(const double time) noexcept {
    target_times_.push_back(time);
    // if this is the first time in the deque, we should use it to determine
    // whether some of the old data is ready to be removed. In other cases, this
    // check can be performed when a time is removed from the queue via
    // `interpolate_and_pop_first_time`
    if (target_times_.size() == 1) {
      remove_unneeded_early_times();
    }
  }

  /// \brief Determines whether enough data before and after the first time in
  /// the target time queue has been provided to interpolate.
  ///
  /// \details If possible, this function will require that the target time to
  /// be interpolated is reasonably centered on the range, but will settle for
  /// non-centered data if the time is too early for the given data, which is
  /// necessary to get the first couple of times out of the simulation. This
  /// function always returns false if all of the provided data is earlier than
  /// the next target time, indicating that the caller should wait until more
  /// data has been provided before interpolating.
  bool first_time_is_ready_to_interpolate() const noexcept;

  const std::deque<std::pair<double, double>>& get_u_bondi_ranges() const
      noexcept {
    return u_bondi_ranges_;
  }

  /// \brief Interpolate to the first target time in the queue, returning both
  /// the time and the interpolated data at that time.
  ///
  /// \note If this function is not able to interpolate to the full accuracy
  /// using a centered stencil, it will perform the best interpolation
  /// available.
  ///
  /// \warning This function will extrapolate if the target times are
  /// outside the range of data the class has been provided. This is intentional
  /// to support small extrapolation at the end of a simulation when no further
  /// data is available, but for full accuracy, check
  /// `first_time_is_ready_to_interpolate` before calling the interpolation
  /// functions
  std::pair<double, VectorTypeToInterpolate> interpolate_first_time() noexcept;

  /// \brief Interpolate to the first target time in the queue, returning both
  /// the time and the interpolated data at that time, and remove the first time
  /// from the queue.
  ///
  /// \note If this function is not able to interpolate to the full accuracy
  /// using a centered stencil, it will perform the best interpolation
  /// available.
  ///
  /// \warning This function will extrapolate if the target times are
  /// outside the range of data the class has been provided. This is intentional
  /// to support small extrapolation at the end of a simulation when no further
  /// data is available, but for full accuracy, check
  /// `first_time_is_ready_to_interpolate` before calling the interpolation
  /// functions
  std::pair<double, VectorTypeToInterpolate>
  interpolate_and_pop_first_time() noexcept;

  /// \brief return the number of times in the target times queue
  size_t number_of_target_times() const noexcept {
    return target_times_.size();
  }

  /// \brief return the number of data points that have been provided to the
  /// interpolation manager
  size_t number_of_data_points() const noexcept {
    return u_bondi_ranges_.size();
  }

 private:
  void remove_unneeded_early_times() noexcept;

  friend struct ScriPlusInterpolationManager<VectorTypeToInterpolate,
                                             Tags::Du<Tag>>;

  std::deque<DataVector> u_bondi_values_;
  std::deque<VectorTypeToInterpolate> to_interpolate_values_;
  std::deque<std::pair<double, double>> u_bondi_ranges_;
  std::deque<double> target_times_;
  size_t vector_size_;
  size_t target_number_of_points_;
  std::unique_ptr<intrp::SpanInterpolator> interpolator_;
};

template <typename VectorTypeToInterpolate, typename Tag>
bool ScriPlusInterpolationManager<
    VectorTypeToInterpolate, Tag>::first_time_is_ready_to_interpolate() const
    noexcept {
  if (target_times_.empty()) {
    return false;
  }
  auto maxes_below = alg::count_if(
      u_bondi_ranges_, [this](const std::pair<double, double> time) noexcept {
        return time.second <= target_times_.front();
      });
  auto mins_above = alg::count_if(
      u_bondi_ranges_, [this](const std::pair<double, double> time) noexcept {
        return time.first > target_times_.front();
      });

  // we might ask for a time that's too close to the end or the beginning of
  // our data, in which case we will settle for at least one point below and
  // above and a sufficient number of total points.
  // This will always be `false` if the earliest target time is later than all
  // provided data points, which would require extrapolation
  return ((static_cast<size_t>(maxes_below) > target_number_of_points_ or
           static_cast<size_t>(maxes_below + mins_above) >
               2 * target_number_of_points_) and
          static_cast<size_t>(mins_above) > target_number_of_points_);
}

template <typename VectorTypeToInterpolate, typename Tag>
std::pair<double, VectorTypeToInterpolate> ScriPlusInterpolationManager<
    VectorTypeToInterpolate, Tag>::interpolate_first_time() noexcept {
  if (target_times_.empty()) {
    ERROR("There are no target times to interpolate.");
  }
  if (to_interpolate_values_.size() < 2 * target_number_of_points_) {
    ERROR("Insufficient data points to continue interpolation: have "
          << to_interpolate_values_.size() << ", need at least"
          << 2 * target_number_of_points_);
  }

  VectorTypeToInterpolate result{vector_size_};
  const size_t interpolation_data_size = to_interpolate_values_.size();

  VectorTypeToInterpolate interpolation_values{2 * target_number_of_points_};
  DataVector interpolation_times{2 * target_number_of_points_};
  for (size_t i = 0; i < vector_size_; ++i) {
    // binary search assumes times placed in sorted order
    auto upper_bound_offset = static_cast<size_t>(std::distance(
        u_bondi_values_.begin(),
        std::upper_bound(
            u_bondi_values_.begin(), u_bondi_values_.end(),
            target_times_.front(),
            [&i](const double rhs, const DataVector& lhs) noexcept {
              return rhs < lhs[i];
            })));
    size_t lower_bound_offset =
        upper_bound_offset == 0 ? 0 : upper_bound_offset - 1;

    if (upper_bound_offset + target_number_of_points_ >
        interpolation_data_size) {
      upper_bound_offset = interpolation_data_size;
      lower_bound_offset =
          interpolation_data_size - 2 * target_number_of_points_;
    } else if (lower_bound_offset < target_number_of_points_ - 1) {
      lower_bound_offset = 0;
      upper_bound_offset = 2 * target_number_of_points_;
    } else {
      lower_bound_offset = lower_bound_offset + 1 - target_number_of_points_;
      upper_bound_offset = lower_bound_offset + 2 * target_number_of_points_;
    }
    auto interpolation_values_begin =
        to_interpolate_values_.begin() +
        static_cast<ptrdiff_t>(lower_bound_offset);
    auto interpolation_times_begin =
        u_bondi_values_.begin() + static_cast<ptrdiff_t>(lower_bound_offset);
    auto interpolation_times_end =
        u_bondi_values_.begin() + static_cast<ptrdiff_t>(upper_bound_offset);

    // interpolate using the data sets in the restricted iterators
    auto value_it = interpolation_values_begin;
    size_t vector_position = 0;
    for (auto time_it = interpolation_times_begin;
         time_it != interpolation_times_end;
         ++time_it, ++value_it, ++vector_position) {
      interpolation_values[vector_position] = (*value_it)[i];
      interpolation_times[vector_position] = (*time_it)[i];
    }
    result[i] = interpolator_->interpolate(
        gsl::span<const double>(interpolation_times.data(),
                                interpolation_times.size()),
        gsl::span<const typename VectorTypeToInterpolate::value_type>(
            interpolation_values.data(), interpolation_values.size()),
        target_times_.front());
  }
  return std::make_pair(target_times_.front(), std::move(result));
}

template <typename VectorTypeToInterpolate, typename Tag>
std::pair<double, VectorTypeToInterpolate> ScriPlusInterpolationManager<
    VectorTypeToInterpolate, Tag>::interpolate_and_pop_first_time() noexcept {
  std::pair<double, VectorTypeToInterpolate> interpolated =
      interpolate_first_time();
  target_times_.pop_front();

  if (not target_times_.empty()) {
    remove_unneeded_early_times();
  }
  return interpolated;
}

template <typename VectorTypeToInterpolate, typename Tag>
void ScriPlusInterpolationManager<VectorTypeToInterpolate,
                                  Tag>::remove_unneeded_early_times() noexcept {
  // pop times we no longer need because their maxes are too far in the past
  auto time_it = u_bondi_ranges_.begin();
  size_t times_counter = 0;
  while (time_it < u_bondi_ranges_.end() and
         (*time_it).second < target_times_.front()) {
    if (times_counter > target_number_of_points_ and
        u_bondi_ranges_.size() >= 2 * target_number_of_points_) {
      u_bondi_ranges_.pop_front();
      u_bondi_values_.pop_front();
      to_interpolate_values_.pop_front();
    } else {
      ++times_counter;
    }
    ++time_it;
  }
}

/*!
 * \brief Stores necessary data and interpolates on to new time points at scri+,
 * multiplying two results together before supplying the result.
 *
 * \details The coordinate time used for the CCE evolution is not the same as
 * the asymptotic inertial retarded time, which is determined through a separate
 * evolution equation. This class manages the two sets of  scri+ data passed in
 * (via `insert_data()`) along with the set of inertial retarded times
 * associated with that data, and interpolates to a set of inertial requested
 * times (supplied via `insert_target_time()`), multiplying the two
 * interpolation results together before returning.
 *
 * Template parameters:
 * - `VectorTypeToInterpolate`: the vector type associated with the values to
 * interpolate.
 * - `::Tags::Multiplies<MultipliesLhs, MultipliesRhs>`: The tag associated with
 * the interpolation procedure. This determines the behavior of the
 * interpolation return value. The default is just interpolation, if `Tag` has
 * prefix  `::Tags::Multiplies` (this case) or `Tags::Du`, the interpolator
 * performs the additional multiplication or time derivative as a step in the
 * interpolation procedure.
 */
template <typename VectorTypeToInterpolate, typename MultipliesLhs,
          typename MultipliesRhs>
struct ScriPlusInterpolationManager<
    VectorTypeToInterpolate, ::Tags::Multiplies<MultipliesLhs, MultipliesRhs>> {
 public:
  ScriPlusInterpolationManager(
      const size_t target_number_of_points, const size_t vector_size,
      std::unique_ptr<intrp::SpanInterpolator> interpolator) noexcept
      : interpolation_manager_lhs_{target_number_of_points, vector_size,
                                   interpolator->get_clone()},
        interpolation_manager_rhs_{target_number_of_points, vector_size,
                                   std::move(interpolator)} {}

  /// \brief provide data to the interpolation manager.
  ///
  /// \details `u_bondi` is a vector of inertial times, and `to_interpolate_lhs`
  /// and `to_interpolate_rhs` are the vector of values that will be
  /// interpolated to target times. The interpolation result will be the product
  /// of the interpolated lhs and rhs vectors.
  void insert_data(const DataVector& u_bondi,
                   const VectorTypeToInterpolate& to_interpolate_lhs,
                   const VectorTypeToInterpolate& to_interpolate_rhs) noexcept {
    interpolation_manager_lhs_.insert_data(u_bondi, to_interpolate_lhs);
    interpolation_manager_rhs_.insert_data(u_bondi, to_interpolate_rhs);
  }

  /// \brief Request a target time to be interpolated to when enough data has
  /// been accumulated.
  ///
  /// For optimization, we assume that these are inserted in ascending order.
  void insert_target_time(const double time) noexcept {
    interpolation_manager_lhs_.insert_target_time(time);
    interpolation_manager_rhs_.insert_target_time(time);
  }

  /// \brief Determines whether enough data before and after the first time in
  /// the target time queue has been provided to interpolate.
  ///
  /// \details If possible, this function will require that the target time to
  /// be interpolated is reasonably centered on the range, but will settle for
  /// non-centered data if the time is too early for the given data, which is
  /// necessary to get the first couple of times out of the simulation. This
  /// function always returns false if all of the provided data is earlier than
  /// the next target time, indicating that the caller should wait until more
  /// data has been provided before interpolating.
  bool first_time_is_ready_to_interpolate() const noexcept {
    return interpolation_manager_lhs_.first_time_is_ready_to_interpolate() and
           interpolation_manager_rhs_.first_time_is_ready_to_interpolate();
  }

  const std::deque<std::pair<double, double>>& get_u_bondi_ranges() const
      noexcept {
    return interpolation_manager_lhs_.get_u_bondi_ranges();
  }

  /// \brief Interpolate to the first target time in the queue, returning both
  /// the time and the interpolated-and-multiplied data at that time.
  ///
  /// \note If this function is not able to interpolate to the full accuracy
  /// using a centered stencil, it will perform the best interpolation
  /// available.
  ///
  /// \warning This function will extrapolate if the target times are
  /// outside the range of data the class has been provided. This is intentional
  /// to support small extrapolation at the end of a simulation when no further
  /// data is available, but for full accuracy, check
  /// `first_time_is_ready_to_interpolate` before calling the interpolation
  /// functions
  std::pair<double, VectorTypeToInterpolate> interpolate_first_time() noexcept {
    const auto lhs_interpolation =
        interpolation_manager_lhs_.interpolate_first_time();
    const auto rhs_interpolation =
        interpolation_manager_rhs_.interpolate_first_time();
    return std::make_pair(lhs_interpolation.first,
                          lhs_interpolation.second * rhs_interpolation.second);
  }

  /// \brief Interpolate to the first target time in the queue, returning both
  /// the time and the interpolated-and-multiplied data at that time, and remove
  /// the first time from the queue.
  ///
  /// \note If this function is not able to interpolate to the full accuracy
  /// using a centered stencil, it will perform the best interpolation
  /// available.
  ///
  /// \warning This function will extrapolate if the target times are
  /// outside the range of data the class has been provided. This is intentional
  /// to support small extrapolation at the end of a simulation when no further
  /// data is available, but for full accuracy, check
  /// `first_time_is_ready_to_interpolate` before calling the interpolation
  /// functions
  std::pair<double, VectorTypeToInterpolate>
  interpolate_and_pop_first_time() noexcept {
    const auto lhs_interpolation =
        interpolation_manager_lhs_.interpolate_and_pop_first_time();
    const auto rhs_interpolation =
        interpolation_manager_rhs_.interpolate_and_pop_first_time();
    return std::make_pair(lhs_interpolation.first,
                          lhs_interpolation.second * rhs_interpolation.second);
  }

  /// \brief return the number of times in the target times queue
  size_t number_of_target_times() const noexcept {
    return interpolation_manager_lhs_.number_of_target_times();
  }

  /// \brief return the number of data points that have been provided to the
  /// interpolation manager
  size_t number_of_data_points() const noexcept {
    return interpolation_manager_lhs_.number_of_data_points();
  }

 private:
  /// \cond
  ScriPlusInterpolationManager<VectorTypeToInterpolate, MultipliesLhs>
      interpolation_manager_lhs_;
  ScriPlusInterpolationManager<VectorTypeToInterpolate, MultipliesRhs>
      interpolation_manager_rhs_;
  /// \endcond
};

/*!
 * \brief Stores necessary data and interpolates on to new time points at scri+,
 * differentiating before supplying the result.
 *
 * \details The coordinate time used for the CCE evolution is not the same as
 * the asymptotic inertial retarded time, which is determined through a separate
 * evolution equation. This class manages the scri+ data passed in
 * (via `insert_data()`) along with the set of inertial retarded times
 * associated with that data, and interpolates to a set of requested times
 * (supplied via `insert_target_time()`), differentiating the interpolation
 * before returning.
 *
 * Template parameters:
 * - `VectorTypeToInterpolate`: the vector type associated with the values to
 * interpolate.
 * - `Tags::Du<Tag>`: The tag associated with the interpolation procedure. This
 * determines the behavior of the interpolation return value. The default is
 * just interpolation, if `Tag` has prefix  `::Tags::Multiplies` or `Tags::Du`
 * (this case), the interpolator performs the additional multiplication or time
 * derivative as a step in the interpolation procedure.
 */
template <typename VectorTypeToInterpolate, typename Tag>
struct ScriPlusInterpolationManager<VectorTypeToInterpolate, Tags::Du<Tag>> {
 public:
  ScriPlusInterpolationManager(
      const size_t target_number_of_points, const size_t vector_size,
      std::unique_ptr<intrp::SpanInterpolator> interpolator) noexcept
      : argument_interpolation_manager_{target_number_of_points, vector_size,
                                        std::move(interpolator)} {}

  /// \brief provide data to the interpolation manager.
  ///
  /// \details `u_bondi` is a vector of
  /// inertial times, and `to_interpolate_argument` is the vector of values that
  /// will be interpolated to target times and differentiated.
  void insert_data(
      const DataVector& u_bondi,
      const VectorTypeToInterpolate& to_interpolate_argument) noexcept {
    argument_interpolation_manager_.insert_data(u_bondi,
                                                to_interpolate_argument);
  }

  /// \brief Request a target time to be interpolated to when enough data has
  /// been accumulated.
  ///
  /// For optimization, we assume that these are inserted in ascending order.
  void insert_target_time(const double time) noexcept {
    argument_interpolation_manager_.insert_target_time(time);
  }

  /// \brief Determines whether enough data before and after the first time in
  /// the target time queue has been provided to interpolate.
  ///
  /// \details If possible, this function will require that the target time to
  /// be interpolated is reasonably centered on the range, but will settle for
  /// non-centered data if the time is too early for the given data, which is
  /// necessary to get the first couple of times out of the simulation. This
  /// function always returns false if all of the provided data is earlier than
  /// the next target time, indicating that the caller should wait until more
  /// data has been provided before interpolating.
  bool first_time_is_ready_to_interpolate() const noexcept {
    return argument_interpolation_manager_.first_time_is_ready_to_interpolate();
  }

  const std::deque<std::pair<double, double>>& get_u_bondi_ranges() const
      noexcept {
    return argument_interpolation_manager_.get_u_bondi_ranges();
  }

  /// \brief Interpolate to the first target time in the queue, returning both
  /// the time and the interpolated data at that time.
  ///
  /// \note If this function is not able to interpolate to the full accuracy
  /// using a centered stencil, it will perform the best interpolation
  /// available.
  ///
  /// \warning This function will extrapolate if the target times are
  /// outside the range of data the class has been provided. This is intentional
  /// to support small extrapolation at the end of a simulation when no further
  /// data is available, but for full accuracy, check
  /// `first_time_is_ready_to_interpolate` before calling the interpolation
  /// functions
  std::pair<double, VectorTypeToInterpolate> interpolate_first_time() noexcept;

  /// \brief Interpolate to the first target time in the queue, returning both
  /// the time and the interpolated data at that time, and remove the first time
  /// from the queue.
  ///
  /// \note If this function is not able to interpolate to the full accuracy
  /// using a centered stencil, it will perform the best interpolation
  /// available.
  ///
  /// \warning This function will extrapolate if the target times are
  /// outside the range of data the class has been provided. This is intentional
  /// to support small extrapolation at the end of a simulation when no further
  /// data is available, but for full accuracy, check
  /// `first_time_is_ready_to_interpolate` before calling the interpolation
  /// functions
  std::pair<double, VectorTypeToInterpolate>
  interpolate_and_pop_first_time() noexcept;

  /// \brief return the number of times in the target times queue
  size_t number_of_target_times() const noexcept {
    return argument_interpolation_manager_.number_of_target_times();
  }

  /// \brief return the number of data points that have been provided to the
  /// interpolation manager
  size_t number_of_data_points() const noexcept {
    return argument_interpolation_manager_.number_of_data_points();
  }

 private:
  // to avoid code duplication, most of the details are stored in an
  // interpolation manager for the argument tag, and this class is a friend of
  // the argument tag class
  /// \cond
  ScriPlusInterpolationManager<VectorTypeToInterpolate, Tag>
      argument_interpolation_manager_;
  /// \endcond
};

template <typename VectorTypeToInterpolate, typename Tag>
std::pair<double, VectorTypeToInterpolate> ScriPlusInterpolationManager<
    VectorTypeToInterpolate, Tags::Du<Tag>>::interpolate_first_time() noexcept {
  const size_t target_number_of_points =
      argument_interpolation_manager_.target_number_of_points_;
  if (argument_interpolation_manager_.target_times_.empty()) {
    ERROR("There are no target times to interpolate.");
  }
  if (argument_interpolation_manager_.to_interpolate_values_.size() <
      2 * argument_interpolation_manager_.target_number_of_points_) {
    ERROR("Insufficient data points to continue interpolation: have "
          << argument_interpolation_manager_.to_interpolate_values_.size()
          << ", need at least"
          << 2 * argument_interpolation_manager_.target_number_of_points_);
  }
  // note that because we demand at least a certain number before and at least
  // a certain number after, we are likely to have a surfeit of points for the
  // interpolator, but this should not cause significant trouble for a
  // reasonable method.
  VectorTypeToInterpolate result{argument_interpolation_manager_.vector_size_};

  VectorTypeToInterpolate interpolation_values{2 * target_number_of_points};
  VectorTypeToInterpolate lobatto_collocation_values{2 *
                                                     target_number_of_points};
  VectorTypeToInterpolate derivative_lobatto_collocation_values{
      2 * target_number_of_points};
  DataVector interpolation_times{2 * target_number_of_points};
  DataVector collocation_points =
      Spectral::collocation_points<Spectral::Basis::Legendre,
                                   Spectral::Quadrature::GaussLobatto>(
          2 * target_number_of_points);

  const size_t interpolation_data_size =
      argument_interpolation_manager_.to_interpolate_values_.size();

  for (size_t i = 0; i < argument_interpolation_manager_.vector_size_; ++i) {
    // binary search assumes times placed in sorted order
    auto upper_bound_offset = static_cast<size_t>(std::distance(
        argument_interpolation_manager_.u_bondi_values_.begin(),
        std::upper_bound(
            argument_interpolation_manager_.u_bondi_values_.begin(),
            argument_interpolation_manager_.u_bondi_values_.end(),
            argument_interpolation_manager_.target_times_.front(),
            [&i](const double rhs, const DataVector& lhs) noexcept {
              return rhs < lhs[i];
            })));
    size_t lower_bound_offset =
        upper_bound_offset == 0 ? 0 : upper_bound_offset - 1;

    if (upper_bound_offset + target_number_of_points >
        interpolation_data_size) {
      upper_bound_offset = interpolation_data_size;
      lower_bound_offset =
          interpolation_data_size - 2 * target_number_of_points;
    } else if (lower_bound_offset < target_number_of_points - 1) {
      lower_bound_offset = 0;
      upper_bound_offset = 2 * target_number_of_points;
    } else {
      lower_bound_offset = lower_bound_offset + 1 - target_number_of_points;
      upper_bound_offset = lower_bound_offset + 2 * target_number_of_points;
    }
    auto interpolation_values_begin =
        argument_interpolation_manager_.to_interpolate_values_.begin() +
        static_cast<ptrdiff_t>(lower_bound_offset);
    auto interpolation_times_begin =
        argument_interpolation_manager_.u_bondi_values_.begin() +
        static_cast<ptrdiff_t>(lower_bound_offset);
    auto interpolation_times_end =
        argument_interpolation_manager_.u_bondi_values_.begin() +
        static_cast<ptrdiff_t>(upper_bound_offset);

    // interpolate using the data sets in the restricted iterators
    auto value_it = interpolation_values_begin;
    size_t vector_position = 0;
    for (auto time_it = interpolation_times_begin;
         time_it != interpolation_times_end;
         ++time_it, ++value_it, ++vector_position) {
      interpolation_values[vector_position] = (*value_it)[i];
      interpolation_times[vector_position] = (*time_it)[i];
    }
    for (size_t j = 0; j < lobatto_collocation_values.size(); ++j) {
      lobatto_collocation_values[j] =
          argument_interpolation_manager_.interpolator_->interpolate(
              gsl::span<const double>(interpolation_times.data(),
                                      interpolation_times.size()),
              gsl::span<const typename VectorTypeToInterpolate::value_type>(
                  interpolation_values.data(), interpolation_values.size()),
              // affine transformation between the Gauss-Lobatto collocation
              // points and the physical times
              (collocation_points[j] + 1.0) * 0.5 *
                      (interpolation_times[interpolation_times.size() - 1] -
                       interpolation_times[0]) +
                  interpolation_times[0]);
    }
    // note the coordinate transformation to and from the Gauss-Lobatto basis
    // range [-1, 1]
    apply_matrices(
        make_not_null(&derivative_lobatto_collocation_values),
        make_array<1>(
            Spectral::differentiation_matrix<
                Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>(
                lobatto_collocation_values.size())),
        lobatto_collocation_values,
        Index<1>(lobatto_collocation_values.size()));

    result[i] =
        argument_interpolation_manager_.interpolator_->interpolate(
            gsl::span<const double>(collocation_points.data(),
                                    collocation_points.size()),
            gsl::span<const typename VectorTypeToInterpolate::value_type>(
                derivative_lobatto_collocation_values.data(),
                derivative_lobatto_collocation_values.size()),
            2.0 *
                    (argument_interpolation_manager_.target_times_.front() -
                     interpolation_times[0]) /
                    (interpolation_times[interpolation_times.size() - 1] -
                     interpolation_times[0]) -
                1.0) *
        2.0 /
        (interpolation_times[interpolation_times.size() - 1] -
         interpolation_times[0]);
  }
  return std::make_pair(argument_interpolation_manager_.target_times_.front(),
                        std::move(result));
}

template <typename VectorTypeToInterpolate, typename Tag>
std::pair<double, VectorTypeToInterpolate> ScriPlusInterpolationManager<
    VectorTypeToInterpolate,
    Tags::Du<Tag>>::interpolate_and_pop_first_time() noexcept {
  std::pair<double, VectorTypeToInterpolate> interpolated =
      interpolate_first_time();
  argument_interpolation_manager_.target_times_.pop_front();

  if (not argument_interpolation_manager_.target_times_.empty()) {
    argument_interpolation_manager_.remove_unneeded_early_times();
  }
  return interpolated;
}
}  // namespace Cce
