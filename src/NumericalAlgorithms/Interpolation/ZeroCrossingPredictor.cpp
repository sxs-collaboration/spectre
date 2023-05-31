// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Interpolation/ZeroCrossingPredictor.hpp"

#include <algorithm>
#include <limits>
#include <pup.h>
#include <pup_stl.h>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Interpolation/PredictedZeroCrossing.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace intrp {
ZeroCrossingPredictor::ZeroCrossingPredictor(const size_t min_size,
                                             const size_t max_size)
    : min_size_(min_size), max_size_(max_size) {
  if (min_size_ < 3) {
    ERROR("min_size must be >= 3, not " << min_size_);
  }
  if (min_size_ > max_size_) {
    ERROR("min_size must be <= max_size, but " << min_size_ << " > "
                                               << max_size_);
  }
}

void ZeroCrossingPredictor::add(const double t, DataVector data_at_time_t) {
  times_.push_back(t);
  data_.emplace_back(std::move(data_at_time_t));

  while (times_.size() > max_size_) {
    times_.pop_front();
    data_.pop_front();
  }
}

DataVector ZeroCrossingPredictor::zero_crossing_time(
    const double current_time) const {
  if (not is_valid()) {
    ERROR("Invalid ZeroCrossingPredictor. min_size = "
          << min_size_ << " times_.size() = " << times_.size());
  }

  // We need times relative to current_time.
  std::deque<double> relative_times(times_.begin(), times_.end());
  for (auto& time : relative_times) {
    time -= current_time;
  }
  return predicted_zero_crossing_value(relative_times, data_);
}

double ZeroCrossingPredictor::min_positive_zero_crossing_time(
    double current_time) const {
  if (not is_valid()) {
    return 0.0;
  }
  auto crossing_time = zero_crossing_time(current_time);
  // Replace all negative crossing times with infinity.
  std::for_each(crossing_time.begin(), crossing_time.end(), [](double& a) {
    a = a < 0.0 ? std::numeric_limits<double>::infinity() : a;
  });

  return *std::min_element(crossing_time.begin(), crossing_time.end());
}

bool ZeroCrossingPredictor::is_valid() const {
  return times_.size() >= min_size_;
}

void ZeroCrossingPredictor::clear() {
  times_.clear();
  data_.clear();
}

void ZeroCrossingPredictor::pup(PUP::er& p) {
  p | min_size_;
  p | max_size_;
  p | times_;
  p | data_;
}

bool operator==(const ZeroCrossingPredictor& predictor1,
                const ZeroCrossingPredictor& predictor2) {
  return (predictor1.min_size_ == predictor2.min_size_) and
         (predictor1.max_size_ == predictor2.max_size_) and
         (predictor1.times_ == predictor2.times_) and
         (predictor1.data_ == predictor2.data_);
}

bool operator!=(const ZeroCrossingPredictor& predictor1,
                const ZeroCrossingPredictor& predictor2) {
  return not(predictor1 == predictor2);
}

}  // namespace intrp
