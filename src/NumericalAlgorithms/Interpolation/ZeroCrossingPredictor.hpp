// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>

#include "DataStructures/DataVector.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace intrp {

/// \ingroup NumericalAlgorithmsGroup
/// \brief A class that predicts when a function crosses zero
class ZeroCrossingPredictor {
 public:
  /// Uses at most max_size times for the fit; throws away old
  /// times as new times are added.
  /// The is_valid function returns false until min_size times
  /// have been added.  min_size must be at least 3.
  ZeroCrossingPredictor(size_t min_size, size_t max_size);

  ZeroCrossingPredictor() = default;

  /// Adds a data point at time t to the ZeroCrossingPredictor.
  void add(double t, DataVector data_at_time_t);

  /// For each component of the data, returns the time, relative to
  /// current_time, that a linear fit to the given component of the
  /// data crosses zero.  The length of the return value is the same
  /// as the length of `data_at_time_t` in the `add` function.
  ///
  /// The zero-crossing time that is computed has error bars
  /// associated with it.  If the error bars are large enough that it
  /// is not clear whether the zero-crossing time is positive or
  /// negative, then zero is returned instead of the best-fit
  /// zero-crossing time.
  DataVector zero_crossing_time(double current_time) const;

  /// The minimum positive value over the DataVector returned
  /// by zero_crossing_time.  Returns zero if is_valid() is false.
  double min_positive_zero_crossing_time(double current_time) const;

  /// Returns whether we have enough data to call zero_crossing_time.
  bool is_valid() const;

  /// Clears the internal arrays.  Used to reset if there is a
  /// discontinuous change in the data that should not be fit over.
  void clear();

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  friend bool operator==(const ZeroCrossingPredictor&,   // NOLINT
                         const ZeroCrossingPredictor&);  // NOLINT

 private:
  size_t min_size_{3};
  size_t max_size_{3};
  std::deque<double> times_{};
  std::deque<DataVector> data_{};
};

bool operator!=(const ZeroCrossingPredictor&, const ZeroCrossingPredictor&);

}  // namespace intrp
