// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSequence.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <pup_stl.h>
#include <utility>

#include "ErrorHandling/Assert.hpp"

namespace TimeSequences {
template <typename T>
EvenlySpaced<T>::EvenlySpaced(const T interval, const T offset,
                              const Options::Context& context)
    : interval_(static_cast<SignedT>(interval)),
      offset_(static_cast<SignedT>(offset)) {
  if (interval_ == 0) {
    PARSE_ERROR(context, "Interval must be positive.");
  }
}

template <typename T>
std::array<std::optional<T>, 3> EvenlySpaced<T>::times_near(
    const T time) const noexcept {
  auto sequence_number =
      std::floor((time + 0.5 * interval_ - offset_) / interval_);

  if (not std::is_signed_v<T> and offset_ + interval_ * sequence_number < 0) {
    ++sequence_number;
  }

  const auto previous = offset_ + interval_ * (sequence_number - 1);
  const auto current = offset_ + interval_ * sequence_number;
  const auto next = offset_ + interval_ * (sequence_number + 1);

  ASSERT(std::is_signed_v<T> or current >= 0,
         "Generated a negative value for an unsigned sequence.");

  return {{std::is_signed_v<T> or previous >= 0 ? previous : std::optional<T>{},
           current, next}};
}

template <typename T>
void EvenlySpaced<T>::pup(PUP::er& p) noexcept {
  TimeSequence<T>::pup(p);
  p | interval_;
  p | offset_;
}

/// \cond
template <typename T>
PUP::able::PUP_ID EvenlySpaced<T>::my_PUP_ID = 0;  // NOLINT
/// \endcond

template <typename T>
Specified<T>::Specified(std::vector<T> values) noexcept
    : values_(std::move(values)) {
  std::sort(values_.begin(), values_.end());
  const auto new_end = std::unique(values_.begin(), values_.end());
  values_.erase(new_end, values_.end());
}

template <typename T>
std::array<std::optional<T>, 3> Specified<T>::times_near(
    const T time) const noexcept {
  if (values_.empty()) {
    return {};
  }

  auto closest_time = std::lower_bound(values_.begin(), values_.end(), time);
  if (closest_time == values_.end() or
      (closest_time != values_.begin() and
       time - *(closest_time - 1) < *closest_time - time)) {
    --closest_time;
  }

  const auto next = closest_time + 1 == values_.end() ? std::optional<T>{}
                                                      : *(closest_time + 1);
  const auto previous = closest_time == values_.begin() ? std::optional<T>{}
                                                        : *(closest_time - 1);
  return {{previous, *closest_time, next}};
}

template <typename T>
void Specified<T>::pup(PUP::er& p) noexcept {
  TimeSequence<T>::pup(p);
  p | values_;
}

/// \cond
template <typename T>
PUP::able::PUP_ID Specified<T>::my_PUP_ID = 0;  // NOLINT
/// \endcond

// For time values
template class EvenlySpaced<double>;
template class Specified<double>;
// For slabs
template class EvenlySpaced<std::uint64_t>;
template class Specified<std::uint64_t>;
}  // namespace TimeSequences
