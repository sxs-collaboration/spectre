// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>
#include <vector>

#include "Utilities/Gsl.hpp"

/// Iterate over a generic "block" of coordinates.  The constructor
/// takes a container of pairs defining the range (inclusive start,
/// exclusive end) of each coordinate.  The utility function
/// make_general_index_iterator() is provided for easier construction.
template <typename Container>
class GeneralIndexIterator {
 public:
  using value_type = typename Container::value_type::first_type;
  using iterator_type = typename std::vector<value_type>::const_iterator;

  explicit GeneralIndexIterator(Container ranges) noexcept
      : ranges_(std::move(ranges)) {
    current_.reserve(ranges_.size());
    for (const auto& range : ranges_) {
      if (range.first == range.second) {
        done_ = true;
        return;
      }
      current_.push_back(range.first);
    }
  }

  /// Access the given coordinate of the current value.
  const value_type& operator[](size_t i) const noexcept { return current_[i]; }

  /// Check whether the iterator is complete.  Calling any of the
  /// other methods on a completed iterator is undefined behavior.
  explicit operator bool() const noexcept { return not done_; }

  void operator++() noexcept {
    for (size_t index = 0; index < current_.size(); ++index) {
      ++current_[index];
      if (current_[index] != gsl::at(ranges_, index).second) { return; }
      current_[index] = gsl::at(ranges_, index).first;
    }
    done_ = true;
  }

  /// Iterate over the coordinates of the current value.
  //@{
  iterator_type cbegin() const noexcept { return current_.cbegin(); }
  iterator_type cend() const noexcept { return current_.cend(); }
  iterator_type begin() const noexcept { return cbegin(); }
  iterator_type end() const noexcept { return cend(); }
  //@}

 private:
  Container ranges_;
  std::vector<value_type> current_{};
  bool done_{false};
};

/// Construct a GeneralIndexIterator
template <typename Container>
GeneralIndexIterator<std::decay_t<Container>> make_general_index_iterator(
    Container&& ranges) noexcept {
  return GeneralIndexIterator<std::decay_t<Container>>(
      std::forward<Container>(ranges));
}
