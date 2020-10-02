// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "Options/Options.hpp"
#include "Utilities/GetOutput.hpp"

namespace Options {
/// \ingroup OptionParsingGroup
/// \brief A class indicating that a parsed value can be automatically
/// computed instead of specified.
///
/// When an `Auto<T>` is parsed from an input file, the value may be
/// specified either as "Auto" or as a value of type `T`.  When this
/// class is passed to the constructor of the class taking it as an
/// option, it can be implicitly converted to a `std::optional<T>`.
///
/// \snippet Test_Auto.cpp example_class
/// \snippet Test_Auto.cpp example_create
template <typename T>
class Auto {
 public:
  Auto() = default;
  explicit Auto(T value) noexcept : value_(std::move(value)) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::optional<T>() && { return std::move(value_); }

  // NOLINTNEXTLINE(google-explicit-constructor)
  operator const std::optional<T>&() const { return value_; }

 private:
  std::optional<T> value_{};
};

template <typename T>
bool operator==(const Auto<T>& a, const Auto<T>& b) noexcept {
  return static_cast<const std::optional<T>&>(a) ==
         static_cast<const std::optional<T>&>(b);
}

template <typename T>
bool operator!=(const Auto<T>& a, const Auto<T>& b) noexcept {
  return not(a == b);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const Auto<T>& x) noexcept {
  const std::optional<T>& value = x;
  if (value) {
    return os << get_output(*value);
  } else {
    return os << "Auto";
  }
}

template <typename T>
struct create_from_yaml<Auto<T>> {
  template <typename Metavariables>
  static Auto<T> create(const Option& options) {
    try {
      if (options.parse_as<std::string>() == "Auto") {
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8 && __GNUC__ < 10
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif  // defined(__GNUC__) && !defined(__clang__) && __GNUC__ => 8 && __GNUC__ < 10
        return {};
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8 && __GNUC__ < 10
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__) && __GNUC__ => 8 && __GNUC__ < 10
      }
    } catch (...) {
      // The node failed to parse as a string.  It is not "Auto".
    }
    return Auto<T>{options.parse_as<T>()};
  }
};
}  // namespace Options
