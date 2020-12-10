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
/// The label representing the absence of a value for `Options::Auto`
enum class AutoLabel { Auto, None };

std::ostream& operator<<(std::ostream& os, AutoLabel label) noexcept;

/// \ingroup OptionParsingGroup
/// \brief A class indicating that a parsed value can be automatically
/// computed instead of specified.
///
/// When an `Auto<T>` is parsed from an input file, the value may be specified
/// either as the `AutoLabel` (defaults to "Auto") or as a value of type `T`.
/// When this class is passed to the constructor of the class taking it as an
/// option, it can be implicitly converted to a `std::optional<T>`.
///
/// \snippet Test_Auto.cpp example_class
/// \snippet Test_Auto.cpp example_create
template <typename T, AutoLabel Label = AutoLabel::Auto>
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

template <typename T, AutoLabel Label>
bool operator==(const Auto<T, Label>& a, const Auto<T, Label>& b) noexcept {
  return static_cast<const std::optional<T>&>(a) ==
         static_cast<const std::optional<T>&>(b);
}

template <typename T, AutoLabel Label>
bool operator!=(const Auto<T, Label>& a, const Auto<T, Label>& b) noexcept {
  return not(a == b);
}

template <typename T, AutoLabel Label>
std::ostream& operator<<(std::ostream& os, const Auto<T, Label>& x) noexcept {
  const std::optional<T>& value = x;
  if (value) {
    return os << get_output(*value);
  } else {
    return os << Label;
  }
}

template <typename T, AutoLabel Label>
struct create_from_yaml<Auto<T, Label>> {
  template <typename Metavariables>
  static Auto<T, Label> create(const Option& options) {
    try {
      if (options.parse_as<std::string>() == get_output(Label)) {
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
      // The node failed to parse as a string.  It is not the AutoLabel.
    }
    return Auto<T, Label>{options.parse_as<T>()};
  }
};
}  // namespace Options
