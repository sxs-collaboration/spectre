// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "Options/Options.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/PrettyType.hpp"

namespace Options {
/// The label representing the absence of a value for `Options::Auto`
namespace AutoLabel {
/// 'Auto' label
struct Auto {};
/// 'None' label
struct None {};
/// 'All' label
struct All {};
}  // namespace AutoLabel

/// \ingroup OptionParsingGroup
/// \brief A class indicating that a parsed value can be automatically
/// computed instead of specified.
///
/// When an `Auto<T>` is parsed from an input file, the value may be specified
/// either as the `AutoLabel` (defaults to "Auto") or as a value of type `T`.
/// When this class is passed to the constructor of the class taking it as an
/// option, it can be implicitly converted to a `std::optional<U>`, for any
/// type `U` implicitly creatable from a `T`.
///
/// \snippet Test_Auto.cpp example_class
/// \snippet Test_Auto.cpp example_create
template <typename T, typename Label = AutoLabel::Auto>
class Auto {
 public:
  Auto() = default;
  explicit Auto(T value) : value_(std::move(value)) {}

  // NOLINTNEXTLINE(google-explicit-constructor)
  template <typename U>
  operator std::optional<U>() && { return std::move(value_); }

  // NOLINTNEXTLINE(google-explicit-constructor)
  operator const std::optional<T>&() const { return value_; }

 private:
  std::optional<T> value_{};
};

template <typename T, typename Label>
bool operator==(const Auto<T, Label>& a, const Auto<T, Label>& b) {
  return static_cast<const std::optional<T>&>(a) ==
         static_cast<const std::optional<T>&>(b);
}

template <typename T, typename Label>
bool operator!=(const Auto<T, Label>& a, const Auto<T, Label>& b) {
  return not(a == b);
}

template <typename T, typename Label>
std::ostream& operator<<(std::ostream& os, const Auto<T, Label>& x) {
  const std::optional<T>& value = x;
  if (value) {
    return os << get_output(*value);
  } else {
    return os << pretty_type::name<Label>();
  }
}

template <typename T, typename Label>
struct create_from_yaml<Auto<T, Label>> {
  template <typename Metavariables>
  static Auto<T, Label> create(const Option& options) {
    try {
      if (options.parse_as<std::string>() == pretty_type::name<Label>()) {
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
    return Auto<T, Label>{options.parse_as<T, Metavariables>()};
  }
};
}  // namespace Options
