// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <utility>
#include <variant>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
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
  using value_type = std::optional<T>;

  Auto() = default;
  explicit Auto(T value) : value_(std::move(value)) {}

  // These lines are just to work around a spurious warning.
  Auto(const Auto&) = default;
  Auto& operator=(const Auto&) = default;
  Auto(Auto&&) = default;
  Auto& operator=(Auto&&) = default;
#if defined(__GNUC__) and not defined(__clang__) and __GNUC__ >= 12 and \
    __GNUC__ < 14
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
  ~Auto() = default;
#if defined(__GNUC__) and not defined(__clang__) and __GNUC__ >= 12 and \
    __GNUC__ < 14
#pragma GCC diagnostic pop
#endif

  // NOLINTNEXTLINE(google-explicit-constructor)
  template <typename U>
  operator std::optional<U>() && {
    return std::move(value_);
  }

  void pup(PUP::er& p) { p | value_; }

  // NOLINTNEXTLINE(google-explicit-constructor)
  operator const value_type&() const { return value_; }

 private:
  value_type value_{};
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

namespace Auto_detail {
template <typename Label>
struct AutoLabel {};
}  // namespace Auto_detail

template <typename Label>
struct create_from_yaml<Auto_detail::AutoLabel<Label>> {
  template <typename Metavariables>
  static Auto_detail::AutoLabel<Label> create(const Option& options) {
    const auto label_string = pretty_type::name<Label>();
    try {
      if (options.parse_as<std::string>() == label_string) {
        return {};
      }
    } catch (...) {
      // The node failed to parse as a string.  It is not the Label.
    }
    // The error if the std::variant parse fails will print the value
    // from the input file (and the T parse probably will too), so we
    // don't need to print it again.
    PARSE_ERROR(options.context(),
                "Failed to parse as Auto label \"" << label_string << "\"");
  }
};

template <typename T, typename Label>
struct create_from_yaml<Auto<T, Label>> {
  template <typename Metavariables>
  static Auto<T, Label> create(const Option& options) {
    auto parsed_variant =
        options.parse_as<std::variant<Auto_detail::AutoLabel<Label>, T>,
                         Metavariables>();
    if (std::holds_alternative<T>(parsed_variant)) {
      return Auto<T, Label>{std::move(std::get<T>(parsed_variant))};
    } else {
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8 && __GNUC__ < 10
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif  // defined(__GNUC__) && !defined(__clang__) && __GNUC__ => 8 && __GNUC__
        // < 10
      return Auto<T, Label>{};
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 8 && __GNUC__ < 10
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__) && __GNUC__ => 8 && __GNUC__
        // < 10
    }
  }
};
}  // namespace Options
