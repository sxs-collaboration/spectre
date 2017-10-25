// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helpers for the Options<T> class

#pragma once

#include <array>
#include <exception>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// The string used in option structs
using OptionString_t = const char* const;

/// Holds details of the implementation of Options
namespace Options_detail {
template <typename S, typename = cpp17::void_t<>>
struct has_default : std::false_type {};
template <typename S>
struct has_default<S,
                   cpp17::void_t<decltype(std::declval<S>().default_value())>>
    : std::true_type {};

template <typename S, typename = cpp17::void_t<>>
struct has_lower_bound : std::false_type {};
template <typename S>
struct has_lower_bound<S,
                       cpp17::void_t<decltype(std::declval<S>().lower_bound())>>
    : std::true_type {};

template <typename S, typename = cpp17::void_t<>>
struct has_upper_bound : std::false_type {};
template <typename S>
struct has_upper_bound<S,
                       cpp17::void_t<decltype(std::declval<S>().upper_bound())>>
    : std::true_type {};

template <typename S, typename = cpp17::void_t<>>
struct has_lower_bound_on_size : std::false_type {};
template <typename S>
struct has_lower_bound_on_size<
    S, cpp17::void_t<decltype(std::declval<S>().lower_bound_on_size())>>
    : std::true_type {};

template <typename S, typename = cpp17::void_t<>>
struct has_upper_bound_on_size : std::false_type {};
template <typename S>
struct has_upper_bound_on_size<
    S, cpp17::void_t<decltype(std::declval<S>().upper_bound_on_size())>>
    : std::true_type {};

struct print {
  explicit print(const int max_label_size) noexcept
      : max_label_size_(max_label_size) {}

  using value_type = std::string;
  template <typename T, Requires<has_default<T>::value> = nullptr>
  static std::string print_default() noexcept {
    std::ostringstream ss;
    ss << "default=" << std::boolalpha << T::default_value();
    return ss.str();
  }
  template <typename T, Requires<not has_default<T>::value> = nullptr>
  static std::string print_default() noexcept {
    return "";
  }
  template <typename T, Requires<has_lower_bound<T>::value> = nullptr>
  static std::string print_lower_bound() noexcept {
    std::ostringstream ss;
    ss << "min=" << T::lower_bound();
    return ss.str();
  }
  template <typename T, Requires<not has_lower_bound<T>::value> = nullptr>
  static std::string print_lower_bound() noexcept {
    return "";
  }
  template <typename T, Requires<has_upper_bound<T>::value> = nullptr>
  static std::string print_upper_bound() noexcept {
    std::ostringstream ss;
    ss << "max=" << T::upper_bound();
    return ss.str();
  }
  template <typename T, Requires<not has_upper_bound<T>::value> = nullptr>
  static std::string print_upper_bound() noexcept {
    return "";
  }
  template <typename T, Requires<has_lower_bound_on_size<T>::value> = nullptr>
  static std::string print_lower_bound_on_size() noexcept {
    std::ostringstream ss;
    ss << "min size=" << T::lower_bound_on_size();
    return ss.str();
  }
  template <typename T,
            Requires<not has_lower_bound_on_size<T>::value> = nullptr>
  static std::string print_lower_bound_on_size() noexcept {
    return "";
  }
  template <typename T, Requires<has_upper_bound_on_size<T>::value> = nullptr>
  static std::string print_upper_bound_on_size() noexcept {
    std::ostringstream ss;
    ss << "max size=" << T::upper_bound_on_size();
    return ss.str();
  }
  template <typename T,
            Requires<not has_upper_bound_on_size<T>::value> = nullptr>
  static std::string print_upper_bound_on_size() noexcept {
    return "";
  }

  template <typename T>
  void operator()(tmpl::type_<T> /*meta*/) noexcept {
    std::ostringstream ss;
    ss << "  " << std::setw(max_label_size_) << std::left
       << pretty_type::short_name<T>()
       << pretty_type::get_name<typename T::type>();
    std::string limits;
    for (const auto& limit : {
        print_default<T>(),
        print_lower_bound<T>(),
        print_upper_bound<T>(),
        print_lower_bound_on_size<T>(),
        print_upper_bound_on_size<T>() }) {
      if (not limits.empty() and not limit.empty()) {
        limits += ", ";
      }
      limits += limit;
    }
    if (not limits.empty()) {
      ss << " [" << limits << "]";
    }
    ss << "\n" << std::setw(max_label_size_ + 2) << "" << T::help << "\n\n";
    value += ss.str();
  }

  value_type value{};

 private:
  const int max_label_size_;
};

// TMP function to create an unordered_set of option names.
struct create_valid_names {
  using value_type = std::unordered_set<std::string>;
  value_type value{};
  template <typename T>
  void operator()(tmpl::type_<T> /*meta*/) noexcept {
    const std::string label = pretty_type::short_name<T>();
    ASSERT(0 == value.count(label), "Duplicate option name: " << label);
    value.insert(label);
  }
};

class propagate_context : public std::exception {
 public:
  explicit propagate_context(std::string message) noexcept
      : message_(std::move(message)) {}

  const char* what() const noexcept override { return message_.c_str(); }
  const std::string& message() const noexcept { return message_; }

 private:
  std::string message_;
};
}  // namespace Options_detail
