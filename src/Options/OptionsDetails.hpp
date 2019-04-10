// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helpers for the Options<T> class

#pragma once

#include <algorithm>
#include <array>
#include <iomanip>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// Holds details of the implementation of Options
namespace Options_detail {
template <typename T, typename = cpp17::void_t<>>
struct name_helper {
  static std::string name() noexcept { return pretty_type::short_name<T>(); }
};

template <typename T>
struct name_helper<T, cpp17::void_t<decltype(T::name())>> {
  static std::string name() noexcept { return T::name(); }
};

// The name in the YAML file for a struct.
template <typename T>
std::string name() noexcept {
  return name_helper<T>::name();
}

// Traverses the group hierarchy of `Tag`, returning the topmost group that is
// a subgroup of `Root`. Directly returns `Tag` if it has no group.
// This means that the returned type is always the direct child node of `Root`
// that contains `Tag` in its hierarchy.
// If `Root` is not in the group hierarchy of `Tag`, this function returns the
// topmost group of `Tag` (meaning that `Root` is treated as the root of its
// group hierarchy).
template <typename Tag, typename Root, typename = cpp17::void_t<>>
struct find_subgroup {
  using type = Tag;
};

template <typename Tag>
struct find_subgroup<Tag, typename Tag::group,
                     cpp17::void_t<typename Tag::group>> {
  using type = Tag;
};

template <typename Tag, typename Root>
struct find_subgroup<Tag, Root, cpp17::void_t<typename Tag::group>> {
  using type = typename find_subgroup<typename Tag::group, Root>::type;
};

/// Checks if `Tag` is within the group hierarchy of `Group`.
template <typename Tag, typename Group, typename = cpp17::void_t<>>
struct is_in_group : std::false_type {};

template <typename Tag>
struct is_in_group<Tag, typename Tag::group, cpp17::void_t<typename Tag::group>>
    : std::true_type {};

template <typename Tag, typename Group>
struct is_in_group<Tag, Group, cpp17::void_t<typename Tag::group>>
    : is_in_group<typename Tag::group, Group> {};

/// The subset of tags in `OptionList` that are in the hierarchy of `Group`
template <typename OptionList, typename Group>
using options_in_group = tmpl::filter<OptionList, is_in_group<tmpl::_1, Group>>;

// Display a type in a pseudo-YAML form, leaving out likely irrelevant
// information.
template <typename T>
struct yaml_type {
  static std::string value() noexcept { return pretty_type::get_name<T>(); }
};

template <typename T>
struct yaml_type<std::unique_ptr<T>> {
  static std::string value() noexcept { return yaml_type<T>::value(); }
};

template <typename T>
struct yaml_type<std::vector<T>> {
  static std::string value() noexcept {
    return "[" + yaml_type<T>::value() + ", ...]";
  }
};

template <typename T>
struct yaml_type<std::list<T>> {
  static std::string value() noexcept {
    return "[" + yaml_type<T>::value() + ", ...]";
  }
};

template <typename T, size_t N>
struct yaml_type<std::array<T, N>> {
  static std::string value() noexcept {
    return "[" + yaml_type<T>::value() + " x" + std::to_string(N) + "]";
  }
};

template <typename K, typename V, typename C>
struct yaml_type<std::map<K, V, C>> {
  static std::string value() noexcept {
    return "{" + yaml_type<K>::value() + ": " + yaml_type<V>::value() + "}";
  }
};

template <typename K, typename V, typename H, typename E>
struct yaml_type<std::unordered_map<K, V, H, E>> {
  static std::string value() noexcept {
    return "{" + yaml_type<K>::value() + ": " + yaml_type<V>::value() + "}";
  }
};

template <typename T, typename U>
struct yaml_type<std::pair<T, U>> {
  static std::string value() noexcept {
    return "[" + yaml_type<T>::value() + ", " + yaml_type<U>::value() + "]";
  }
};

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

template <typename Group, typename OptionList, typename = std::nullptr_t>
struct print_impl {
  static std::string apply(const int max_label_size) noexcept {
    std::ostringstream ss;
    ss << "  " << std::setw(max_label_size + 2) << std::left << name<Group>()
       << Group::help << "\n\n";
    return ss.str();
  }
};

template <typename Tag, typename OptionList>
struct print_impl<Tag, OptionList,
                  Requires<tmpl::list_contains_v<OptionList, Tag>>> {
  template <typename LocalTag, Requires<has_default<LocalTag>::value> = nullptr>
  static std::string print_default() noexcept {
    std::ostringstream ss;
    ss << "default=" << std::boolalpha << LocalTag::default_value();
    return ss.str();
  }
  template <typename LocalTag,
            Requires<not has_default<LocalTag>::value> = nullptr>
  static std::string print_default() noexcept {
    return "";
  }
  template <typename LocalTag,
            Requires<has_lower_bound<LocalTag>::value> = nullptr>
  static std::string print_lower_bound() noexcept {
    std::ostringstream ss;
    ss << "min=" << LocalTag::lower_bound();
    return ss.str();
  }
  template <typename LocalTag,
            Requires<not has_lower_bound<LocalTag>::value> = nullptr>
  static std::string print_lower_bound() noexcept {
    return "";
  }
  template <typename LocalTag,
            Requires<has_upper_bound<LocalTag>::value> = nullptr>
  static std::string print_upper_bound() noexcept {
    std::ostringstream ss;
    ss << "max=" << LocalTag::upper_bound();
    return ss.str();
  }
  template <typename LocalTag,
            Requires<not has_upper_bound<LocalTag>::value> = nullptr>
  static std::string print_upper_bound() noexcept {
    return "";
  }
  template <typename LocalTag,
            Requires<has_lower_bound_on_size<LocalTag>::value> = nullptr>
  static std::string print_lower_bound_on_size() noexcept {
    std::ostringstream ss;
    ss << "min size=" << LocalTag::lower_bound_on_size();
    return ss.str();
  }
  template <typename LocalTag,
            Requires<not has_lower_bound_on_size<LocalTag>::value> = nullptr>
  static std::string print_lower_bound_on_size() noexcept {
    return "";
  }
  template <typename LocalTag,
            Requires<has_upper_bound_on_size<LocalTag>::value> = nullptr>
  static std::string print_upper_bound_on_size() noexcept {
    std::ostringstream ss;
    ss << "max size=" << LocalTag::upper_bound_on_size();
    return ss.str();
  }
  template <typename LocalTag,
            Requires<not has_upper_bound_on_size<LocalTag>::value> = nullptr>
  static std::string print_upper_bound_on_size() noexcept {
    return "";
  }

  static std::string apply(const int max_label_size) noexcept {
    std::ostringstream ss;
    ss << "  " << std::setw(max_label_size + 2) << std::left << name<Tag>()
       << yaml_type<typename Tag::type>::value();
    std::string limits;
    for (const auto& limit :
         {print_default<Tag>(), print_lower_bound<Tag>(),
          print_upper_bound<Tag>(), print_lower_bound_on_size<Tag>(),
          print_upper_bound_on_size<Tag>()}) {
      if (not limits.empty() and not limit.empty()) {
        limits += ", ";
      }
      limits += limit;
    }
    if (not limits.empty()) {
      ss << " [" << limits << "]";
    }
    ss << "\n" << std::setw(max_label_size + 4) << "" << Tag::help << "\n\n";
    return ss.str();
  }
};

template <typename OptionList>
struct print {
  explicit print(const int max_label_size) noexcept
      : max_label_size_(max_label_size) {}
  using value_type = std::string;
  template <typename Tag>
  void operator()(tmpl::type_<Tag> /*meta*/) noexcept {
    value += print_impl<Tag, OptionList>::apply(max_label_size_);
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
    const std::string label = name<T>();
    ASSERT(0 == value.count(label), "Duplicate option name: " << label);
    value.insert(label);
  }
};

template <typename T, typename Metavariables>
struct CreateWrapper {
  using metavariables = Metavariables;
  T data{};
};
#define CREATE_WRAPPER_FORWARD_OP(op)                          \
  template <typename T, typename Metavariables>                \
  bool operator op(const CreateWrapper<T, Metavariables>& a,   \
                   const CreateWrapper<T, Metavariables>& b) { \
    return a.data op b.data;                                   \
  }
CREATE_WRAPPER_FORWARD_OP(==)
CREATE_WRAPPER_FORWARD_OP(!=)
CREATE_WRAPPER_FORWARD_OP(<)
CREATE_WRAPPER_FORWARD_OP(>)
CREATE_WRAPPER_FORWARD_OP(<=)
CREATE_WRAPPER_FORWARD_OP(>=)
#undef CREATE_WRAPPER_FORWARD_OP

template <typename T, typename = std::nullptr_t>
struct wrap_create_types_impl;

template <typename T, typename Metavariables>
using wrap_create_types =
    typename wrap_create_types_impl<T>::template wrapped_type<Metavariables>;

template <typename T>
auto unwrap_create_types(T wrapped) {
  return wrap_create_types_impl<T>::unwrap(std::move(wrapped));
}

template <typename T, typename>
struct wrap_create_types_impl {
  template <typename Metavariables>
  using wrapped_type = CreateWrapper<T, Metavariables>;
};

template <typename T, typename Metavariables>
struct wrap_create_types_impl<CreateWrapper<T, Metavariables>> {
  // Never actually used, but instantiated during unwrapping
  template <typename /*Metavars*/>
  using wrapped_type = void;

  static T unwrap(CreateWrapper<T, Metavariables> wrapped) {
    return std::move(wrapped.data);
  }
};

template <typename T>
struct wrap_create_types_impl<T, Requires<std::is_fundamental<T>::value>> {
  template <typename Metavariables>
  using wrapped_type = T;

  static T unwrap(T wrapped) { return wrapped; }
};

// Classes convertible by yaml-cpp
template <>
struct wrap_create_types_impl<std::string> {
  template <typename Metavariables>
  using wrapped_type = std::string;

  static std::string unwrap(std::string wrapped) { return wrapped; }
};

template <typename K, typename V>
struct wrap_create_types_impl<std::map<K, V>> {
  template <typename Metavariables>
  using wrapped_type = std::map<wrap_create_types<K, Metavariables>,
                                wrap_create_types<V, Metavariables>>;

  static auto unwrap(std::map<K, V> wrapped) {
    using UnwrappedK = decltype(unwrap_create_types<K>(std::declval<K>()));
    using UnwrappedV = decltype(unwrap_create_types<V>(std::declval<V>()));
    std::map<UnwrappedK, UnwrappedV> result;
    for (auto& w : wrapped) {
      result.emplace(unwrap_create_types<K>(std::move(w.first)),
                     unwrap_create_types<V>(std::move(w.second)));
    }
    return result;
  }
};

template <typename T>
struct wrap_create_types_impl<std::vector<T>> {
  template <typename Metavariables>
  using wrapped_type = std::vector<wrap_create_types<T, Metavariables>>;

  static auto unwrap(std::vector<T> wrapped) {
    using UnwrappedT = decltype(unwrap_create_types<T>(std::declval<T>()));
    std::vector<UnwrappedT> result;
    result.reserve(wrapped.size());
    for (auto& w : wrapped) {
      result.push_back(unwrap_create_types<T>(std::move(w)));
    }
    return result;
  }
};

template <typename T>
struct wrap_create_types_impl<std::list<T>> {
  template <typename Metavariables>
  using wrapped_type = std::list<wrap_create_types<T, Metavariables>>;

  static auto unwrap(std::list<T> wrapped) {
    using UnwrappedT = decltype(unwrap_create_types<T>(std::declval<T>()));
    std::list<UnwrappedT> result;
    for (auto& w : wrapped) {
      result.push_back(unwrap_create_types<T>(std::move(w)));
    }
    return result;
  }
};

template <typename T, size_t N>
struct wrap_create_types_impl<std::array<T, N>> {
  template <typename Metavariables>
  using wrapped_type = std::array<wrap_create_types<T, Metavariables>, N>;

  static auto unwrap(std::array<T, N> wrapped) {
    return unwrap_helper(std::move(wrapped), std::make_index_sequence<N>{});
  }

  template <size_t... Is>
  static auto unwrap_helper(std::array<T, N> wrapped,
                            std::integer_sequence<size_t, Is...> /*meta*/) {
    using UnwrappedT = decltype(unwrap_create_types<T>(std::declval<T>()));
    static_cast<void>(wrapped);  // Work around broken GCC warning
    return std::array<UnwrappedT, N>{
        {unwrap_create_types<T>(std::move(wrapped[Is]))...}};
  }
};

template <typename T, typename U>
struct wrap_create_types_impl<std::pair<T, U>> {
  template <typename Metavariables>
  using wrapped_type = std::pair<wrap_create_types<T, Metavariables>,
                                 wrap_create_types<U, Metavariables>>;

  static auto unwrap(std::pair<T, U> wrapped) {
    using UnwrappedT = decltype(unwrap_create_types<T>(std::declval<T>()));
    using UnwrappedU = decltype(unwrap_create_types<U>(std::declval<U>()));
    return std::pair<UnwrappedT, UnwrappedU>(
        unwrap_create_types<T>(std::move(wrapped.first)),
        unwrap_create_types<U>(std::move(wrapped.second)));
  }
};
}  // namespace Options_detail
