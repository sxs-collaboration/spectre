// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines helpers for the Parser<T> class

#pragma once

#include <algorithm>
#include <array>
#include <iomanip>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Options/Options.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsA.hpp"
#include "Utilities/WrapText.hpp"

namespace Options::Options_detail {

// Traverses the group hierarchy of `Tag`, returning the topmost group that is
// a subgroup of `Root`. Directly returns `Tag` if it has no group.
// This means that the returned type is always the direct child node of `Root`
// that contains `Tag` in its hierarchy.
// If `Root` is not in the group hierarchy of `Tag`, this function returns the
// topmost group of `Tag` (meaning that `Root` is treated as the root of its
// group hierarchy).
template <typename Tag, typename Root, typename = std::void_t<>>
struct find_subgroup {
  using type = Tag;
};

template <typename Tag>
struct find_subgroup<Tag, typename Tag::group,
                     std::void_t<typename Tag::group>> {
  using type = Tag;
};

template <typename Tag, typename Root>
struct find_subgroup<Tag, Root, std::void_t<typename Tag::group>> {
  using type = typename find_subgroup<typename Tag::group, Root>::type;
};

/// Checks if `Tag` is within the group hierarchy of `Group`.
template <typename Tag, typename Group, typename = std::void_t<>>
struct is_in_group : std::false_type {};

template <typename Tag>
struct is_in_group<Tag, typename Tag::group, std::void_t<typename Tag::group>>
    : std::true_type {};

template <typename Tag, typename Group>
struct is_in_group<Tag, Group, std::void_t<typename Tag::group>>
    : is_in_group<typename Tag::group, Group> {};

/// The subset of tags in `OptionList` that are in the hierarchy of `Group`
template <typename OptionList, typename Group>
using options_in_group = tmpl::filter<OptionList, is_in_group<tmpl::_1, Group>>;

// Display a type in a pseudo-YAML form, leaving out likely irrelevant
// information.
template <typename T>
struct yaml_type {
  static std::string value() noexcept { return pretty_type::short_name<T>(); }
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

template <typename S, typename = std::void_t<>>
struct has_suggested : std::false_type {};
template <typename S>
struct has_suggested<
    S, std::void_t<decltype(std::declval<S>().suggested_value())>>
    : std::true_type {};

template <typename S, typename = std::void_t<>>
struct has_lower_bound : std::false_type {};
template <typename S>
struct has_lower_bound<S,
                       std::void_t<decltype(std::declval<S>().lower_bound())>>
    : std::true_type {};

template <typename S, typename = std::void_t<>>
struct has_upper_bound : std::false_type {};
template <typename S>
struct has_upper_bound<S,
                       std::void_t<decltype(std::declval<S>().upper_bound())>>
    : std::true_type {};

template <typename S, typename = std::void_t<>>
struct has_lower_bound_on_size : std::false_type {};
template <typename S>
struct has_lower_bound_on_size<
    S, std::void_t<decltype(std::declval<S>().lower_bound_on_size())>>
    : std::true_type {};

template <typename S, typename = std::void_t<>>
struct has_upper_bound_on_size : std::false_type {};
template <typename S>
struct has_upper_bound_on_size<
    S, std::void_t<decltype(std::declval<S>().upper_bound_on_size())>>
    : std::true_type {};

template <typename Group, typename OptionList, typename = std::nullptr_t>
struct print_impl {
  static std::string apply() noexcept {
    std::ostringstream ss;
    ss << "  " << name<Group>() << ":\n"
       << wrap_text(Group::help, 77, "    ") << "\n\n";
    return ss.str();
  }
};

template <typename Tag, typename OptionList>
struct print_impl<Tag, OptionList,
                  Requires<tmpl::list_contains_v<OptionList, Tag>>> {
  static std::string apply() noexcept {
    std::ostringstream ss;
    ss << "  " << name<Tag>() << ":\n"
       << "    " << "type=" << yaml_type<typename Tag::type>::value();
    if constexpr (has_suggested<Tag>::value) {
      if constexpr (tt::is_a_v<std::unique_ptr, typename Tag::type>) {
        call_with_dynamic_type<
            void, typename Tag::type::element_type::creatable_classes>(
            Tag::suggested_value().get(), [&ss](const auto* derived) noexcept {
              ss << "\n    suggested=" << std::boolalpha
                 << pretty_type::short_name<decltype(*derived)>();
            });
      } else {
        ss << "\n    suggested="
           << (MakeString{} << std::boolalpha << Tag::suggested_value());
      }
    }
    if constexpr (has_lower_bound<Tag>::value) {
      ss << "\n    min=" << (MakeString{} << Tag::lower_bound());
    }
    if constexpr (has_upper_bound<Tag>::value) {
      ss << "\n    max=" << (MakeString{} << Tag::upper_bound());
    }
    if constexpr (has_lower_bound_on_size<Tag>::value) {
      ss << "\n    min size=" << Tag::lower_bound_on_size();
    }
    if constexpr (has_upper_bound_on_size<Tag>::value) {
      ss << "\n    max size=" << Tag::upper_bound_on_size();
    }
    ss << "\n" << wrap_text(Tag::help, 77, "    ") << "\n\n";
    return ss.str();
  }
};

template <typename OptionList>
struct print {
  print() = default;
  using value_type = std::string;
  template <typename Tag>
  void operator()(tmpl::type_<Tag> /*meta*/) noexcept {
    value += print_impl<Tag, OptionList>::apply();
  }
  value_type value{};
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
    for (auto it = wrapped.begin(); it != wrapped.end();) {
      auto node = wrapped.extract(it++);
      result.emplace(unwrap_create_types<K>(std::move(node.key())),
                     unwrap_create_types<V>(std::move(node.mapped())));
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
}  // namespace Options::Options_detail
