// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Utilities to retrieve values from maps in tagged containers

#pragma once

#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TupleSlice.hpp"

namespace elliptic::util {
namespace detail {

template <bool PassThrough, typename... MapKeys>
struct unmap_arg;

template <typename... MapKeys>
struct unmap_arg<true, MapKeys...> {
  template <typename T>
  static constexpr const T& apply(
      const T& arg, const std::tuple<MapKeys...>& /*map_keys*/) noexcept {
    return arg;
  }

  template <typename T>
  static constexpr gsl::not_null<T*> apply(
      const gsl::not_null<T*> arg,
      const std::tuple<MapKeys...>& /*map_keys*/) noexcept {
    return arg;
  }
};

template <typename FirstMapKey, typename... MapKeys>
struct unmap_arg<false, FirstMapKey, MapKeys...> {
  template <typename T>
  static constexpr decltype(auto) apply(
      const T& arg,
      const std::tuple<FirstMapKey, MapKeys...>& map_keys) noexcept {
    return unmap_arg<sizeof...(MapKeys) == 0, MapKeys...>::apply(
        arg.at(std::get<0>(map_keys)),
        tuple_tail<sizeof...(MapKeys)>(map_keys));
  }

  template <typename T>
  static constexpr decltype(auto) apply(
      const gsl::not_null<T*> arg,
      const std::tuple<FirstMapKey, MapKeys...>& map_keys) noexcept {
    return unmap_arg<sizeof...(MapKeys) == 0, MapKeys...>::apply(
        make_not_null(&arg->at(std::get<0>(map_keys))),
        tuple_tail<sizeof...(MapKeys)>(map_keys));
  }
};

template <typename... ArgumentTags, typename PassthroughArgumentTags,
          typename F, typename TaggedContainer, typename... MapKeys,
          typename... Args>
SPECTRE_ALWAYS_INLINE decltype(auto) apply_at(
    F&& f, const TaggedContainer& box, const std::tuple<MapKeys...>& map_keys,
    tmpl::list<ArgumentTags...> /*meta*/, PassthroughArgumentTags /*meta*/,
    Args&&... args) noexcept {
  using ::db::apply;
  using ::tuples::apply;
  return apply<tmpl::list<ArgumentTags...>>(
      [&f, &map_keys, &args...](const auto&... args_items) noexcept {
        (void)map_keys;
        return std::forward<F>(f)(
            unmap_arg<
                tmpl::list_contains_v<PassthroughArgumentTags, ArgumentTags>,
                MapKeys...>::apply(args_items, map_keys)...,
            std::forward<Args>(args)...);
      },
      box);
}

}  // namespace detail

// @{
/*!
 * \brief Apply the invokable `f` with arguments from maps in the DataBox
 *
 * Retrieves the `ArgumentTags` from the DataBox, evaluates them at the
 * `map_key(s)` (by calling their `at` member function for every map key in
 * turn) and calls the invokable `f` with the unmapped arguments. The tags in
 * `PassthroughArgumentTags` are passed directly to `f` without unmapping them.
 *
 * For example, a DataBox may have these tags of which two are maps:
 *
 * \snippet Test_ApplyAt.cpp apply_at_tags
 *
 * You can use `apply_at` to evaluate a function at a particular key for these
 * maps:
 *
 * \snippet Test_ApplyAt.cpp apply_at_example
 *
 * \see db::apply
 */
template <typename ArgumentTags, typename PassthroughArgumentTags,
          typename... MapKeys, typename F, typename TaggedContainer,
          typename... Args>
SPECTRE_ALWAYS_INLINE decltype(auto) apply_at(
    F&& f, const TaggedContainer& box, const std::tuple<MapKeys...>& map_keys,
    Args&&... args) noexcept {
  return detail::apply_at(std::forward<F>(f), box, map_keys, ArgumentTags{},
                          PassthroughArgumentTags{},
                          std::forward<Args>(args)...);
}

template <typename ArgumentTags, typename PassthroughArgumentTags,
          typename MapKey, typename F, typename TaggedContainer,
          typename... Args>
SPECTRE_ALWAYS_INLINE decltype(auto) apply_at(F&& f, const TaggedContainer& box,
                                              const MapKey& map_key,
                                              Args&&... args) noexcept {
  return detail::apply_at(
      std::forward<F>(f), box, std::forward_as_tuple(map_key), ArgumentTags{},
      PassthroughArgumentTags{}, std::forward<Args>(args)...);
}
// @}

namespace detail {

template <typename... ReturnTags, typename... ArgumentTags,
          typename PassthroughTags, typename F, typename TaggedContainer,
          typename... MapKeys, typename... Args>
SPECTRE_ALWAYS_INLINE void mutate_apply_at(
    F&& f, const gsl::not_null<TaggedContainer*> box,
    const std::tuple<MapKeys...>& map_keys, tmpl::list<ReturnTags...> /*meta*/,
    tmpl::list<ArgumentTags...> /*meta*/, PassthroughTags /*meta*/,
    Args&&... args) noexcept {
  using ::db::apply;
  using ::db::mutate_apply;
  using ::tuples::apply;
  const auto args_items = apply<tmpl::list<ArgumentTags...>>(
      [](const auto&... expanded_args_items) noexcept {
        return std::forward_as_tuple(expanded_args_items...);
      },
      *box);
  mutate_apply<tmpl::list<ReturnTags...>, tmpl::list<>>(
      [&f, &map_keys, &args_items,
       &args...](const auto... mutated_items) noexcept {
        (void)map_keys;
        (void)args_items;
        std::forward<F>(f)(
            unmap_arg<tmpl::list_contains_v<PassthroughTags, ReturnTags>,
                      MapKeys...>::apply(mutated_items, map_keys)...,
            unmap_arg<tmpl::list_contains_v<PassthroughTags, ArgumentTags>,
                      MapKeys...>::
                apply(std::get<tmpl::index_of<tmpl::list<ArgumentTags...>,
                                              ArgumentTags>::value>(args_items),
                      map_keys)...,
            std::forward<Args>(args)...);
      },
      box);
}

}  // namespace detail

// @{
/*!
 * \brief Apply the invokable `f` to mutate items in maps in the DataBox
 *
 * Retrieves the `MutateTags` and `ArgumentTags` from the DataBox, evaluates
 * them at the `map_key(s)` (by calling their `at` member function for every map
 * key in turn) and calls the invokable `f` with the unmapped arguments. The
 * tags in `PassthroughTags` are passed directly to `f` without unmapping them.
 *
 * For example, a DataBox may have these tags of which two are maps:
 *
 * \snippet Test_ApplyAt.cpp apply_at_tags
 *
 * You can use `mutate_apply_at` to mutate items at a particular key for these
 * maps:
 *
 * \snippet Test_ApplyAt.cpp mutate_apply_at_example
 *
 * \see db::mutate_apply
 */
template <typename MutateTags, typename ArgumentTags, typename PassthroughTags,
          typename F, typename TaggedContainer, typename... MapKeys,
          typename... Args>
SPECTRE_ALWAYS_INLINE void mutate_apply_at(
    F&& f, const gsl::not_null<TaggedContainer*> box,
    const std::tuple<MapKeys...>& map_keys, Args&&... args) noexcept {
  detail::mutate_apply_at(std::forward<F>(f), box, map_keys, MutateTags{},
                          ArgumentTags{}, PassthroughTags{},
                          std::forward<Args>(args)...);
}

template <typename MutateTags, typename ArgumentTags, typename PassthroughTags,
          typename F, typename TaggedContainer, typename MapKey,
          typename... Args>
SPECTRE_ALWAYS_INLINE void mutate_apply_at(
    F&& f, const gsl::not_null<TaggedContainer*> box, const MapKey& map_key,
    Args&&... args) noexcept {
  detail::mutate_apply_at(
      std::forward<F>(f), box, std::forward_as_tuple(map_key), MutateTags{},
      ArgumentTags{}, PassthroughTags{}, std::forward<Args>(args)...);
}
// @}

}  // namespace elliptic::util
