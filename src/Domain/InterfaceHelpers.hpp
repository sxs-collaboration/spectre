// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DirectionMap.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

namespace InterfaceHelpers_detail {

template <typename T, typename = std::void_t<>>
struct get_volume_tags_impl {
  using type = tmpl::list<>;
};
template <typename T>
struct get_volume_tags_impl<T, std::void_t<typename T::volume_tags>> {
  using type = typename T::volume_tags;
};

}  // namespace InterfaceHelpers_detail

/// Retrieve `T::volume_tags`, defaulting to an empty list
template <typename T>
using get_volume_tags =
    tmpl::type_from<InterfaceHelpers_detail::get_volume_tags_impl<T>>;

namespace InterfaceHelpers_detail {

template <typename Tag, typename DirectionsTag, typename VolumeTags>
struct make_interface_tag_impl {
  using type = tmpl::conditional_t<tmpl::list_contains_v<VolumeTags, Tag>, Tag,
                                   domain::Tags::Interface<DirectionsTag, Tag>>;
};

// Retrieve the `argument_tags` from the `InterfaceInvokable` and wrap them in
// `::Tags::Interface` if they are not listed in
// `InterfaceInvokable::volume_tags`.
template <typename InterfaceInvokable, typename DirectionsTag>
using get_interface_argument_tags = tmpl::transform<
    typename InterfaceInvokable::argument_tags,
    make_interface_tag_impl<tmpl::_1, tmpl::pin<DirectionsTag>,
                            tmpl::pin<get_volume_tags<InterfaceInvokable>>>>;

/// Pull the direction's entry from interface arguments, passing volume
/// arguments through unchanged.
template <bool IsVolumeTag>
struct unmap_interface_args;

template <>
struct unmap_interface_args<true> {
  template <typename T>
  using f = T;

  template <size_t VolumeDim, typename T>
  static constexpr const T& apply(const ::Direction<VolumeDim>& /*direction*/,
                                  const T& arg) noexcept {
    return arg;
  }
};

template <>
struct unmap_interface_args<false> {
  template <typename T>
  using f = typename T::mapped_type;

  template <size_t VolumeDim, typename T>
  static constexpr decltype(auto) apply(const ::Direction<VolumeDim>& direction,
                                        const T& arg) noexcept {
    return arg.at(direction);
  }
};

CREATE_IS_CALLABLE(apply)
CREATE_IS_CALLABLE_V(apply)

template <bool HasStaticApply, typename InterfaceReturnType,
          typename DirectionsTag, typename VolumeTagsList,
          typename... ArgumentTags>
struct DispatchInterfaceInvokable;

template <typename InterfaceReturnType, typename DirectionsTag,
          typename VolumeTagsList, typename... ArgumentTags>
struct DispatchInterfaceInvokable<true, InterfaceReturnType, DirectionsTag,
                                  VolumeTagsList, ArgumentTags...> {
  template <typename InterfaceInvokable, typename DbTagsList,
            typename... ExtraArgs>
  static constexpr auto apply(InterfaceInvokable&& /*interface_invokable*/,
                              const db::DataBox<DbTagsList>& box,
                              ExtraArgs&&... extra_args) noexcept {
    DirectionMap<DirectionsTag::volume_dim, InterfaceReturnType> result{};
    for (const auto& direction : get<DirectionsTag>(box)) {
      auto interface_value = InterfaceInvokable::apply(
          unmap_interface_args<
              tmpl::list_contains_v<VolumeTagsList, ArgumentTags>>::
              apply(direction,
                    get<tmpl::type_from<make_interface_tag_impl<
                        ArgumentTags, DirectionsTag, VolumeTagsList>>>(box))...,
          extra_args...);
      result.insert({direction, std::move(interface_value)});
    }
    return result;
  }
};

template <typename DirectionsTag, typename VolumeTagsList,
          typename... ArgumentTags>
struct DispatchInterfaceInvokable<true, void, DirectionsTag, VolumeTagsList,
                                  ArgumentTags...> {
  template <typename InterfaceInvokable, typename DbTagsList,
            typename... ExtraArgs>
  static constexpr void apply(InterfaceInvokable&& /*interface_invokable*/,
                              const db::DataBox<DbTagsList>& box,
                              ExtraArgs&&... extra_args) noexcept {
    for (const auto& direction : get<DirectionsTag>(box)) {
      InterfaceInvokable::apply(
          unmap_interface_args<
              tmpl::list_contains_v<VolumeTagsList, ArgumentTags>>::
              apply(direction,
                    get<tmpl::type_from<make_interface_tag_impl<
                        ArgumentTags, DirectionsTag, VolumeTagsList>>>(box))...,
          extra_args...);
    }
  }
};

template <typename InterfaceReturnType, typename DirectionsTag,
          typename VolumeTagsList, typename... ArgumentTags>
struct DispatchInterfaceInvokable<false, InterfaceReturnType, DirectionsTag,
                                  VolumeTagsList, ArgumentTags...> {
  template <typename InterfaceInvokable, typename DbTagsList,
            typename... ExtraArgs>
  static constexpr auto apply(InterfaceInvokable&& interface_invokable,
                              const db::DataBox<DbTagsList>& box,
                              ExtraArgs&&... extra_args) noexcept {
    DirectionMap<DirectionsTag::volume_dim, InterfaceReturnType> result{};
    for (const auto& direction : get<DirectionsTag>(box)) {
      auto interface_value = interface_invokable(
          unmap_interface_args<
              tmpl::list_contains_v<VolumeTagsList, ArgumentTags>>::
              apply(direction,
                    get<tmpl::type_from<make_interface_tag_impl<
                        ArgumentTags, DirectionsTag, VolumeTagsList>>>(box))...,
          extra_args...);
      result.insert({direction, std::move(interface_value)});
    }
    return result;
  }
};

template <typename DirectionsTag, typename VolumeTagsList,
          typename... ArgumentTags>
struct DispatchInterfaceInvokable<false, void, DirectionsTag, VolumeTagsList,
                                  ArgumentTags...> {
  template <typename InterfaceInvokable, typename DbTagsList,
            typename... ExtraArgs>
  static constexpr void apply(InterfaceInvokable&& interface_invokable,
                              const db::DataBox<DbTagsList>& box,
                              ExtraArgs&&... extra_args) noexcept {
    for (const auto& direction : get<DirectionsTag>(box)) {
      interface_invokable(
          unmap_interface_args<
              tmpl::list_contains_v<VolumeTagsList, ArgumentTags>>::
              apply(direction,
                    get<tmpl::type_from<make_interface_tag_impl<
                        ArgumentTags, DirectionsTag, VolumeTagsList>>>(box))...,
          extra_args...);
    }
  }
};

template <typename DirectionsTag, typename ArgumentTags, typename VolumeTags>
struct InterfaceApplyImpl;

template <typename DirectionsTag, typename VolumeTagsList,
          typename... ArgumentTags>
struct InterfaceApplyImpl<DirectionsTag, tmpl::list<ArgumentTags...>,
                          VolumeTagsList> {
  static constexpr size_t volume_dim = DirectionsTag::volume_dim;

  template <typename InterfaceInvokable, typename DbTagsList,
            typename... ExtraArgs,
            Requires<is_apply_callable_v<
                InterfaceInvokable,
                const db::const_item_type<ArgumentTags, DbTagsList>&...,
                ExtraArgs...>> = nullptr>
  static constexpr auto apply(InterfaceInvokable&& interface_invokable,
                              const db::DataBox<DbTagsList>& box,
                              ExtraArgs&&... extra_args) noexcept {
    using interface_return_type =
        std::decay_t<decltype(InterfaceInvokable::apply(
            std::declval<db::const_item_type<ArgumentTags, DbTagsList>>()...,
            std::declval<ExtraArgs&&>()...))>;
    return DispatchInterfaceInvokable<true, interface_return_type,
                                      DirectionsTag, VolumeTagsList,
                                      ArgumentTags...>::
        apply(std::forward<InterfaceInvokable>(interface_invokable), box,
              std::forward<ExtraArgs>(extra_args)...);
  }

  template <typename InterfaceInvokable, typename DbTagsList,
            typename... ExtraArgs,
            Requires<not is_apply_callable_v<
                InterfaceInvokable,
                const db::const_item_type<ArgumentTags, DbTagsList>&...,
                ExtraArgs...>> = nullptr>
  static constexpr auto apply(InterfaceInvokable&& interface_invokable,
                              const db::DataBox<DbTagsList>& box,
                              ExtraArgs&&... extra_args) noexcept {
    using interface_return_type = std::decay_t<decltype(interface_invokable(
        std::declval<db::const_item_type<ArgumentTags, DbTagsList>>()...,
        std::declval<ExtraArgs&&>()...))>;
    return DispatchInterfaceInvokable<false, interface_return_type,
                                      DirectionsTag, VolumeTagsList,
                                      ArgumentTags...>::
        apply(std::forward<InterfaceInvokable>(interface_invokable), box,
              std::forward<ExtraArgs>(extra_args)...);
  }
};

}  // namespace InterfaceHelpers_detail

// @{
/*!
 * \brief Apply an invokable to the `box` on all interfaces given by the
 * `DirectionsTag`.
 *
 * This function has an overload that takes an `interface_invokable` as its
 * first argument, and one that takes an `InterfaceInvokable` class as its first
 * template parameter instead. For the latter, the `InterfaceInvokable` class
 * must have a static `apply` function. The `interface_invokable` or static
 * `apply` function, respectively, is expected to be invokable with the types
 * held by the `ArgumentTags`, followed by the `extra_args`. The `ArgumentTags`
 * will be prefixed as `::Tags::Interface<DirectionsTag, ArgumentTag>` and thus
 * taken from the interface, except for those specified in the `VolumeTags`.
 *
 * This function returns a `DirectionMap` that holds the value returned by
 * the `interface_invokable` in every direction of the `DirectionsTag`.
 *
 * Here is an example how to use this function:
 *
 * \snippet Test_InterfaceHelpers.cpp interface_apply_example
 *
 * Here is another example how to use this function with a stateless invokable
 * class:
 *
 * \snippet Test_InterfaceHelpers.cpp interface_apply_example_stateless
 *
 * This is the class that defines the invokable in the example above:
 *
 * \snippet Test_InterfaceHelpers.cpp interface_invokable_example
 */
template <typename DirectionsTag, typename ArgumentTags, typename VolumeTags,
          typename InterfaceInvokable, typename DbTagsList,
          typename... ExtraArgs>
SPECTRE_ALWAYS_INLINE constexpr auto interface_apply(
    InterfaceInvokable&& interface_invokable,
    const db::DataBox<DbTagsList>& box, ExtraArgs&&... extra_args) noexcept {
  return InterfaceHelpers_detail::
      InterfaceApplyImpl<DirectionsTag, ArgumentTags, VolumeTags>::apply(
          std::forward<InterfaceInvokable>(interface_invokable), box,
          std::forward<ExtraArgs>(extra_args)...);
}

// The `box` argument to this function overload is not constrained to
// `db::DataBox` to work around an issue with GCC <= 8.
//
// Details on the issue:
//
// GCC <= 8 tries to instantiate the `db::DataBox<DbTagsList>` template even
// when this function overload is _not_ SFINAE-selected. In that case the
// template parameter substitution for `DbTagsList` may contain base tags that
// causes errors with the `db::DataBox` template instantiation.
template <typename DirectionsTag, typename InterfaceInvokable,
          typename DataBoxType, typename... ExtraArgs,
          // Needed to disambiguate the overloads
          typename ArgumentTags = typename InterfaceInvokable::argument_tags>
SPECTRE_ALWAYS_INLINE constexpr auto interface_apply(
    const DataBoxType& box, ExtraArgs&&... extra_args) noexcept {
  return interface_apply<DirectionsTag, ArgumentTags,
                         get_volume_tags<InterfaceInvokable>>(
      InterfaceInvokable{}, box, std::forward<ExtraArgs>(extra_args)...);
}
// @}
