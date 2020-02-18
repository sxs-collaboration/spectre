// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DirectionMap.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace InterfaceHelpers_detail {

template <typename T, typename = cpp17::void_t<>>
struct get_volume_tags_impl {
  using type = tmpl::list<>;
};
template <typename T>
struct get_volume_tags_impl<T, cpp17::void_t<typename T::volume_tags>> {
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
                                   ::Tags::Interface<DirectionsTag, Tag>>;
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

template <typename DirectionsTag, typename VolumeTags,
          typename InterfaceInvokable, typename DbTagsList,
          typename... ArgumentTags, typename... ExtraArgs>
SPECTRE_ALWAYS_INLINE constexpr auto interface_apply_impl(
    InterfaceInvokable&& interface_invokable,
    const db::DataBox<DbTagsList>& box, tmpl::list<ArgumentTags...> /*meta*/,
    ExtraArgs&&... extra_args) noexcept {
  using interface_return_type = std::decay_t<decltype(interface_invokable(
      std::declval<db::const_item_type<ArgumentTags, DbTagsList>>()...,
      std::declval<ExtraArgs&&>()...))>;
  constexpr size_t volume_dim = DirectionsTag::volume_dim;
  DirectionMap<volume_dim, interface_return_type> result{};
  for (const auto& direction : get<DirectionsTag>(box)) {
    auto interface_value = interface_invokable(
        unmap_interface_args<tmpl::list_contains_v<VolumeTags, ArgumentTags>>::
            apply(direction,
                  get<tmpl::type_from<make_interface_tag_impl<
                      ArgumentTags, DirectionsTag, VolumeTags>>>(box))...,
        extra_args...);
    result.insert({direction, std::move(interface_value)});
  }
  return result;
}

}  // namespace InterfaceHelpers_detail

/*!
 * \brief Apply the `interface_invokable` to the `box` on all interfaces given
 * by the `DirectionsTag`.
 *
 * \details The `interface_invokable` is expected to be invokable with the types
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
 */
template <typename DirectionsTag, typename ArgumentTags, typename VolumeTags,
          typename InterfaceInvokable, typename DbTagsList,
          typename... ExtraArgs>
SPECTRE_ALWAYS_INLINE constexpr auto interface_apply(
    InterfaceInvokable&& interface_invokable,
    const db::DataBox<DbTagsList>& box, ExtraArgs&&... extra_args) noexcept {
  return InterfaceHelpers_detail::interface_apply_impl<DirectionsTag,
                                                       VolumeTags>(
      std::forward<InterfaceInvokable>(interface_invokable), box,
      ArgumentTags{}, std::forward<ExtraArgs>(extra_args)...);
}
