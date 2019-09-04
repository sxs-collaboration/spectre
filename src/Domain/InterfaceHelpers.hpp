// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

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

}  // namespace InterfaceHelpers_detail
