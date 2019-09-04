// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "Domain/Direction.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace InterfaceHelpers_detail {

// Pull volume_tags from BaseComputeItem, defaulting to an empty list.
template <typename BaseComputeItem, typename = cpp17::void_t<>>
struct volume_tags {
  using type = tmpl::list<>;
};

template <typename BaseComputeItem>
struct volume_tags<BaseComputeItem,
                   cpp17::void_t<typename BaseComputeItem::volume_tags>> {
  using type = typename BaseComputeItem::volume_tags;
};

// Add an Interface wrapper to a tag if it is not listed as being
// taken from the volume.
template <typename DirectionsTag, typename Tag, typename VolumeTags>
struct interface_compute_item_argument_tag {
  using type = tmpl::conditional_t<tmpl::list_contains_v<VolumeTags, Tag>, Tag,
                                   ::Tags::Interface<DirectionsTag, Tag>>;
};

// Compute the argument tags for the interface version of a compute item.
template <typename DirectionsTag, typename BaseComputeItem>
using interface_compute_item_argument_tags = tmpl::transform<
    typename BaseComputeItem::argument_tags,
    interface_compute_item_argument_tag<
        tmpl::pin<DirectionsTag>, tmpl::_1,
        tmpl::pin<typename volume_tags<BaseComputeItem>::type>>>;

// Pull the direction's entry from interface arguments, passing volume
// arguments through unchanged.
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
