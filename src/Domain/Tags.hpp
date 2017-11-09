// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines tags related to domain quantities

#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DomainCreators/DomainCreator.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

class DataVector;
template <size_t Dim>
class Element;
template <size_t Dim>
class Index;
template <size_t Dim, typename TargetFrame>
class ElementMap;
namespace Frame {
struct Logical;
struct Inertial;
}  // namespace Frame

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup ComputationalDomainGroup
/// The input file tag for the DomainCreator to use
template <size_t Dim, typename TargetFrame>
struct DomainCreator {
  using type = std::unique_ptr<::DomainCreator<Dim, TargetFrame>>;
  static constexpr OptionString help = {"The domain to create initially"};
};
}  // namespace OptionTags

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The ::Element associated with the DataBox
template <size_t VolumeDim>
struct Element : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Element";
  using type = ::Element<VolumeDim>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The extents of DataVectors in the DataBox
template <size_t VolumeDim>
struct Extents : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Extents";
  using type = ::Index<VolumeDim>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The logical coordinates in the Element
template <size_t VolumeDim>
struct LogicalCoordinates : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "LogicalCoordinates";
  using argument_tags = tmpl::list<Tags::Extents<VolumeDim>>;
  static constexpr auto function = logical_coordinates<VolumeDim>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The coordinate map from logical to grid coordinate
template <size_t VolumeDim, typename Frame = ::Frame::Inertial>
struct ElementMap : db::DataBoxTag {
  static constexpr db::DataBoxString label = "ElementMap";
  using type = ::ElementMap<VolumeDim, Frame>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The coordinates in the target frame of `MapTag`. The `SourceCoordsTag`'s
/// frame must be the source frame of `MapTag`
template <class MapTag, class SourceCoordsTag>
struct Coordinates : db::ComputeItemTag, db::DataBoxPrefix {
  using tag = MapTag;
  static constexpr db::DataBoxString label = "Coordinates";
  static constexpr auto function(
      const db::item_type<MapTag>& element_map,
      const db::item_type<SourceCoordsTag>& source_coords) noexcept {
    return element_map(source_coords);
  }
  using argument_tags = typelist<MapTag, SourceCoordsTag>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// Computes the inverse Jacobian of the map held by `MapTag` at the coordinates
/// held by `SourceCoordsTag`. The coordinates must be in the source frame of
/// the map.
template <typename MapTag, typename SourceCoordsTag>
struct InverseJacobian : db::ComputeItemTag, db::DataBoxPrefix {
  using tag = MapTag;
  static constexpr db::DataBoxString label = "InverseJacobian";
  static constexpr auto function(
      const db::item_type<MapTag>& element_map,
      const db::item_type<SourceCoordsTag>& source_coords) noexcept {
    return element_map.inv_jacobian(source_coords);
  }
  using argument_tags = typelist<MapTag, SourceCoordsTag>;
};

namespace detail {
template <size_t VolumeDim>
auto make_unnormalized_face_normals(
    const Index<VolumeDim>& extents,
    const ::ElementMap<VolumeDim, Frame::Inertial>& map) noexcept {
  std::unordered_map<Direction<VolumeDim>,
                     tnsr::i<DataVector, VolumeDim, Frame::Inertial>>
      result;
  for (const auto& d : Direction<VolumeDim>::all_directions()) {
    result.emplace(
        d, unnormalized_face_normal(extents.slice_away(d.dimension()), map, d));
  }
  return result;
}
}  // namespace detail

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The unnormalized grid normal one form on each side
template <size_t VolumeDim>
struct UnnormalizedFaceNormal : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "UnnormalizedFaceNormal";
  static constexpr auto function =
      detail::make_unnormalized_face_normals<VolumeDim>;
  using argument_tags = tmpl::list<Extents<VolumeDim>, ElementMap<VolumeDim>>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The set of directions to neighboring Elements
template <size_t VolumeDim>
struct InternalDirections : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "InternalDirections";
  using argument_tags = tmpl::list<Element<VolumeDim>>;
  static constexpr auto function(const ::Element<VolumeDim>& element) noexcept {
    std::unordered_set<::Direction<VolumeDim>> result;
    for (const auto& direction_neighbors : element.neighbors()) {
      result.insert(direction_neighbors.first);
    }
    return result;
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// Base type for the Interface tag.  Exposed so that specializations
/// can reuse the implementation.
/// \tparam DirectionsTag the item of directions
/// \tparam NameTag the tag labeling the item
/// \tparam FunctionTag the tag to call to compute the result
template <typename DirectionsTag, typename NameTag, typename FunctionTag>
struct InterfaceBase;

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// \brief Prefix for an object on interfaces.
///
/// The contained object will be a map from ::Direction to the item
/// type of `Tag`, with the set of directions being those produced by
/// `DirectionsTag`.  If `Tag` is a compute item, its function will be
/// applied separately to the data on each interface.  If some of the
/// compute item's inputs should be taken from the volume even when
/// applied on a slice, it may indicate them using a `volume_tags`
/// typelist in the tag struct.
///
/// \tparam DirectionsTag the item of directions
/// \tparam Tag the tag labeling the item
template <typename DirectionsTag, typename Tag>
struct Interface : InterfaceBase<DirectionsTag, Tag, Tag> {};

namespace Interface_detail {
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
                                   Interface<DirectionsTag, Tag>>;
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
  template <size_t VolumeDim, typename T>
  static constexpr decltype(auto) apply(
      const ::Direction<VolumeDim>& /*direction*/, const T& arg) noexcept {
    return arg;
  }
};

template <>
struct unmap_interface_args<false> {
  template <size_t VolumeDim, typename T>
  static constexpr decltype(auto) apply(const ::Direction<VolumeDim>& direction,
                                        const T& arg) noexcept {
    return arg.at(direction);
  }
};

template <typename DirectionsTag, typename BaseComputeItem,
          typename ArgumentTags>
struct evaluate_compute_item;

template <typename DirectionsTag, typename BaseComputeItem,
          typename... ArgumentTags>
struct evaluate_compute_item<DirectionsTag, BaseComputeItem,
                             tmpl::list<ArgumentTags...>> {
  using volume_tags = typename volume_tags<BaseComputeItem>::type;
  static_assert(
      tmpl::size<tmpl::list_difference<
          volume_tags, typename BaseComputeItem::argument_tags>>::value == 0,
      "volume_tags contains tags not in argument_tags");

  static constexpr auto apply(
      const db::item_type<DirectionsTag>& directions,
      const db::item_type<ArgumentTags>&... args) noexcept {
    std::unordered_map<
        typename db::item_type<DirectionsTag>::value_type,
        std::decay_t<decltype(BaseComputeItem::function(
            unmap_interface_args<tmpl::list_contains_v<
                volume_tags, ArgumentTags>>::apply(*directions.begin(),
                                                   args)...))>>
        result;
    for (const auto& direction : directions) {
      result[direction] = BaseComputeItem::function(
          unmap_interface_args<tmpl::list_contains_v<
              volume_tags, ArgumentTags>>::apply(direction, args)...);
    }
    return result;
  }
};

template <bool IsComputeItem, typename DirectionsTag, typename NameTag,
          typename FunctionTag>
struct InterfaceImpl;

template <typename DirectionsTag, typename NameTag, typename FunctionTag>
struct InterfaceImpl<false, DirectionsTag, NameTag, FunctionTag> {
  static_assert(cpp17::is_same_v<NameTag, FunctionTag>,
                "Can't specify a function for a simple item tag");
  using tag = NameTag;
  using type =
      std::unordered_map<typename db::item_type<DirectionsTag>::value_type,
                         db::item_type<NameTag>>;
};

template <typename DirectionsTag, typename NameTag, typename FunctionTag>
struct InterfaceImpl<true, DirectionsTag, NameTag, FunctionTag>
    : db::ComputeItemTag {
  using tag = NameTag;
  using forwarded_argument_tags =
      interface_compute_item_argument_tags<DirectionsTag, FunctionTag>;
  using argument_tags =
      tmpl::push_front<forwarded_argument_tags, DirectionsTag>;
  static constexpr auto function = evaluate_compute_item<
    DirectionsTag, FunctionTag, forwarded_argument_tags>::apply;
};
}  // namespace Interface_detail

template <typename DirectionsTag, typename NameTag, typename FunctionTag>
struct InterfaceBase
    : db::DataBoxPrefix,
      Interface_detail::InterfaceImpl<db::is_compute_item_v<FunctionTag>,
                                      DirectionsTag, NameTag, FunctionTag> {
  static constexpr db::DataBoxString label = "Interface";
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// ::Direction to an interface
template <size_t VolumeDim>
struct Direction : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Direction";
  using type = ::Direction<VolumeDim>;
};

/// \cond
template <typename DirectionsTag, size_t VolumeDim>
struct Interface<DirectionsTag, Direction<VolumeDim>>
    : db::DataBoxPrefix, db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Interface";
  using tag = Direction<VolumeDim>;
  static constexpr auto function(
      const std::unordered_set<::Direction<VolumeDim>>& directions) noexcept {
    std::unordered_map<::Direction<VolumeDim>, ::Direction<VolumeDim>> result;
    for (const auto& d : directions) {
      result.emplace(d, d);
    }
    return result;
  }
  using argument_tags = tmpl::list<DirectionsTag>;
};
/// \endcond
}  // namespace Tags
