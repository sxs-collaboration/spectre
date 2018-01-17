// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines tags related to domain quantities

#pragma once

#include <memory>
#include <unordered_map>

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

}  // namespace Tags
