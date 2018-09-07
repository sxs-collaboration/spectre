// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines tags related to domain quantities

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/Direction.hpp"
#include "Domain/DomainCreators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Element.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Side.hpp"
#include "Options/Options.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
template <typename T>
class DataVectorImpl;
using DataVector = DataVectorImpl<double>;
/// \endcond

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
struct Element : db::SimpleTag {
  static std::string name() noexcept { return "Element"; }
  using type = ::Element<VolumeDim>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// \brief The computational grid of the Element in the DataBox
/// \details The corresponding interface tag uses Mesh::slice_through to compute
/// the mesh on the face of the element.
template <size_t VolumeDim>
struct Mesh : db::SimpleTag {
  static std::string name() noexcept { return "Mesh"; }
  using type = ::Mesh<VolumeDim>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The coordinate map from logical to grid coordinate
template <size_t VolumeDim, typename Frame = ::Frame::Inertial>
struct ElementMap : db::SimpleTag {
  static std::string name() noexcept { return "ElementMap"; }
  using type = ::ElementMap<VolumeDim, Frame>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The coordinates in a given frame.
///
/// \snippet Test_CoordinatesTag.cpp coordinates_name
template <size_t Dim, typename Frame>
struct Coordinates : db::SimpleTag {
  static std::string name() noexcept {
    return get_output(Frame{}) + "Coordinates";
  }
  using type = tnsr::I<DataVector, Dim, Frame>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The coordinates in the target frame of `MapTag`. The `SourceCoordsTag`'s
/// frame must be the source frame of `MapTag`
template <class MapTag, class SourceCoordsTag>
struct MappedCoordinates
    : Coordinates<db::item_type<MapTag>::dim,
                  typename db::item_type<MapTag>::target_frame>,
      db::ComputeTag {
  static constexpr auto function(
      const db::item_type<MapTag>& element_map,
      const db::item_type<SourceCoordsTag>& source_coords) noexcept {
    return element_map(source_coords);
  }
  using argument_tags = tmpl::list<MapTag, SourceCoordsTag>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// Computes the inverse Jacobian of the map held by `MapTag` at the coordinates
/// held by `SourceCoordsTag`. The coordinates must be in the source frame of
/// the map.
template <typename MapTag, typename SourceCoordsTag>
struct InverseJacobian : db::ComputeTag, db::PrefixTag {
  using tag = MapTag;
  static std::string name() noexcept { return "InverseJacobian"; }
  static constexpr auto function(
      const db::item_type<MapTag>& element_map,
      const db::item_type<SourceCoordsTag>& source_coords) noexcept {
    return element_map.inv_jacobian(source_coords);
  }
  using argument_tags = tmpl::list<MapTag, SourceCoordsTag>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup DomainGroup
/// Base tag for boundary data needed for updating the variables.
struct VariablesBoundaryData : db::BaseTag {};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The set of directions to neighboring Elements
template <size_t VolumeDim>
struct InternalDirections : db::ComputeTag {
  static std::string name() noexcept { return "InternalDirections"; }
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
/// The set of directions which correspond to external boundaries
template <size_t VolumeDim>
struct BoundaryDirections : db::ComputeTag {
  static std::string name() noexcept { return "BoundaryDirections"; }
  using argument_tags = tmpl::list<Element<VolumeDim>>;
  static constexpr auto function(const ::Element<VolumeDim>& element) noexcept {
    return element.external_boundaries();
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
/// If using the base tag mechanism for an interface tag is desired,
/// then `Tag` can have a `base` type alias pointing to its base
/// class.  (This requirement is due to the lack of a way to determine
/// a type's base classes in C++.)
///
/// If the type of the item is a Variables, then each tag in the
/// Variables must specify whether it should be sliced from the volume
/// or computed (or set) on the interface.  This is done using a
/// `static constexpr bool should_be_sliced_to_boundary` member of the
/// tag.  All tags in the Variables must have the same value for this
/// member.  DataBox tags for sliced Variables will be compute items
/// performing the slicing.  %Tags for non-sliced Variables will
/// perform the same action as the corresponding volume tag.
///
/// It must be possible to determine the type associated with `Tag`
/// without reference to a DataBox, i.e., `db::item_type<Tag>` must
/// work.
///
/// \tparam DirectionsTag the item of directions
/// \tparam Tag the tag labeling the item
template <typename DirectionsTag, typename Tag, typename = std::nullptr_t>
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

template <typename DirectionsTag, typename Tag, typename = cpp17::void_t<>>
struct GetBaseTagIfPresent {};

template <typename DirectionsTag, typename Tag>
struct GetBaseTagIfPresent<DirectionsTag, Tag,
                           cpp17::void_t<typename Tag::base>>
    : Interface<DirectionsTag, typename Tag::base> {
  static_assert(cpp17::is_base_of_v<typename Tag::base, Tag>,
                "Tag `base` alias must be a base class of `Tag`.");
};

template <bool IsComputeItem, typename DirectionsTag, typename NameTag,
          typename FunctionTag>
struct InterfaceImpl;

template <typename DirectionsTag, typename NameTag, typename FunctionTag>
struct InterfaceImpl<false, DirectionsTag, NameTag, FunctionTag>
    : virtual db::SimpleTag, GetBaseTagIfPresent<DirectionsTag, NameTag> {
  static_assert(cpp17::is_same_v<NameTag, FunctionTag>,
                "Can't specify a function for a simple item tag");
  using tag = NameTag;
  using type =
      std::unordered_map<typename db::item_type<DirectionsTag>::value_type,
                         db::item_type<NameTag>>;
};

template <typename DirectionsTag, typename NameTag, typename FunctionTag>
struct InterfaceImpl<true, DirectionsTag, NameTag, FunctionTag>
    : db::ComputeTag, GetBaseTagIfPresent<DirectionsTag, NameTag> {
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
    : virtual db::PrefixTag,
      Interface_detail::InterfaceImpl<db::is_compute_item_v<FunctionTag>,
                                      DirectionsTag, NameTag, FunctionTag> {
  static std::string name() noexcept { return "Interface"; }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// ::Direction to an interface
template <size_t VolumeDim>
struct Direction : db::SimpleTag {
  static std::string name() noexcept { return "Direction"; }
  using type = ::Direction<VolumeDim>;
};

/// \cond
template <typename DirectionsTag, size_t VolumeDim>
struct Interface<DirectionsTag, Direction<VolumeDim>> : db::PrefixTag,
                                                        db::ComputeTag {
  static std::string name() noexcept { return "Interface"; }
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

namespace detail {
template <size_t VolumeDim>
struct InterfaceMesh : db::ComputeTag {
  static constexpr auto function(
      const ::Direction<VolumeDim>& direction,
      const ::Mesh<VolumeDim>& volume_mesh) noexcept {
    return volume_mesh.slice_away(direction.dimension());
  }
  using argument_tags = tmpl::list<Direction<VolumeDim>, Mesh<VolumeDim>>;
  using volume_tags = tmpl::list<Mesh<VolumeDim>>;
};
}  // namespace detail

/// \cond
template <typename DirectionsTag, size_t InterfaceDim>
struct Interface<DirectionsTag, Mesh<InterfaceDim>>
    : InterfaceBase<DirectionsTag, Mesh<InterfaceDim>,
                    detail::InterfaceMesh<InterfaceDim + 1>> {};
/// \endcond
}  // namespace Tags

namespace db {
template <typename TagList, typename DirectionsTag, typename VariablesTag>
struct Subitems<
    TagList, Tags::Interface<DirectionsTag, VariablesTag>,
    Requires<tt::is_a_v<Variables, item_type<VariablesTag, TagList>>>> {
  using type = tmpl::transform<
      typename item_type<VariablesTag>::tags_list,
      tmpl::bind<Tags::Interface, tmpl::pin<DirectionsTag>, tmpl::_1>>;

  using tag = Tags::Interface<DirectionsTag, VariablesTag>;

  template <typename Subtag>
  static void create_item(
      const gsl::not_null<item_type<tag>*> parent_value,
      const gsl::not_null<item_type<Subtag>*> sub_value) noexcept {
    sub_value->clear();
    for (auto& direction_vars : *parent_value) {
      const auto& direction = direction_vars.first;
      auto& parent_vars = get<typename Subtag::tag>(direction_vars.second);
      auto& sub_var = (*sub_value)[direction];
      for (auto vars_it = parent_vars.begin(), sub_var_it = sub_var.begin();
           vars_it != parent_vars.end(); ++vars_it, ++sub_var_it) {
        sub_var_it->set_data_ref(&*vars_it);
      }
    }
  }

  template <typename Subtag>
  static auto create_compute_item(const item_type<tag>& parent_value) noexcept {
    item_type<Subtag> sub_value;
    for (const auto& direction_vars : parent_value) {
      const auto& direction = direction_vars.first;
      const auto& parent_vars =
          get<typename Subtag::tag>(direction_vars.second);
      auto& sub_var = sub_value[direction];
      auto sub_var_it = sub_var.begin();
      for (auto vars_it = parent_vars.begin();
           vars_it != parent_vars.end(); ++vars_it, ++sub_var_it) {
        // clang-tidy: do not use const_cast
        // The DataBox will only give out a const reference to the
        // result of a compute item.  Here, that is a reference to a
        // const map to Tensors of DataVectors.  There is no (publicly
        // visible) indirection there, so having the map const will
        // allow only allow const access to the contained DataVectors,
        // so no modification through the pointer cast here is
        // possible.
        sub_var_it->set_data_ref(const_cast<DataVector*>(&*vars_it));  // NOLINT
      }
    }
    return sub_value;
  }
};
}  // namespace db

namespace Tags {
namespace detail {
template <size_t VolumeDim, typename>
struct Slice;

template <size_t VolumeDim, typename... SlicedTags>
struct Slice<VolumeDim, tmpl::list<SlicedTags...>> : db::ComputeTag {
  static constexpr auto function(
      const ::Mesh<VolumeDim>& mesh, const ::Direction<VolumeDim>& direction,
      const db::item_type<SlicedTags>&... tensors) noexcept {
    return data_on_slice<SlicedTags...>(
        mesh.extents(), direction.dimension(),
        index_to_slice_at(mesh.extents(), direction), tensors...);
  }
  using argument_tags =
      tmpl::list<Mesh<VolumeDim>, Direction<VolumeDim>, SlicedTags...>;
  using volume_tags = tmpl::list<Mesh<VolumeDim>, SlicedTags...>;
};

template <typename Tag>
struct ShouldBeSlicedToBoundary {
  using type = std::integral_constant<bool, Tag::should_be_sliced_to_boundary>;
};
}  // namespace detail

/// \cond
template <typename DirectionsTag, typename VariablesTag>
struct Interface<
    DirectionsTag, VariablesTag,
    Requires<tmpl::all<typename db::item_type<VariablesTag>::tags_list,
                       detail::ShouldBeSlicedToBoundary<tmpl::_1>>::value>>
    : InterfaceBase<
          DirectionsTag, VariablesTag,
          detail::Slice<db::item_type<DirectionsTag>::value_type::volume_dim,
                        typename db::item_type<VariablesTag>::tags_list>> {};

template <typename DirectionsTag, typename VariablesTag>
struct Interface<
    DirectionsTag, VariablesTag,
    Requires<
        tmpl::any<typename db::item_type<VariablesTag>::tags_list,
                  detail::ShouldBeSlicedToBoundary<tmpl::_1>>::value and
        not tmpl::all<typename db::item_type<VariablesTag>::tags_list,
                      detail::ShouldBeSlicedToBoundary<tmpl::_1>>::value>> {
  // This static_assert should always trigger. We compare the VariablesTag to a
  // very unusual type that in general will not appear as a tag.
  static_assert(
      cpp17::is_same_v<VariablesTag, void* volatile const* volatile const*>,
      "Cannot compute a partially-sliced Variables");
};
/// \endcond
}  // namespace Tags
