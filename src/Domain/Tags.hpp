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
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/LogicalCoordinates.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"
#include "Domain/Side.hpp"
#include "Options/Options.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
class DataVector;
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
/// The ::Domain.
template <size_t VolumeDim, typename Frame>
struct Domain : db::SimpleTag {
  static std::string name() noexcept { return "Domain"; }
  using type = ::Domain<VolumeDim, Frame>;
};

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
/// The set of directions which correspond to external boundaries.
/// Used for representing data on the interior side of the external boundary
/// faces.
template <size_t VolumeDim>
struct BoundaryDirectionsInterior : db::ComputeTag {
  static std::string name() noexcept { return "BoundaryDirectionsInterior"; }
  using argument_tags = tmpl::list<Element<VolumeDim>>;
  static constexpr auto function(const ::Element<VolumeDim>& element) noexcept {
    return element.external_boundaries();
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The set of directions which correspond to external boundaries. To be used
/// to represent data which exists on the exterior side of the external boundary
/// faces.
template <size_t VolumeDim>
struct BoundaryDirectionsExterior : db::ComputeTag {
  static std::string name() noexcept { return "BoundaryDirectionsExterior"; }
  using argument_tags = tmpl::list<Element<VolumeDim>>;
  static constexpr auto function(const ::Element<VolumeDim>& element) noexcept {
    return element.external_boundaries();
  }
};


/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// \brief Tag which is either a SimpleTag for quantities on an
/// interface, base tag to a compute item which acts on tags on an interface, or
/// base tag to a compute item which slices a tag from the volume to an
/// interface.
///
/// The contained object will be a map from ::Direction to the item type of
/// `Tag`, with the set of directions being those produced by `DirectionsTag`.
///
/// If a SimpleTag is desired on the interface, then this tag can be added to
/// the DataBox directly. If a ComputeTag which acts on Tags on the interface is
/// desired, then the tag should be added using `InterfaceComputeItem`. If a
/// ComputeTag which slices a TensorTag or a VariablesTag in the volume to an
/// Interface is desired, then it should be added using `Slice`. In all cases,
/// the tag can then be retrieved using `Tags::Interface<DirectionsTag, Tag>`.
///
/// If using the base tag mechanism for an interface tag is desired,
/// then `Tag` can have a `base` type alias pointing to its base
/// class.  (This requirement is due to the lack of a way to determine
/// a type's base classes in C++.)
///
/// It must be possible to determine the type associated with `Tag`
/// without reference to a DataBox, i.e., `db::item_type<Tag>` must
/// work.
///
/// \tparam DirectionsTag the item of directions
/// \tparam Tag the tag labeling the item
///
/// \see InterfaceComputeItem, Slice
template <typename DirectionsTag, typename Tag>
struct Interface;

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

 private:
  // Order matters so we mix public/private
  template <class ComputeItem, bool = db::has_return_type_member_v<ComputeItem>>
  struct ComputeItemType {
    using type = std::decay_t<decltype(BaseComputeItem::function(
        std::declval<typename unmap_interface_args<
            tmpl::list_contains_v<volume_tags, ArgumentTags>>::
                         template f<db::item_type<ArgumentTags>>>()...))>;
  };

  template <class ComputeItem>
  struct ComputeItemType<ComputeItem, true> {
    using type = typename BaseComputeItem::return_type;
  };

 public:
  using return_type =
      std::unordered_map<typename db::item_type<DirectionsTag>::value_type,
                         typename ComputeItemType<BaseComputeItem>::type>;

  static constexpr void apply(
      const gsl::not_null<return_type*> result,
      const db::item_type<DirectionsTag>& directions,
      const db::item_type<ArgumentTags>&... args) noexcept {
    apply_helper(
        std::integral_constant<bool,
                               db::has_return_type_member_v<BaseComputeItem>>{},
        result, directions, args...);
  }

 private:
  static constexpr void apply_helper(
      std::false_type /*has_return_type_member*/,
      const gsl::not_null<return_type*> result,
      const db::item_type<DirectionsTag>& directions,
      const db::item_type<ArgumentTags>&... args) noexcept {
    for (const auto& direction : directions) {
      (*result)[direction] = BaseComputeItem::function(
          unmap_interface_args<tmpl::list_contains_v<
              volume_tags, ArgumentTags>>::apply(direction, args)...);
    }
  }

  static constexpr void apply_helper(
      std::true_type /*has_return_type_member*/,
      const gsl::not_null<return_type*> result,
      const db::item_type<DirectionsTag>& directions,
      const db::item_type<ArgumentTags>&... args) noexcept {
    for (const auto& direction : directions) {
      BaseComputeItem::function(
          make_not_null(&(*result)[direction]),
          unmap_interface_args<tmpl::list_contains_v<
              volume_tags, ArgumentTags>>::apply(direction, args)...);
    }
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
}  // namespace Interface_detail

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// ::Direction to an interface
template<size_t VolumeDim>
struct Direction : db::SimpleTag {
  static std::string name() noexcept { return "Direction"; }
  using type = ::Direction<VolumeDim>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// Computes the `VolumeDim-1` dimensional mesh on an interface from the volume
/// mesh. `Tags::InterfaceComputeItem<Dirs, InterfaceMesh<VolumeDim>>` is
/// retrievable as Tags::Interface<Dirs, Mesh<VolumeDim>>` from the DataBox.
template<size_t VolumeDim>
struct InterfaceMesh : db::ComputeTag, Tags::Mesh<VolumeDim - 1> {
  static constexpr auto function(
      const ::Direction<VolumeDim> &direction,
      const ::Mesh<VolumeDim> &volume_mesh) noexcept {
    return volume_mesh.slice_away(direction.dimension());
  }
  using base = Tags::Mesh<VolumeDim - 1>;
  using argument_tags = tmpl::list<Direction<VolumeDim>, Mesh<VolumeDim>>;
  using volume_tags = tmpl::list<Mesh<VolumeDim>>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationDomainGroup
/// Computes the coordinates in the frame `Frame` on the faces defined by
/// `Direction`. Intended to be prefixed by a `Tags::InterfaceComputeItem` to
/// define the directions on which to compute the coordinates.
template <size_t VolumeDim, typename Frame = ::Frame::Inertial>
struct BoundaryCoordinates : db::ComputeTag,
                             Tags::Coordinates<VolumeDim, Frame> {
  static constexpr auto function(
      const ::Direction<VolumeDim> &direction,
      const ::Mesh<VolumeDim - 1> &interface_mesh,
      const ::ElementMap<VolumeDim, Frame> &map) noexcept {
    return map(interface_logical_coordinates(interface_mesh, direction));
  }
  static std::string name() noexcept { return "BoundaryCoordinates"; }
  using base = Tags::Coordinates<VolumeDim, Frame>;
  using argument_tags = tmpl::list<Direction<VolumeDim>, Mesh<VolumeDim - 1>,
                                   ElementMap<VolumeDim, Frame>>;
  using volume_tags = tmpl::list<ElementMap<VolumeDim, Frame>>;
};

// Virtual inheritance is used here to prevent a compiler warning: Derived class
// SimpleTag is inaccessible
template <typename DirectionsTag, typename Tag>
struct Interface : virtual db::SimpleTag,
                   Interface_detail::GetBaseTagIfPresent<DirectionsTag, Tag> {
  static std::string name() noexcept {
    return "Interface<" + DirectionsTag::name() + ", " + Tag::name() + ">";
  };
  using tag = Tag;
  using type =
      std::unordered_map<typename db::item_type<DirectionsTag>::value_type,
                         db::item_type<Tag>>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// \brief Derived tag for representing a compute item which acts on Tags on an
/// interface. Can be retrieved using Tags::Interface<DirectionsTag, Tag>
///
/// The contained object will be a map from ::Direction to the item type of
/// `Tag`, with the set of directions being those produced by `DirectionsTag`.
/// `Tag::function` will be applied separately to the data on each interface. If
/// some of the compute item's inputs should be taken from the volume even when
/// applied on a slice, it may indicate them using `volume_tags`.
///
/// If using the base tag mechanism for an interface tag is desired,
/// then `Tag` can have a `base` type alias pointing to its base
/// class.  (This requirement is due to the lack of a way to determine
/// a type's base classes in C++.)
///
/// \tparam DirectionsTag the item of Directions
/// \tparam Tag the tag labeling the item
template <typename DirectionsTag, typename Tag>
struct InterfaceComputeItem
    : Interface<DirectionsTag, Tag>,
      db::ComputeTag,
      virtual db::PrefixTag {
  // Defining name here prevents an ambiguous function call when using base
  // tags; Both Interface<Dirs, Tag> and Interface<Dirs, Tag::base> will have a
  // name function and so cannot be disambiguated.
  static std::string name() noexcept {
    return "Interface<" + DirectionsTag::name() + ", " + Tag::name() + ">";
  };
  using tag = Tag;
  using forwarded_argument_tags =
      Interface_detail::interface_compute_item_argument_tags<DirectionsTag,
                                                             Tag>;
  using argument_tags =
      tmpl::push_front<forwarded_argument_tags, DirectionsTag>;

  using return_type = typename Interface_detail::evaluate_compute_item<
      DirectionsTag, Tag, forwarded_argument_tags>::return_type;
  static constexpr auto function =
      Interface_detail::evaluate_compute_item<DirectionsTag, Tag,
                                              forwarded_argument_tags>::apply;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// \brief Derived tag for representing a compute item which slices a Tag
/// containing a `Tensor` or a `Variables` from the volume to an interface.
/// Retrievable from the DataBox using `Tags::Interface<DirectionsTag, Tag>`
///
/// The contained object will be a map from ::Direction to the item
/// type of `Tag`, with the set of directions being those produced by
/// `DirectionsTag`.
///
/// \requires `Tag` correspond to a `Tensor` or a `Variables`
///
/// \tparam DirectionsTag the item of Directions
/// \tparam Tag the tag labeling the item
template <typename DirectionsTag, typename Tag>
struct Slice : Interface<DirectionsTag, Tag>, db::ComputeTag {
  static constexpr size_t volume_dim =
      db::item_type<DirectionsTag>::value_type::volume_dim;

  using return_type =
      std::unordered_map<::Direction<volume_dim>, db::item_type<Tag>>;

  static constexpr void function(
      const gsl::not_null<
          std::unordered_map<::Direction<volume_dim>, db::item_type<Tag>>*>
          sliced_vars,
      const ::Mesh<volume_dim>& mesh,
      const std::unordered_set<::Direction<volume_dim>>& directions,
      const db::item_type<Tag>& variables) noexcept {
    for (const auto& direction : directions) {
      data_on_slice(make_not_null(&((*sliced_vars)[direction])), variables,
                    mesh.extents(), direction.dimension(),
                    index_to_slice_at(mesh.extents(), direction));
    }
  }
  static std::string name() { return "Interface<" + Tag::name() + ">"; };
  using argument_tags = tmpl::list<Mesh<volume_dim>, DirectionsTag, Tag>;
  using volume_tags = tmpl::list<Mesh<volume_dim>, Tag>;
};

/// \cond
template <typename DirectionsTag, size_t VolumeDim>
struct InterfaceComputeItem<DirectionsTag, Direction<VolumeDim>>
    : db::PrefixTag,
      db::ComputeTag,
      Tags::Interface<DirectionsTag, Direction<VolumeDim>> {
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
}  // namespace Tags

namespace db {
namespace detail {
template <typename TagList, typename DirectionsTag, typename VariablesTag>
struct InterfaceSubitemsImpl {
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

  // The `return_type` can be anything for Subitems because the DataBox figures
  // out the correct return type, we just use the `return_type` type alias to
  // signal to the DataBox we want mutating behavior.
  using return_type = NoSuchType;

  template <typename Subtag>
  static void create_compute_item(
      const gsl::not_null<item_type<Subtag>*> sub_value,
      const item_type<tag>& parent_value) noexcept {
    for (const auto& direction_vars : parent_value) {
      const auto& direction = direction_vars.first;
      const auto& parent_vars =
          get<typename Subtag::tag>(direction_vars.second);
      auto& sub_var = (*sub_value)[direction];
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
  }
};
}  // namespace detail

template <typename TagList, typename DirectionsTag, typename VariablesTag>
struct Subitems<
    TagList, Tags::Interface<DirectionsTag, VariablesTag>,
    Requires<tt::is_a_v<Variables, item_type<VariablesTag, TagList>>>>
    : detail::InterfaceSubitemsImpl<TagList, DirectionsTag, VariablesTag> {
};
template <typename TagList, typename DirectionsTag, typename VariablesTag>
struct Subitems<
    TagList, Tags::InterfaceComputeItem<DirectionsTag, VariablesTag>,
    Requires<tt::is_a_v<Variables, item_type<VariablesTag, TagList>>>>
    : detail::InterfaceSubitemsImpl<TagList, DirectionsTag, VariablesTag> {};

template <typename TagList, typename DirectionsTag, typename VariablesTag>
struct Subitems<
    TagList, Tags::Slice<DirectionsTag, VariablesTag>,
    Requires<tt::is_a_v<Variables, item_type<VariablesTag, TagList>>>>
: detail::InterfaceSubitemsImpl<TagList, DirectionsTag, VariablesTag> {};
}  // namespace db
