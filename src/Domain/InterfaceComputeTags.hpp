// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataBox/TagTraits.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/IndexToSliceAt.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace domain {
namespace Tags {
struct FunctionsOfTime;
}  // namespace Tags

namespace Tags {

namespace Interface_detail {

template <typename DirectionsTag, typename BaseComputeItem,
          typename ArgumentTags>
struct evaluate_compute_item;

template <typename DirectionsTag, typename BaseComputeItem,
          typename... ArgumentTags>
struct evaluate_compute_item<DirectionsTag, BaseComputeItem,
                             tmpl::list<ArgumentTags...>> {
  using volume_tags = get_volume_tags<BaseComputeItem>;
  static_assert(
      tmpl::size<tmpl::list_difference<
              volume_tags, typename BaseComputeItem::argument_tags>>::value ==
          0,
      "volume_tags contains tags not in argument_tags");

 private:
  // Order matters so we mix public/private
  template <class ComputeItem, typename = std::nullptr_t>
  struct ComputeItemType {
    using type = typename BaseComputeItem::return_type;
  };

  template <class ComputeItem>
  struct ComputeItemType<
      ComputeItem, Requires<not db::has_return_type_member_v<ComputeItem>>> {
    using type = std::decay_t<decltype(BaseComputeItem::function(
        std::declval<typename InterfaceHelpers_detail::unmap_interface_args<
            tmpl::list_contains_v<volume_tags, ArgumentTags>>::
                         template f<db::const_item_type<ArgumentTags>>>()...))>;
  };

 public:
  using return_type = std::unordered_map<
      typename db::const_item_type<DirectionsTag>::value_type,
      typename ComputeItemType<BaseComputeItem>::type>;

  template <typename... ArgTypes>
  static constexpr void apply(
      const gsl::not_null<return_type*> result,
      const db::const_item_type<DirectionsTag>& directions,
      const ArgTypes&... args) noexcept {
    apply_helper(
        std::integral_constant<bool,
                               db::has_return_type_member_v<BaseComputeItem>>{},
        result, directions, args...);
  }

 private:
  template <typename... ArgTypes>
  static constexpr void apply_helper(
      std::false_type /*has_return_type_member*/,
      const gsl::not_null<return_type*> result,
      const db::const_item_type<DirectionsTag>& directions,
      const ArgTypes&... args) noexcept {
    for (const auto& direction : directions) {
      (*result)[direction] = BaseComputeItem::function(
          InterfaceHelpers_detail::unmap_interface_args<tmpl::list_contains_v<
              volume_tags, ArgumentTags>>::apply(direction, args)...);
    }
  }

  template <typename... ArgTypes>
  static constexpr void apply_helper(
      std::true_type /*has_return_type_member*/,
      const gsl::not_null<return_type*> result,
      const db::const_item_type<DirectionsTag>& directions,
      const ArgTypes&... args) noexcept {
    for (const auto& direction : directions) {
      BaseComputeItem::function(
          make_not_null(&(*result)[direction]),
          InterfaceHelpers_detail::unmap_interface_args<tmpl::list_contains_v<
              volume_tags, ArgumentTags>>::apply(direction, args)...);
    }
  }
};

}  // namespace Interface_detail

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
struct InterfaceCompute : Interface<DirectionsTag, Tag>,
                          db::ComputeTag,
                          virtual db::PrefixTag {
  static_assert(db::is_compute_item_v<Tag>,
                "Cannot use a non compute item as an interface compute item.");
  // Defining name here prevents an ambiguous function call when using base
  // tags; Both Interface<Dirs, Tag> and Interface<Dirs, Tag::base> will have a
  // name function and so cannot be disambiguated.
  static std::string name() noexcept {
    return "Interface<" + db::tag_name<DirectionsTag>() + ", " +
           db::tag_name<Tag>() + ">";
  };
  using tag = Tag;
  using forwarded_argument_tags =
      InterfaceHelpers_detail::get_interface_argument_tags<Tag, DirectionsTag>;
  using argument_tags =
      tmpl::push_front<forwarded_argument_tags, DirectionsTag>;

  using return_type = typename Interface_detail::evaluate_compute_item<
      DirectionsTag, Tag, forwarded_argument_tags>::return_type;
  template <typename... Ts>
  static constexpr auto function(Ts&&... ts) noexcept {
    return Interface_detail::evaluate_compute_item<
        DirectionsTag, Tag,
        forwarded_argument_tags>::apply(std::forward<Ts>(ts)...);
  }
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
      db::const_item_type<DirectionsTag>::value_type::volume_dim;

  using return_type =
      std::unordered_map<::Direction<volume_dim>, db::const_item_type<Tag>>;

  static constexpr void function(
      const gsl::not_null<return_type*> sliced_vars,
      const ::Mesh<volume_dim>& mesh,
      const std::unordered_set<::Direction<volume_dim>>& directions,
      const db::const_item_type<Tag>& variables) noexcept {
    for (const auto& direction : directions) {
      data_on_slice(make_not_null(&((*sliced_vars)[direction])), variables,
                    mesh.extents(), direction.dimension(),
                    index_to_slice_at(mesh.extents(), direction));
    }
  }
  static std::string name() {
    return "Interface<" + db::tag_name<Tag>() + ">";
  };
  using argument_tags = tmpl::list<Mesh<volume_dim>, DirectionsTag, Tag>;
  using volume_tags = tmpl::list<Mesh<volume_dim>, Tag>;
};

/// \cond
template <typename DirectionsTag, size_t VolumeDim>
struct InterfaceCompute<DirectionsTag, Direction<VolumeDim>>
    : db::PrefixTag,
      db::ComputeTag,
      Tags::Interface<DirectionsTag, Direction<VolumeDim>> {
  static std::string name() noexcept { return "Interface"; }
  using tag = Direction<VolumeDim>;
  using return_type =
      std::unordered_map<::Direction<VolumeDim>, ::Direction<VolumeDim>>;
  using argument_tags = tmpl::list<DirectionsTag>;
  static constexpr auto function(
      const gsl::not_null<
          std::unordered_map<::Direction<VolumeDim>, ::Direction<VolumeDim>>*>
          result,
      const std::unordered_set<::Direction<VolumeDim>>& directions) noexcept {
    for (const auto& d : directions) {
      result->insert_or_assign(d, d);
    }
  }
};
/// \endcond

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// Computes the `VolumeDim-1` dimensional mesh on an interface from the volume
/// mesh. `Tags::InterfaceCompute<Dirs, InterfaceMesh<VolumeDim>>` is
/// retrievable as Tags::Interface<Dirs, Mesh<VolumeDim>>` from the DataBox.
template <size_t VolumeDim>
struct InterfaceMesh : db::ComputeTag, Tags::Mesh<VolumeDim - 1> {
  using base = Tags::Mesh<VolumeDim - 1>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<Direction<VolumeDim>, Mesh<VolumeDim>>;
  static constexpr auto function(
      const gsl::not_null<return_type*> mesh,
      const ::Direction<VolumeDim>& direction,
      const ::Mesh<VolumeDim>& volume_mesh) noexcept {
    *mesh = volume_mesh.slice_away(direction.dimension());
  }
  using volume_tags = tmpl::list<Mesh<VolumeDim>>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationDomainGroup
/// Computes the coordinates in the frame `Frame` on the faces defined by
/// `Direction`. Intended to be prefixed by a `Tags::InterfaceCompute` to
/// define the directions on which to compute the coordinates.
template <size_t VolumeDim, bool MovingMesh = false>
struct BoundaryCoordinates : db::ComputeTag,
                             Tags::Coordinates<VolumeDim, Frame::Inertial> {
  using base = Tags::Coordinates<VolumeDim, Frame::Inertial>;
  using return_type = typename base::type;
  static void function(
      const gsl::not_null<return_type*> boundary_coords,
      const ::Direction<VolumeDim>& direction,
      const ::Mesh<VolumeDim - 1>& interface_mesh,
      const ::ElementMap<VolumeDim, Frame::Inertial>& map) noexcept {
    *boundary_coords =
        map(interface_logical_coordinates(interface_mesh, direction));
  }

  static void function(
      const gsl::not_null<return_type*> boundary_coords,
      const ::Direction<VolumeDim>& direction,
      const ::Mesh<VolumeDim - 1>& interface_mesh,
      const ::ElementMap<VolumeDim, ::Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>&
          grid_to_inertial_map,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) noexcept {
    *boundary_coords =
        grid_to_inertial_map(logical_to_grid_map(interface_logical_coordinates(
                                 interface_mesh, direction)),
                             time, functions_of_time);
  }

  static std::string name() noexcept { return "BoundaryCoordinates"; }
  using argument_tags = tmpl::conditional_t<
      MovingMesh,
      tmpl::list<Direction<VolumeDim>, Mesh<VolumeDim - 1>,
                 Tags::ElementMap<VolumeDim, Frame::Grid>,
                 CoordinateMaps::Tags::CoordinateMap<VolumeDim, Frame::Grid,
                                                     Frame::Inertial>,
                 ::Tags::Time, domain::Tags::FunctionsOfTime>,
      tmpl::list<Direction<VolumeDim>, Mesh<VolumeDim - 1>,
                 ElementMap<VolumeDim, Frame::Inertial>>>;
  using volume_tags = tmpl::conditional_t<
      MovingMesh,
      tmpl::list<Tags::ElementMap<VolumeDim, Frame::Grid>,
                 CoordinateMaps::Tags::CoordinateMap<VolumeDim, Frame::Grid,
                                                     Frame::Inertial>,
                 ::Tags::Time, domain::Tags::FunctionsOfTime>,
      tmpl::list<ElementMap<VolumeDim, Frame::Inertial>>>;
};

}  // namespace Tags
}  // namespace domain

namespace db {
template <typename TagList, typename DirectionsTag, typename VariablesTag>
struct Subitems<
    TagList, domain::Tags::InterfaceCompute<DirectionsTag, VariablesTag>,
    Requires<tt::is_a_v<Variables, const_item_type<VariablesTag, TagList>>>>
    : detail::InterfaceSubitemsImpl<TagList, DirectionsTag, VariablesTag> {};

template <typename TagList, typename DirectionsTag, typename VariablesTag>
struct Subitems<
    TagList, domain::Tags::Slice<DirectionsTag, VariablesTag>,
    Requires<tt::is_a_v<Variables, const_item_type<VariablesTag, TagList>>>>
    : detail::InterfaceSubitemsImpl<TagList, DirectionsTag, VariablesTag> {};
}  // namespace db
