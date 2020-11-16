// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Subitems.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataBox/TagTraits.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
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

template <typename ComputeTag, typename VolumeTags, typename... ArgumentTags,
          size_t Dim, typename... ArgTypes>
constexpr void evaluate_compute_item(
    const gsl::not_null<typename ComputeTag::type*> result,
    const ::Direction<Dim>& direction, tmpl::list<ArgumentTags...> /*meta*/,
    const ArgTypes&... args) noexcept {
  ComputeTag::function(
      result,
      InterfaceHelpers_detail::unmap_interface_args<
          tmpl::list_contains_v<VolumeTags, ArgumentTags>>::apply(direction,
                                                                  args)...);
}
}  // namespace Interface_detail

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// \brief Compute tag for representing items computed on a set of interfaces.
/// Can be retrieved using `Tags::Interface<DirectionsTag, Tag>`
///
/// The computed object will be a map from a ::Direction to the item type of
/// `Tag` (i.e. `Tag::type`), with the set of directions being those produced by
/// `DirectionsTag`. `Tag::function` will be applied separately to the data on
/// each interface. If some of the compute item's inputs (i.e.
/// `Tag::argument_tags`) should be taken from the volume rather than the
/// interface, they should be specified in the typelist `Tag::volume_tags`.
///
/// \tparam DirectionsTag the simple tag labeling the set of Directions
/// \tparam Tag the compute tag to apply on each interface
template <typename DirectionsTag, typename Tag>
struct InterfaceCompute : Interface<DirectionsTag, typename Tag::base>,
                          db::ComputeTag {
  static_assert(db::is_simple_tag_v<DirectionsTag>);
  static_assert(db::is_compute_tag_v<Tag>,
                "Cannot use a non compute item as an interface compute item.");
  using base = Interface<DirectionsTag, typename Tag::base>;
  static constexpr size_t volume_dim = DirectionsTag::volume_dim;
  using return_type =
      std::unordered_map<::Direction<volume_dim>, typename Tag::type>;

  using forwarded_argument_tags =
      InterfaceHelpers_detail::get_interface_argument_tags<Tag, DirectionsTag>;
  using argument_tags =
      tmpl::push_front<forwarded_argument_tags, DirectionsTag>;

  using volume_tags = get_volume_tags<Tag>;
  static_assert(
      tmpl::size<tmpl::list_difference<volume_tags, argument_tags>>::value == 0,
      "volume_tags contains tags not in argument_tags");

  template <typename... ArgTypes>
  static constexpr void function(
      const gsl::not_null<return_type*> result,
      const std::unordered_set<::Direction<volume_dim>>& directions,
      const ArgTypes&... args) noexcept {
    for (const auto& direction : directions) {
      Interface_detail::evaluate_compute_item<Tag, volume_tags>(
          make_not_null(&(*result)[direction]), direction,
          forwarded_argument_tags{}, args...);
    }
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// \brief Compute tag for representing a compute item that slices data from
/// the volume to a set of interfaces.
///
/// The computed object will be a map from a ::Direction to the item type of
/// `Tag` (i.e. `Tag::type`), with the set of directions being those produced by
/// `DirectionsTag`. `Tag::type` must be a `Tensor` or a `Variables`.
/// Retrievable from the DataBox using `Tags::Interface<DirectionsTag, Tag>`
///
/// \requires `Tag` correspond to a `Tensor` or a `Variables`
///
/// \tparam DirectionsTag the simple tag labeling the set of Directions
/// \tparam Tag the simple tag for the volume data to be sliced
template <typename DirectionsTag, typename Tag>
struct Slice : Interface<DirectionsTag, Tag>, db::ComputeTag {
  static_assert(db::is_simple_tag_v<DirectionsTag>);
  static_assert(db::is_simple_tag_v<Tag>);
  using base = Interface<DirectionsTag, Tag>;
  static constexpr size_t volume_dim = DirectionsTag::volume_dim;

  using return_type =
      std::unordered_map<::Direction<volume_dim>, typename Tag::type>;

  static constexpr void function(
      const gsl::not_null<return_type*> sliced_vars,
      const ::Mesh<volume_dim>& mesh,
      const std::unordered_set<::Direction<volume_dim>>& directions,
      const typename Tag::type& variables) noexcept {
    for (const auto& direction : directions) {
      data_on_slice(make_not_null(&((*sliced_vars)[direction])), variables,
                    mesh.extents(), direction.dimension(),
                    index_to_slice_at(mesh.extents(), direction));
    }
  }

  using argument_tags = tmpl::list<Mesh<volume_dim>, DirectionsTag, Tag>;
  using volume_tags = tmpl::list<Mesh<volume_dim>, Tag>;
};

/// \cond
template <typename DirectionsTag, size_t VolumeDim>
struct InterfaceCompute<DirectionsTag, Direction<VolumeDim>>
    : db::ComputeTag, Interface<DirectionsTag, Direction<VolumeDim>> {
  static_assert(db::is_simple_tag_v<DirectionsTag>);
  using base = Interface<DirectionsTag, Direction<VolumeDim>>;
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
/// retrievable as `Tags::Interface<Dirs, Mesh<VolumeDim>>` from the DataBox.
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
/// \ingroup ComputationalDomainGroup
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
template <typename DirectionsTag, typename VariablesTag>
struct Subitems<domain::Tags::InterfaceCompute<DirectionsTag, VariablesTag>,
                Requires<tt::is_a_v<Variables, typename VariablesTag::type>>>
    : detail::InterfaceSubitemsImpl<DirectionsTag,
                                    typename VariablesTag::base> {};

template <typename DirectionsTag, typename VariablesTag>
struct Subitems<domain::Tags::Slice<DirectionsTag, VariablesTag>,
                Requires<tt::is_a_v<Variables, typename VariablesTag::type>>>
    : detail::InterfaceSubitemsImpl<DirectionsTag, VariablesTag> {};
}  // namespace db
