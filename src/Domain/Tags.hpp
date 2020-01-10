// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines tags related to domain quantities

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Side.hpp"
#include "Options/Options.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
class DataVector;
/// \endcond

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// \ingroup ComputationalDomainGroup
/// The input file tag for the DomainCreator to use
template <size_t Dim>
struct DomainCreator {
  using type = std::unique_ptr<::DomainCreator<Dim>>;
  static constexpr OptionString help = {"The domain to create initially"};
};
}  // namespace OptionTags

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The ::Domain.
template <size_t VolumeDim>
struct Domain : db::SimpleTag {
  using type = ::Domain<VolumeDim>;
  using option_tags = tmpl::list<::OptionTags::DomainCreator<VolumeDim>>;

  template <typename Metavariables>
  static ::Domain<VolumeDim> create_from_options(
      const std::unique_ptr<::DomainCreator<VolumeDim>>&
          domain_creator) noexcept {
    return domain_creator->create_domain();
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The number of grid points per dimension for all elements in each block of
/// the initial computational domain
template <size_t Dim>
struct InitialExtents : db::SimpleTag {
  using type = std::vector<std::array<size_t, Dim>>;
  using option_tags = tmpl::list<::OptionTags::DomainCreator<Dim>>;

  template <typename Metavariables>
  static std::vector<std::array<size_t, Dim>> create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) noexcept {
    return domain_creator->initial_extents();
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The initial refinement level per dimension for all elements in each block of
/// the initial computational domain
template <size_t Dim>
struct InitialRefinementLevels : db::SimpleTag {
  using type = std::vector<std::array<size_t, Dim>>;
  using option_tags = tmpl::list<::OptionTags::DomainCreator<Dim>>;

  template <typename Metavariables>
  static std::vector<std::array<size_t, Dim>> create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) noexcept {
    return domain_creator->initial_refinement_levels();
  }
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The ::Element associated with the DataBox
template <size_t VolumeDim>
struct Element : db::SimpleTag {
  using type = ::Element<VolumeDim>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// \brief The computational grid of the Element in the DataBox
/// \details The corresponding interface tag uses Mesh::slice_through to compute
/// the mesh on the face of the element.
template <size_t VolumeDim>
struct Mesh : db::SimpleTag {
  using type = ::Mesh<VolumeDim>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The coordinate map from logical to grid coordinate
template <size_t VolumeDim, typename Frame = ::Frame::Inertial>
struct ElementMap : db::SimpleTag {
  static std::string name() noexcept {
    return "ElementMap(" + get_output(Frame{}) + ")";
  }
  using type = ::ElementMap<VolumeDim, Frame>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// The coordinates in a given frame.
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
    : Coordinates<db::const_item_type<MapTag>::dim,
                  typename db::const_item_type<MapTag>::target_frame>,
      db::ComputeTag {
  static constexpr auto function(
      const db::const_item_type<MapTag>& element_map,
      const db::const_item_type<SourceCoordsTag>& source_coords) noexcept {
    return element_map(source_coords);
  }
  using argument_tags = tmpl::list<MapTag, SourceCoordsTag>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// \brief The inverse Jacobian from the source frame to the target frame.
///
/// Specifically, \f$\partial x^{\bar{i}} / \partial x^i\f$, where \f$\bar{i}\f$
/// denotes the source frame and \f$i\f$ denotes the target frame.
template <size_t Dim, typename SourceFrame, typename TargetFrame>
struct InverseJacobian : db::SimpleTag {
  static std::string name() noexcept {
    return "InverseJacobian(" + get_output(SourceFrame{}) + "," +
           get_output(TargetFrame{}) + ")";
  }
  using type = ::InverseJacobian<DataVector, Dim, SourceFrame, TargetFrame>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// Computes the inverse Jacobian of the map held by `MapTag` at the coordinates
/// held by `SourceCoordsTag`. The coordinates must be in the source frame of
/// the map.
template <typename MapTag, typename SourceCoordsTag>
struct InverseJacobianCompute
    : InverseJacobian<db::const_item_type<MapTag>::dim,
                      typename db::const_item_type<MapTag>::source_frame,
                      typename db::const_item_type<MapTag>::target_frame>,
      db::ComputeTag,
      db::PrefixTag {
  using base =
      InverseJacobian<db::const_item_type<MapTag>::dim,
                      typename db::const_item_type<MapTag>::source_frame,
                      typename db::const_item_type<MapTag>::target_frame>;
  using tag = MapTag;
  static constexpr auto function(
      const db::const_item_type<MapTag>& element_map,
      const db::const_item_type<SourceCoordsTag>& source_coords) noexcept {
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
  static constexpr size_t volume_dim = VolumeDim;
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
  static constexpr size_t volume_dim = VolumeDim;
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
  static constexpr size_t volume_dim = VolumeDim;
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
/// desired, then the tag should be added using `InterfaceCompute`. If a
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
/// without reference to a DataBox, i.e., `db::const_item_type<Tag>` must
/// work.
///
/// \tparam DirectionsTag the item of directions
/// \tparam Tag the tag labeling the item
///
/// \see InterfaceCompute, Slice
template <typename DirectionsTag, typename Tag>
struct Interface;

namespace Interface_detail {
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

// Virtual inheritance is used here to prevent a compiler warning: Derived class
// SimpleTag is inaccessible
template <typename DirectionsTag, typename Tag>
struct Interface : virtual db::SimpleTag,
                   Interface_detail::GetBaseTagIfPresent<DirectionsTag, Tag> {
  static std::string name() noexcept {
    return "Interface<" + DirectionsTag::name() + ", " + Tag::name() + ">";
  };
  using tag = Tag;
  // The use of db::const_item_type<Tag> assumes we will never store
  // unique_ptrs on interfaces.  This is probably a reasonable
  // assumption.
  using type = std::unordered_map<
      typename db::const_item_type<DirectionsTag>::value_type,
      db::const_item_type<Tag>>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup ComputationalDomainGroup
/// ::Direction to an interface
template <size_t VolumeDim>
struct Direction : db::SimpleTag {
  static std::string name() noexcept { return "Direction"; }
  using type = ::Direction<VolumeDim>;
};

}  // namespace Tags

namespace db {
namespace detail {
template <typename TagList, typename DirectionsTag, typename VariablesTag>
struct InterfaceSubitemsImpl {
  using type = tmpl::transform<
      typename const_item_type<VariablesTag>::tags_list,
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

}  // namespace db
