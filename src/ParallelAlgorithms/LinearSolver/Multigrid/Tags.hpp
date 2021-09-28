// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver::multigrid {

namespace OptionTags {

template <typename OptionsGroup>
struct MaxLevels {
  using type = Options::Auto<size_t>;
  static constexpr Options::String help =
      "Maximum number of levels in the multigrid hierarchy. Includes the "
      "finest grid, i.e. set to '1' to disable multigrids. Set to 'Auto' to "
      "coarsen all the way up to single-element blocks.";
  using group = OptionsGroup;
};

template <typename OptionsGroup>
struct OutputVolumeData {
  using type = bool;
  static constexpr Options::String help =
      "Record volume data for debugging purposes.";
  using group = OptionsGroup;
  static bool suggested_value() { return false; }
};

}  // namespace OptionTags

/// DataBox tags for the `LinearSolver::multigrid::Multigrid` linear solver
namespace Tags {

/// Initial refinement of the next-finer (child) grid
template <size_t Dim>
struct ChildrenRefinementLevels : db::SimpleTag {
 private:
  using base = domain::Tags::InitialRefinementLevels<Dim>;

 public:
  using type = typename base::type;
  static constexpr bool pass_metavariables = base::pass_metavariables;
  using option_tags = typename base::option_tags;
  static constexpr auto create_from_options = base::create_from_options;
};

/// Initial refinement of the next-coarser (parent) grid
template <size_t Dim>
struct ParentRefinementLevels : db::SimpleTag {
 private:
  using base = domain::Tags::InitialRefinementLevels<Dim>;

 public:
  using type = typename base::type;
  static constexpr bool pass_metavariables = base::pass_metavariables;
  using option_tags = typename base::option_tags;
  static constexpr auto create_from_options = base::create_from_options;
};

/// Maximum number of multigrid levels that will be created. A value of '1'
/// effectively disables the multigrid, and `std::nullopt` means the number
/// of multigrid levels is not capped.
template <typename OptionsGroup>
struct MaxLevels : db::SimpleTag {
  using type = std::optional<size_t>;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::MaxLevels<OptionsGroup>>;
  static type create_from_options(const type value) { return value; };
  static std::string name() {
    return "MaxLevels(" + Options::name<OptionsGroup>() + ")";
  }
};

/// Whether or not volume data should be recorded for debugging purposes
template <typename OptionsGroup>
struct OutputVolumeData : db::SimpleTag {
  using type = bool;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::OutputVolumeData<OptionsGroup>>;
  static type create_from_options(const type value) { return value; };
  static std::string name() {
    return "OutputVolumeData(" + Options::name<OptionsGroup>() + ")";
  }
};

/// The multigrid level. The finest grid is always level 0 and the coarsest grid
/// has the highest level.
struct MultigridLevel : db::SimpleTag {
  using type = size_t;
};

/// Indicates the root of the multigrid hierarchy, i.e. level 0.
struct IsFinestGrid : db::SimpleTag {
  using type = bool;
};

/// The ID of the element that covers the same region or more on the coarser
/// (parent) grid
template <size_t Dim>
struct ParentId : db::SimpleTag {
  using type = std::optional<ElementId<Dim>>;
};

/// The IDs of the elements that cover the same region on the finer (child) grid
template <size_t Dim>
struct ChildIds : db::SimpleTag {
  using type = std::unordered_set<ElementId<Dim>>;
};

/// The mesh of the parent element. Needed for projections between grids.
template <size_t Dim>
struct ParentMesh : db::SimpleTag {
  using type = std::optional<Mesh<Dim>>;
};

// The following tags are related to volume data output

/// Continuously incrementing ID for volume observations
template <typename OptionsGroup>
struct ObservationId : db::SimpleTag {
  using type = size_t;
  static std::string name() {
    return "ObservationId(" + Options::name<OptionsGroup>() + ")";
  }
};
/// @{
/// Prefix tag for recording volume data in
/// `LinearSolver::multigrid::Tags::VolumeDataForOutput`
template <typename Tag>
struct PreSmoothingInitial : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename Tag>
struct PreSmoothingSource : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename Tag>
struct PreSmoothingResult : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename Tag>
struct PreSmoothingResidual : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename Tag>
struct PostSmoothingInitial : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename Tag>
struct PostSmoothingSource : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename Tag>
struct PostSmoothingResult : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
template <typename Tag>
struct PostSmoothingResidual : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
/// @}
/// Buffer for recording volume data
template <typename OptionsGroup, typename FieldsTag>
struct VolumeDataForOutput : db::SimpleTag {
  using fields_tags = typename FieldsTag::type::tags_list;
  using type = Variables<
      tmpl::append<db::wrap_tags_in<PreSmoothingInitial, fields_tags>,
                   db::wrap_tags_in<PreSmoothingSource, fields_tags>,
                   db::wrap_tags_in<PreSmoothingResult, fields_tags>,
                   db::wrap_tags_in<PreSmoothingResidual, fields_tags>,
                   db::wrap_tags_in<PostSmoothingInitial, fields_tags>,
                   db::wrap_tags_in<PostSmoothingSource, fields_tags>,
                   db::wrap_tags_in<PostSmoothingResult, fields_tags>,
                   db::wrap_tags_in<PostSmoothingResidual, fields_tags>>>;
  static std::string name() {
    return "VolumeDataForOutput(" + Options::name<OptionsGroup>() + ")";
  }
};

}  // namespace Tags
}  // namespace LinearSolver::multigrid
