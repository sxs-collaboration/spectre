// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Parallel/Tags/Section.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// Options for AMR
namespace amr::OptionTags {

struct AmrGroup {
  static std::string name() { return "Amr"; }
  static constexpr Options::String help =
      "Options for adaptive mesh refinement (AMR)";
};

struct MaxCoarseLevels {
  using type = Options::Auto<size_t>;
  static constexpr Options::String help =
      "Maximum number of coarser AMR levels to keep. A value of '0' means that "
      "only the finest grid is kept, and 'Auto' means the number of levels "
      "is not restricted.";
  using group = AmrGroup;
};

}  // namespace amr::OptionTags

/// AMR tags
namespace amr::Tags {

/// Maximum number of AMR levels that will be kept. A value of '0' means that
/// only the finest grid is kept, and `std::nullopt` means the number of levels
/// is not restricted.
struct MaxCoarseLevels : db::SimpleTag {
  using type = std::optional<size_t>;
  static constexpr bool pass_metavariables = false;
  using option_tags = tmpl::list<OptionTags::MaxCoarseLevels>;
  static type create_from_options(const type value) { return value; };
};

/// All element IDs grouped by grid index are stored in this tag. The element
/// IDs are registered and deregistered during AMR.
template <size_t Dim>
struct AllElementIds : db::SimpleTag {
  using type = std::map<size_t, std::unordered_set<ElementId<Dim>>>;
};

/// The ID of the element that covers the same region or more on the coarser
/// (parent) grid. Only important if AMR is configured to keep coarse grids
/// around.
template <size_t Dim>
struct ParentId : db::SimpleTag {
  using type = std::optional<ElementId<Dim>>;
};

/// The IDs of the elements that cover the same region on the finer (child)
/// grid. Only important if AMR is configured to keep coarse grids around.
template <size_t Dim>
struct ChildIds : db::SimpleTag {
  using type = std::unordered_set<ElementId<Dim>>;
};

/// The mesh of the parent element. Needed for projections between grids.
/// Only important if AMR is configured to keep coarse grids around.
template <size_t Dim>
struct ParentMesh : db::SimpleTag {
  using type = std::optional<Mesh<Dim>>;
};

/// The AMR level of the element. This is used to tag a
/// `Parallel::Tags::Section` that contains all elements on the same grid.
/// Only important if AMR is configured to keep coarse grids around.
struct GridIndex {
  using type = size_t;
};

/// True on the finest AMR grid (the one with the highest grid index), false on
/// all other grids. This is used to tag a `Parallel::Tags::Section` that
/// contains all elements on the finest grid.
/// Only important if AMR is configured to keep coarse grids around.
struct IsFinestGrid {
  using type = bool;
};

/// An `observers::Tags::ObservationKey` that identifies the grid index.
/// Can be used to tag observations with the grid index.
template <size_t Dim>
struct GridIndexObservationKeyCompute
    : db::ComputeTag,
      observers::Tags::ObservationKey<GridIndex> {
  using base = observers::Tags::ObservationKey<GridIndex>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<domain::Tags::Element<Dim>, amr::Tags::ChildIds<Dim>>;
  static void function(
      const gsl::not_null<std::optional<std::string>*> observation_key,
      const Element<Dim>& element,
      const std::unordered_set<ElementId<Dim>>& child_ids) {
    const auto& element_id = element.id();
    const bool is_finest_grid = child_ids.empty();
    *observation_key =
        is_finest_grid
            ? std::string{""}
            : (std::string{"Level"} + std::to_string(element_id.grid_index()));
  }
};

/// An `observers::Tags::ObservationKey` that identifies the finest grid.
/// Can be used to observe things only on the finest grid.
template <size_t Dim>
struct IsFinestGridObservationKeyCompute
    : db::ComputeTag,
      observers::Tags::ObservationKey<IsFinestGrid> {
  using base = observers::Tags::ObservationKey<IsFinestGrid>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<amr::Tags::ChildIds<Dim>>;
  static void function(
      const gsl::not_null<std::optional<std::string>*> observation_key,
      const std::unordered_set<ElementId<Dim>>& child_ids) {
    const bool is_finest_grid = child_ids.empty();
    if (is_finest_grid) {
      *observation_key = std::string{""};
    } else {
      *observation_key = std::nullopt;
    }
  }
};

}  // namespace amr::Tags
