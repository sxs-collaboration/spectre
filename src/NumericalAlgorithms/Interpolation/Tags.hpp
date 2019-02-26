// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
template <size_t VolumeDim>
class ElementId;
/// \endcond

namespace intrp {

/// Tags for items held in the `DataBox` of `InterpolationTarget` or
/// `Interpolator`.
namespace Tags {

/// Keeps track of which points have been filled with interpolated data.
struct IndicesOfFilledInterpPoints : db::SimpleTag {
  static std::string name() noexcept { return "IndicesOfFilledInterpPoints"; }
  using type = std::unordered_set<size_t>;
};

/// `temporal_id`s on which to interpolate.
template <typename Metavariables>
struct TemporalIds : db::SimpleTag {
  using type = std::deque<typename Metavariables::temporal_id::type>;
  static std::string name() noexcept { return "TemporalIds"; }
};

/// `temporal_id`s that we have already interpolated onto.
///  This is used to prevent problems with multiple late calls to
///  AddTemporalIdsToInterpolationTarget.
template <typename Metavariables>
struct CompletedTemporalIds : db::SimpleTag {
  using type = std::deque<typename Metavariables::temporal_id::type>;
  static std::string name() noexcept { return "CompletedTemporalIds"; }
};

/// Volume variables at all `temporal_id`s for all local `Element`s.
template <typename Metavariables>
struct VolumeVarsInfo : db::SimpleTag {
  struct Info {
    Mesh<Metavariables::domain_dim> mesh;
    Variables<typename Metavariables::interpolator_source_vars> vars;
  };
  using type = std::unordered_map<
      typename Metavariables::temporal_id::type,
      std::unordered_map<ElementId<Metavariables::domain_dim>, Info>>;
  static std::string name() noexcept { return "VolumeVarsInfo"; }
};

namespace holders_detail {
template <typename InterpolationTargetTag, typename Metavariables>
using WrappedHolderTag = Vars::HolderTag<InterpolationTargetTag, Metavariables>;
}  // namespace holders_detail

/// `TaggedTuple` containing all local `Vars::Holder`s for
/// all `InterpolationTarget`s.
///
/// A particular `Vars::Holder` can be retrieved from this
/// `TaggedTuple` via a `Vars::HolderTag`.  An `Interpolator` uses the
/// object in `InterpolatedVarsHolders` to iterate over all of the
/// `InterpolationTarget`s.
template <typename Metavariables>
struct InterpolatedVarsHolders : db::SimpleTag {
  using type = tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
      holders_detail::WrappedHolderTag,
      typename Metavariables::interpolation_target_tags, Metavariables>>;
  static std::string name() noexcept { return "InterpolatedVarsHolders"; }
};

/// Number of local `Element`s.
struct NumberOfElements : db::SimpleTag {
  static std::string name() noexcept { return "NumberOfElements"; }
  using type = size_t;
};

}  // namespace Tags
}  // namespace intrp
