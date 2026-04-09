// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Variables.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"

/// \cond
template <size_t VolumeDim>
class ElementId;
/// \endcond

namespace intrp {

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups option tags for InterpolationTargets.
 */
struct InterpolationTargets {
  static constexpr Options::String help{"Options for interpolation targets"};
};

/*!
 * \ingroup OptionGroupsGroup
 * \brief Groups option tags for the Interpolator.
 */
struct Interpolator {
  static constexpr Options::String help{
      "Options related to the Interpolator parallel component"};
};
}  // namespace OptionTags

/// Tags for items held in the `DataBox` of `InterpolationTarget` or
/// `Interpolator`.
namespace Tags {
/// Tag that determines the verbosity of output from the interpolation target
struct Verbosity : db::SimpleTag {
  using type = ::Verbosity;

  using option_tags =
      tmpl::list<logging::OptionTags::Verbosity<OptionTags::Interpolator>>;
  static constexpr bool pass_metavariables = false;
  static ::Verbosity create_from_options(const ::Verbosity& verbosity) {
    return verbosity;
  }
};

/// Keeps track of which points have been filled with interpolated data.
template <typename TemporalId>
struct IndicesOfFilledInterpPoints : db::SimpleTag {
  using type = std::unordered_map<TemporalId, std::unordered_set<size_t>>;
};

/// Keeps track of points that cannot be filled with interpolated data.
///
/// The InterpolationTarget can decide what to do with these points.
/// In most cases the correct action is to throw an error, but in other
/// cases one might wish to fill these points with a default value or
/// take some other action.
template <typename TemporalId>
struct IndicesOfInvalidInterpPoints : db::SimpleTag {
  using type = std::unordered_map<TemporalId, std::unordered_set<size_t>>;
};

/// `temporal_id`s on which to interpolate.
///
/// \note This tag is only used in non-sequential targets
template <typename TemporalId>
struct TemporalIds : db::SimpleTag {
  using type = std::unordered_set<TemporalId>;
};

/// `temporal_id`s that we have already interpolated onto.
///  This is used to prevent problems with multiple late calls.
template <typename TemporalId>
struct CompletedTemporalIds : db::SimpleTag {
  using type = std::deque<TemporalId>;
};

/// Holds interpolated variables on an InterpolationTarget.
template <typename InterpolationTargetTag, typename TemporalId>
struct InterpolatedVars : db::SimpleTag {
  using type = std::unordered_map<
      TemporalId,
      Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>;
};
}  // namespace Tags
}  // namespace intrp
