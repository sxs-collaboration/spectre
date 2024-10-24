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
#include "DataStructures/Variables.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/Interpolation/InterpolatedVars.hpp"
#include "Utilities/TaggedTuple.hpp"

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

/// Option tag that determines if volume data will be dumped from the
/// Interpolator upon a failure.
struct DumpVolumeDataOnFailure {
  using type = bool;
  static constexpr Options::String help{
      "Whether or not to dump all volume data currently stored by the "
      "interpolator. Volume data is written to the file corresponding to the "
      "node it was collected on."};
  using group = Interpolator;
};
}  // namespace OptionTags

/// Tags for items held in the `DataBox` of `InterpolationTarget` or
/// `Interpolator`.
namespace Tags {
/// Tag that determines if volume data will be dumped form the Interpolator upon
/// failure
struct DumpVolumeDataOnFailure : db::SimpleTag {
  using type = bool;
  using option_tags = tmpl::list<OptionTags::DumpVolumeDataOnFailure>;
  static constexpr bool pass_metavariables = false;

  static bool create_from_options(const bool input) { return input; }
};

/// Tag that determines the verbosity of output from the interpolator
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

/// `temporal_id`s that have been flagged to interpolate on, but that
/// have not yet been added to Tags::CurrentTemporalId.  A `temporal_id` is
/// pending if the `FunctionOfTime`s are not up to date for the time
/// associated with the `temporal_id`.
template <typename TemporalId>
struct PendingTemporalIds : db::SimpleTag {
  using type = std::deque<TemporalId>;
};

/// `temporal_id` on which to interpolate.
///
/// \note This tag is only used in sequential targets because only one temporal
/// id can be interpolated to at any given time
template <typename TemporalId>
struct CurrentTemporalId : db::SimpleTag {
  using type = std::optional<TemporalId>;
};

/// `temporal_id`s on which to interpolate.
///
/// \note This tag is only used in non-sequential targets
template <typename TemporalId>
struct TemporalIds : db::SimpleTag {
  using type = std::unordered_set<TemporalId>;
};

/// `temporal_id`s that we have already interpolated onto.
///  This is used to prevent problems with multiple late calls to
///  AddTemporalIdsToInterpolationTarget.
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

template <typename InterpolationTargetTag>
struct VarsToInterpolateToTarget {
  using type =
      Variables<typename InterpolationTargetTag::vars_to_interpolate_to_target>;
};

/// Volume variables at all `temporal_id`s for all local `Element`s.
/// Held by the Interpolator.
template <typename Metavariables, typename TemporalId>
struct VolumeVarsInfo : db::SimpleTag {
  struct Info {
    Mesh<Metavariables::volume_dim> mesh;
    // Variables that have been sent from the Elements.
    Variables<typename Metavariables::interpolator_source_vars>
        source_vars_from_element;
    // Variables, for each InterpolationTargetTag, that have been
    // computed in the volume before interpolation, and that will
    // be interpolated.
    tuples::tagged_tuple_from_typelist<
        db::wrap_tags_in<VarsToInterpolateToTarget,
                         typename Metavariables::interpolation_target_tags>>
        vars_to_interpolate;
    Info() = default;
    Info(
        Mesh<Metavariables::volume_dim> mesh_in,
        Variables<typename Metavariables::interpolator_source_vars>
            source_vars_from_element_in,
        tuples::tagged_tuple_from_typelist<
            db::wrap_tags_in<VarsToInterpolateToTarget,
                             typename Metavariables::interpolation_target_tags>>
            vars_to_interpolate_in)
        : mesh(std::move(mesh_in)),
          source_vars_from_element(std::move(source_vars_from_element_in)),
          vars_to_interpolate(std::move(vars_to_interpolate_in)) {}
    // NOLINTNEXTLINE(google-runtime-references)
    void pup(PUP::er& p) {
      p | mesh;
      p | source_vars_from_element;
      p | vars_to_interpolate;
    }
  };
  using type = std::unordered_map<
      typename TemporalId::type,
      std::unordered_map<ElementId<Metavariables::volume_dim>, Info>>;
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
};

/// Number of local `Element`s.
struct NumberOfElements : db::SimpleTag {
  using type = size_t;
};

}  // namespace Tags
}  // namespace intrp
