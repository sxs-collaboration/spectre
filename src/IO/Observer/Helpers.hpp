// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "IO/Observer/Tags.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace observers {
namespace detail {
template <class ObservingAction, class = cpp17::void_t<>>
struct get_reduction_data_tags_from_observing_action {
  using type = tmpl::list<>;
};

template <class ObservingAction>
struct get_reduction_data_tags_from_observing_action<
    ObservingAction,
    cpp17::void_t<typename ObservingAction::reduction_data_tags>> {
  using type = typename ObservingAction::reduction_data_tags;
};
}  // namespace detail

/// Each Action that sends data to the reduction Observer must specify
/// a type alias `reduction_data_tags` that describes the data it
/// sends.  Given a list of such Actions, this metafunction is used to
/// create `Metavariables::reduction_data_tags` (which is required to
/// initialize the Observer).
template <class ObservingActionList>
using get_reduction_data_tags_from_observing_actions =
    tmpl::remove_duplicates<tmpl::transform<
        ObservingActionList,
        detail::get_reduction_data_tags_from_observing_action<tmpl::_1>>>;

/// Given a tmpl::list of ReductionDatums, makes an
/// observers::Tags::ReductionData.
template <typename ReductionDatumList>
using make_reduction_data_tags_t =
    tmpl::wrap<ReductionDatumList, ::observers::Tags::ReductionData>;

/// Given a tmpl::list of ReductionDatums, makes a
/// Parallel::ReductionData.
template <typename ReductionDatumList>
using make_reduction_data_t =
    tmpl::wrap<ReductionDatumList, ::Parallel::ReductionData>;
}  // namespace observers
