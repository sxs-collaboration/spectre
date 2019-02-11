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
struct get_reduction_data_tags {
  using type = tmpl::list<>;
};

template <class ObservingAction>
struct get_reduction_data_tags<
    ObservingAction,
    cpp17::void_t<typename ObservingAction::observed_reduction_data_tags>> {
  using type = typename ObservingAction::observed_reduction_data_tags;
};

template <class ReductionDataType>
struct make_reduction_data_tag_impl {
  using type = tmpl::wrap<typename ReductionDataType::datum_list,
                          ::observers::Tags::ReductionData>;
};
}  // namespace detail

/// Each Action that sends data to the reduction Observer must specify
/// a type alias `observed_reduction_data_tags` that describes the data it
/// sends.  Given a list of such Actions (or other types that expose the alias),
/// this metafunction is used to create
/// `Metavariables::observed_reduction_data_tags` (which is required to
/// initialize the Observer).
template <class ObservingActionList>
using collect_reduction_data_tags =
    tmpl::remove_duplicates<tmpl::flatten<tmpl::transform<
        ObservingActionList, detail::get_reduction_data_tags<tmpl::_1>>>>;

/// Produces the `tmpl::list` of `observers::Tags::ReductionData` tags that
/// corresponds to the `tmpl::list` of `Parallel::ReductionData` passed into
/// this metafunction.
template <typename ReductionDataList>
using make_reduction_data_tags = tmpl::remove_duplicates<tmpl::transform<
    ReductionDataList, detail::make_reduction_data_tag_impl<tmpl::_1>>>;
}  // namespace observers
