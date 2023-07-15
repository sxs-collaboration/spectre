// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/History.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// Tag for the TimeStepper history
///
/// Leaving the template parameter unspecified gives a base tag.
///
/// \tparam Tag tag for the variables
template <typename Tag = void>
struct HistoryEvolvedVariables;

/// \cond
template <>
struct HistoryEvolvedVariables<> : db::BaseTag {};

template <typename Tag>
struct HistoryEvolvedVariables : HistoryEvolvedVariables<>, db::SimpleTag {
  using type = TimeSteppers::History<typename Tag::type>;
};
/// \endcond

/// \ingroup TimeGroup
/// From a list of tags `TagList`, extract all tags that are template
/// specializations of `HistoryEvolvedVariables`.
template <typename TagList>
using get_all_history_tags =
    tmpl::filter<TagList, tt::is_a<::Tags::HistoryEvolvedVariables, tmpl::_1>>;
}  // namespace Tags
