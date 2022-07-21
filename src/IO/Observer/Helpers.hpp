// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "IO/Observer/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/Tags/InputSource.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace observers {
namespace detail {
template <class ObservingAction, class = std::void_t<>>
struct get_reduction_data_tags {
  using type = tmpl::list<>;
};

template <class ObservingAction>
struct get_reduction_data_tags<
    ObservingAction,
    std::void_t<typename ObservingAction::observed_reduction_data_tags>> {
  using type = typename ObservingAction::observed_reduction_data_tags;
};

template <class ReductionDataType>
struct make_reduction_data_tag_impl {
  using type = tmpl::wrap<typename ReductionDataType::datum_list,
                          ::observers::Tags::ReductionData>;
};
}  // namespace detail

/// Function that returns from the global cache a string containing the
/// options provided in the yaml-formatted input file, if those options are
/// in the global cache. Otherwise, returns an empty string.
template <typename Metavariables>
std::string input_source_from_cache(
    const Parallel::GlobalCache<Metavariables>& cache) {
  if constexpr (tmpl::list_contains_v<
                    ::Parallel::get_const_global_cache_tags<Metavariables>,
                    ::Parallel::Tags::InputSource>) {
    const std::vector<std::string> input_source_vector{
        Parallel::get<::Parallel::Tags::InputSource>(cache)};
    std::string input_source{};
    for (auto it = input_source_vector.begin(); it != input_source_vector.end();
         ++it) {
      input_source += *it;
    }
    return input_source;
  } else {
    return ""s;
  }
}

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
