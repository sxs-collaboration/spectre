// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"

namespace CacheTags {
/// \ingroup CacheTagsGroup
/// Can be used to retrieve the analytic solution from the cache without having
/// to know the template parameters of AnalyticSolution.
struct AnalyticSolutionBase {};

/// \ingroup CacheTagsGroup
/// The analytic solution, with the type of the analytic solution set as the
/// template parameter
template <typename SolutionType>
struct AnalyticSolution : AnalyticSolutionBase {
  static constexpr OptionString help =
      "Analytic solution used for the initial data and errors";
  using type = SolutionType;
};
}  // namespace CacheTags
