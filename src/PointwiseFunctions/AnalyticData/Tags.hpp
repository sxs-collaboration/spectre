// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"

namespace OptionTags {
/// \ingroup OptionTagsGroup
/// Can be used to retrieve the analytic data from the cache without having
/// to know the template parameters of AnalyticData.
struct AnalyticDataBase {};

/// \ingroup OptionTagsGroup
/// The analytic data, with the type of the analytic data set as the template
/// parameter
template <typename SolutionType>
struct AnalyticData : AnalyticDataBase {
  static constexpr OptionString help =
      "Analytic data used for the initial data";
  using type = SolutionType;
};
}  // namespace OptionTags
