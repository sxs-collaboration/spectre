// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"

namespace OptionTags {
/// \ingroup OptionGroupsGroup
/// Holds the `OptionTags::AnalyticData` option in the input file
struct AnalyticDataGroup {
  static std::string name() noexcept { return "AnalyticData"; }
  static constexpr OptionString help =
      "Analytic data used for the initial data";
};

/// \ingroup OptionTagsGroup
/// Can be used to retrieve the analytic data from the cache without having
/// to know the template parameters of AnalyticData.
struct AnalyticDataBase {};

/// \ingroup OptionTagsGroup
/// The analytic data, with the type of the analytic data set as the template
/// parameter
template <typename SolutionType>
struct AnalyticData : AnalyticDataBase {
  static std::string name() noexcept { return option_name<SolutionType>(); }
  static constexpr OptionString help = "Options for the analytic data";
  using type = SolutionType;
  using group = AnalyticDataGroup;
};
}  // namespace OptionTags
