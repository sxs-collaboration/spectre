// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"

namespace Tags {
/// Can be used to retrieve the analytic data computer from the DataBox without
/// having to know the template parameters of AnalyticDataComputer.
struct AnalyticDataComputerBase {};

/// The analytic data computer, with the type of the analytic data set as the
/// template parameter
template <typename SolutionType>
struct AnalyticDataComputer : AnalyticDataComputerBase, db::SimpleTag {
  static std::string name() noexcept {
    return "AnalyticDataComputer(" + pretty_type::short_name<SolutionType>() +
           ")";
  }
  using type = SolutionType;
};
}  // namespace Tags

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
  using container_tag = Tags::AnalyticDataComputer<SolutionType>;
};
}  // namespace OptionTags
