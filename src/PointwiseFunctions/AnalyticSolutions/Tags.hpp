// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"

namespace Tags {
/// Can be used to retrieve the analytic solution computer from the DataBox
/// without having to know the template parameters of AnalyticSolutionComputer.
struct AnalyticSolutionComputerBase : db::BaseTag {};

/// The analytic solution, with the type of the analytic solution set as the
/// template parameter
template <typename SolutionType>
struct AnalyticSolutionComputer : AnalyticSolutionComputerBase, db::SimpleTag {
  static std::string name() noexcept {
    return "AnalyticSolutionComputer(" +
           pretty_type::short_name<SolutionType>() + ")";
  }
  using type = SolutionType;
};
}  // namespace Tags

namespace OptionTags {
/// \ingroup OptionGroupsGroup
/// Holds the `OptionTags::AnalyticSolution` option in the input file
struct AnalyticSolutionGroup {
  static std::string name() noexcept { return "AnalyticSolution"; }
  static constexpr OptionString help =
      "Analytic solution used for the initial data and errors";
};

/// \ingroup OptionTagsGroup
/// Can be used to retrieve the analytic solution from the cache without having
/// to know the template parameters of AnalyticSolution.
struct AnalyticSolutionBase {};

/// \ingroup OptionTagsGroup
/// Base tag with which to retrieve the BoundaryConditionType
struct BoundaryConditionBase {};

/// \ingroup OptionTagsGroup
/// The analytic solution, with the type of the analytic solution set as the
/// template parameter
template <typename SolutionType>
struct AnalyticSolution : AnalyticSolutionBase {
  static std::string name() noexcept { return option_name<SolutionType>(); }
  static constexpr OptionString help = "Options for the analytic solution";
  using type = SolutionType;
  using group = AnalyticSolutionGroup;
  using container_tag = Tags::AnalyticSolutionComputer<SolutionType>;
};
/// \ingroup OptionTagsGroup
/// The boundary condition to be applied at all external boundaries.
template <typename BoundaryConditionType>
struct BoundaryCondition : BoundaryConditionBase {
  static constexpr OptionString help = "Boundary condition to be used";
  using type = BoundaryConditionType;
};
}  // namespace OptionTags
