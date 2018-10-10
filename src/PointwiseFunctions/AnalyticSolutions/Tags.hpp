// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"

namespace OptionTags {
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
  static constexpr OptionString help =
      "Analytic solution used for the initial data and errors";
  using type = SolutionType;
};
/// \ingroup OptionTagsGroup
/// The boundary condition to be applied at all external boundaries.
template <typename BoundaryConditionType>
struct BoundaryCondition : BoundaryConditionBase {
  static constexpr OptionString help = "Boundary condition to be used";
  using type = BoundaryConditionType;
};
}  // namespace OptionTags
