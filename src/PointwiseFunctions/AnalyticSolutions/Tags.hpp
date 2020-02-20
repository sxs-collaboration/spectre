// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Options/Options.hpp"
#include "Parallel/Serialize.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"

namespace OptionTags {
/// \ingroup OptionGroupsGroup
/// Holds the `OptionTags::AnalyticSolution` option in the input file
struct AnalyticSolutionGroup {
  static std::string name() noexcept { return "AnalyticSolution"; }
  static constexpr OptionString help =
      "Analytic solution used for the initial data and errors";
};

/// \ingroup OptionTagsGroup
/// The analytic solution, with the type of the analytic solution set as the
/// template parameter
template <typename SolutionType>
struct AnalyticSolution {
  static std::string name() noexcept { return option_name<SolutionType>(); }
  static constexpr OptionString help = "Options for the analytic solution";
  using type = SolutionType;
  using group = AnalyticSolutionGroup;
};
/// \ingroup OptionTagsGroup
/// The boundary condition to be applied at all external boundaries.
template <typename BoundaryConditionType>
struct BoundaryCondition {
  static constexpr OptionString help = "Boundary condition to be used";
  using type = BoundaryConditionType;
};
}  // namespace OptionTags

namespace Tags {
/// Can be used to retrieve the analytic solution from the cache without having
/// to know the template parameters of AnalyticSolution.
struct AnalyticSolutionBase : AnalyticSolutionOrData {};

/// Base tag with which to retrieve the BoundaryConditionType
struct BoundaryConditionBase : db::BaseTag {};

/// \ingroup OptionTagsGroup
/// The analytic solution, with the type of the analytic solution set as the
/// template parameter
template <typename SolutionType>
struct AnalyticSolution : AnalyticSolutionBase, db::SimpleTag {
  using type = SolutionType;
  using option_tags = tmpl::list<::OptionTags::AnalyticSolution<SolutionType>>;

  template <typename Metavariables>
  static SolutionType create_from_options(
      const SolutionType& analytic_solution) noexcept {
    return deserialize<type>(serialize<type>(analytic_solution).data());
  }
};
/// \ingroup OptionTagsGroup
/// The boundary condition to be applied at all external boundaries.
template <typename BoundaryConditionType>
struct BoundaryCondition : BoundaryConditionBase, db::SimpleTag {
  using type = BoundaryConditionType;
  using option_tags =
      tmpl::list<::OptionTags::BoundaryCondition<BoundaryConditionType>>;

  template <typename Metavariables>
  static BoundaryConditionType create_from_options(
      const BoundaryConditionType& boundary_condition) noexcept {
    return boundary_condition;
  }
};
}  // namespace Tags
