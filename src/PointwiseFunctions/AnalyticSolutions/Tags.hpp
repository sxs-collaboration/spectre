// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Variables.hpp"
#include "Options/Options.hpp"
#include "Parallel/Serialize.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace OptionTags {
/// \ingroup OptionGroupsGroup
/// Holds the `OptionTags::AnalyticSolution` option in the input file
struct AnalyticSolutionGroup {
  static std::string name() { return "AnalyticSolution"; }
  static constexpr Options::String help =
      "Analytic solution used for the initial data and errors";
};

/// \ingroup OptionTagsGroup
/// The analytic solution, with the type of the analytic solution set as the
/// template parameter
template <typename SolutionType>
struct AnalyticSolution {
  static std::string name() { return Options::name<SolutionType>(); }
  static constexpr Options::String help = "Options for the analytic solution";
  using type = SolutionType;
  using group = AnalyticSolutionGroup;
};
}  // namespace OptionTags

namespace Tags {
/// Can be used to retrieve the analytic solution from the cache without having
/// to know the template parameters of AnalyticSolution.
struct AnalyticSolutionBase : AnalyticSolutionOrData {};

/// \ingroup OptionTagsGroup
/// The analytic solution, with the type of the analytic solution set as the
/// template parameter
template <typename SolutionType>
struct AnalyticSolution : AnalyticSolutionBase, db::SimpleTag {
  using type = SolutionType;
  using option_tags = tmpl::list<::OptionTags::AnalyticSolution<SolutionType>>;

  static constexpr bool pass_metavariables = false;
  static SolutionType create_from_options(
      const SolutionType& analytic_solution) {
    return deserialize<type>(serialize<type>(analytic_solution).data());
  }
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the analytic solution value for a quantity
///
/// \snippet AnalyticSolutions/Test_Tags.cpp analytic_name
template <typename Tag>
struct Analytic : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \brief Prefix indicating the error of a value represented by `Tag`
 */
template <typename Tag>
struct Error : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/// Base tag for the analytic solution tensors. Retrieved values can be either
/// `Variables` or `std::optional<Variables>`.
///
/// \see ::Tags::AnalyticSolutions
struct AnalyticSolutionsBase : db::BaseTag {};

/// The analytic solutions for all `FieldTags`
template <typename FieldTags>
struct AnalyticSolutions : AnalyticSolutionsBase, db::SimpleTag {
  using type = ::Variables<db::wrap_tags_in<Analytic, FieldTags>>;
};

/// The analytic solutions for all `FieldTags`, or `std::nullopt` if no analytic
/// solutions are available
template <typename FieldTags>
struct AnalyticSolutionsOptional : AnalyticSolutionsBase, db::SimpleTag {
  static std::string name() { return "AnalyticSolutions"; }
  using type =
      std::optional<::Variables<db::wrap_tags_in<Analytic, FieldTags>>>;
};
}  // namespace Tags
