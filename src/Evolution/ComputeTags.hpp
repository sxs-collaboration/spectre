// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution {
namespace Tags {
/*!
 * \brief Use the `AnalyticSolutionTag` to compute the analytic solution of the
 * tags in `AnalyticFieldsTagList`.
 */
template <size_t Dim, typename AnalyticSolutionTag,
          typename AnalyticFieldsTagList>
struct AnalyticCompute : ::Tags::AnalyticSolutions<AnalyticFieldsTagList>,
                         db::ComputeTag {
  using base = ::Tags::AnalyticSolutions<AnalyticFieldsTagList>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<AnalyticSolutionTag,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>, ::Tags::Time>;
  static void function(
      const gsl::not_null<return_type*> analytic_solution,
      const typename AnalyticSolutionTag::type& analytic_solution_computer,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
      const double time) {
    *analytic_solution =
        variables_from_tagged_tuple(analytic_solution_computer.variables(
            inertial_coords, time, AnalyticFieldsTagList{}));
  }
};

/// @{
/*!
 * \brief For each `Tag` in `TagsList`, compute its difference from the
 * analytic solution.
 */
template <size_t VolumeDim, typename AnalyticSolutionTag, typename TagsList>
struct ErrorsCompute
    : db::add_tag_prefix<::Tags::Error, ::Tags::Variables<TagsList>>,
      db::ComputeTag {
  using base = db::add_tag_prefix<::Tags::Error, ::Tags::Variables<TagsList>>;
  using return_type = tmpl::type_from<base>;

  using argument_tags = tmpl::append<
      tmpl::list<AnalyticSolutionTag,
                 domain::Tags::Coordinates<VolumeDim, Frame::Inertial>,
                 ::Tags::Time>,
      TagsList>;

  template <typename AnalyticSolution, typename... ErrorTags,
            typename... FieldTypes>
  static constexpr void function(
      const gsl::not_null<Variables<tmpl::list<ErrorTags...>>*> errors,
      const AnalyticSolution& analytic_solution_computer,
      const tnsr::I<DataVector, VolumeDim, Frame::Inertial>& inertial_coords,
      const double time, const FieldTypes&... fields) {
    *errors = return_type{get<0>(inertial_coords).size()};
    const auto helper = [](const auto error, const auto& field) {
      for (size_t i = 0; i < field.size(); ++i) {
        (*error)[i] = field[i];
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(
        helper(make_not_null(&get<ErrorTags>(*errors)), fields));

    const auto analytic =
        variables_from_tagged_tuple(analytic_solution_computer.variables(
            inertial_coords, time, TagsList{}));
    *errors -= analytic;
  }
};
/// @}
}  // namespace Tags
}  // namespace evolution
