// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::Tags {
/*!
 * \brief Computes the analytic solution and adds `::Tags::Analytic` of the
 * `std::optional<Tensor>`s to the DataBox.
 */
template <size_t Dim, typename AnalyticFieldsTagList>
struct AnalyticSolutionsCompute
    : ::Tags::AnalyticSolutions<AnalyticFieldsTagList>,
      db::ComputeTag {
  using field_tags = AnalyticFieldsTagList;
  using base = ::Tags::AnalyticSolutions<AnalyticFieldsTagList>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<::Tags::AnalyticSolutionOrData,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>, ::Tags::Time>;
  template <typename AnalyticSolution>
  static void function(
      const gsl::not_null<return_type*> analytic_solution,
      const AnalyticSolution& analytic_solution_computer,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
      const double time) {
    if constexpr (::is_analytic_solution_v<AnalyticSolution>) {
      *analytic_solution =
          variables_from_tagged_tuple(analytic_solution_computer.variables(
              inertial_coords, time, AnalyticFieldsTagList{}));
    } else {
      (void)analytic_solution_computer;
      (void)inertial_coords;
      (void)time;
      *analytic_solution = std::nullopt;
    }
  }
};
}  // namespace evolution::Tags
