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
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace evolution::Tags {
/*!
 * \brief Computes the analytic solution and adds `::Tags::Analytic` of the
 * `std::optional<Tensor>`s to the DataBox.
 *
 * \note If `InitialDataList` is not an empty `tmpl::list`, then
 * `evolution::initial_data::Tags::InitialData` is retrieved and downcast to the
 * initial data for computing the errors.
 */
template <size_t Dim, typename AnalyticFieldsTagList, bool UsingDgSubcell,
          typename InitialDataList = tmpl::list<>>
struct AnalyticSolutionsCompute
    : ::Tags::AnalyticSolutions<AnalyticFieldsTagList>,
      db::ComputeTag {
  using field_tags = AnalyticFieldsTagList;
  using base = ::Tags::AnalyticSolutions<AnalyticFieldsTagList>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      tmpl::conditional_t<std::is_same_v<InitialDataList, tmpl::list<>>,
                          ::Tags::AnalyticSolutionOrData,
                          evolution::initial_data::Tags::InitialData>,
      tmpl::conditional_t<
          UsingDgSubcell,
          Events::Tags::ObserverCoordinates<Dim, Frame::Inertial>,
          domain::Tags::Coordinates<Dim, Frame::Inertial>>,
      ::Tags::Time>;

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

  static void function(
      const gsl::not_null<return_type*> analytic_solution,
      const evolution::initial_data::InitialData& initial_data,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
      const double time) {
    call_with_dynamic_type<void, InitialDataList>(
        &initial_data, [&analytic_solution, &inertial_coords,
                        time](const auto* const data_or_solution) {
          function(analytic_solution, *data_or_solution, inertial_coords, time);
        });
  }
};
}  // namespace evolution::Tags
