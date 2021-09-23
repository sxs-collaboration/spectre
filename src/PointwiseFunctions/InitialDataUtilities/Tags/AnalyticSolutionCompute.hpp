// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/TypeTraits.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace db {
struct ComputeTag;
}  // namespace db
namespace Frame {
struct Inertial;
}
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace InitialDataUtilities::Tags {
/*!
 * \brief A compute tag for (optional) analytic solution fields
 *
 * This compute tag returns the solution of field variables over time if
 * the initial data of an evolution system is set to be an analytic solution.
 * Otherwise the solution is not computed and `std::nullopt` is returned.
 */
template <size_t Dim, typename FieldTagsToAnalyticCompute>
struct AnalyticSolutionOptionalCompute
    : ::Tags::AnalyticSolutionsOptional<FieldTagsToAnalyticCompute>,
      db::ComputeTag {
  using base = ::Tags::AnalyticSolutionsOptional<FieldTagsToAnalyticCompute>;
  using argument_tags =
      tmpl::list<Parallel::Tags::Metavariables, InitialDataBase,
                 ::domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 ::Tags::Time>;

  template <typename Metavariables, typename InitialDataType>
  static void function(
      const gsl::not_null<std::optional<::Variables<
          db::wrap_tags_in<::Tags::Analytic, FieldTagsToAnalyticCompute>>>*>
          analytic_solution_computed,
      const Metavariables& /*meta*/, const InitialDataType& initial_data,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
      const double time) noexcept {
    using derived_classes =
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 InitialDataType>;

    call_with_dynamic_type<void, derived_classes>(
        &initial_data, [&analytic_solution_computed, &inertial_coords,
                        &time](auto* const initial_data_derived) noexcept {
          using derived = std::decay_t<decltype(*initial_data_derived)>;

          static_assert(evolution::is_analytic_data_v<derived> xor
                            evolution::is_analytic_solution_v<derived>,
                        "initial_data must be either an analytic_data or an "
                        "analytic_solution");

          if constexpr (evolution::is_analytic_data_v<derived>) {
            *analytic_solution_computed = std::nullopt;
          } else if constexpr (evolution::is_analytic_solution_v<derived>) {
            *analytic_solution_computed =
                variables_from_tagged_tuple(initial_data_derived->variables(
                    inertial_coords, time, FieldTagsToAnalyticCompute{}));
          }
        });
  }
};
}  // namespace InitialDataUtilities::Tags
