// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace elliptic::Actions {

/// @{
/*!
 * \brief Place the analytic solution of the system fields in the DataBox.
 *
 * The `::Tags::AnalyticSolutionsBase` tag retrieved from the DataBox will hold
 * a `std::optional`. The analytic solution is only evaluated and stored in the
 * DataBox if the `BackgroundTag` holds a type that inherits from the
 * `AnalyticSolutionType`.
 *
 * Uses:
 * - DataBox:
 *   - `AnalyticSolutionTag` or `BackgroundTag`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `::Tags::AnalyticSolutionsBase`
 */
template <size_t Dim, typename BackgroundTag, typename AnalyticSolutionFields,
          typename AnalyticSolutionType>
struct InitializeOptionalAnalyticSolution
    : tt::ConformsTo<::amr::protocols::Projector> {
 private:
  using analytic_fields_tag = ::Tags::AnalyticSolutions<AnalyticSolutionFields>;

 public:  // Iterable action
  using simple_tags = tmpl::list<analytic_fields_tag>;
  using compute_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<BackgroundTag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate_apply<InitializeOptionalAnalyticSolution>(make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

 public:  // DataBox mutator, amr::protocols::Projector
  using return_tags = tmpl::list<analytic_fields_tag>;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim>,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>, BackgroundTag,
                 Parallel::Tags::Metavariables>;

  template <typename Background, typename Metavariables, typename... AmrData>
  static void apply(const gsl::not_null<typename analytic_fields_tag::type*>
                        analytic_solution_fields,
                    const Mesh<Dim>& mesh,
                    const tnsr::I<DataVector, Dim> inertial_coords,
                    const Background& background, const Metavariables& /*meta*/,
                    const AmrData&... amr_data) {
    if constexpr (sizeof...(AmrData) == 1) {
      if constexpr (std::is_same_v<AmrData...,
                                   std::pair<Mesh<Dim>, Element<Dim>>>) {
        if (((mesh == amr_data.first) and ...)) {
          // This element hasn't changed during AMR. Nothing to do.
          return;
        }
      }
    }

    const auto analytic_solution =
        dynamic_cast<const AnalyticSolutionType*>(&background);
    if (analytic_solution != nullptr) {
      using factory_classes = typename std::decay_t<
          Metavariables>::factory_creation::factory_classes;
      *analytic_solution_fields = call_with_dynamic_type<
          Variables<AnalyticSolutionFields>,
          tmpl::at<factory_classes, AnalyticSolutionType>>(
          analytic_solution, [&inertial_coords](const auto* const derived) {
            return variables_from_tagged_tuple(
                derived->variables(inertial_coords, AnalyticSolutionFields{}));
          });
    } else {
      *analytic_solution_fields = std::nullopt;
    }
  }
};
/// @}
}  // namespace elliptic::Actions
