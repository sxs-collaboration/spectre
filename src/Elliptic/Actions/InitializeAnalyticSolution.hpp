// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace elliptic::Actions {

// @{
/*!
 * \brief Place the analytic solution of the system fields in the DataBox.
 *
 * Use `InitializeAnalyticSolution` if it is clear at compile-time that an
 * analytic solution is available, and `InitializeOptionalAnalyticSolution` if
 * that is a runtime decision, e.g. based on a choice in the input file. The
 * `::Tags::AnalyticSolutionsBase` tag can be retrieved from the DataBox in
 * either case, but it will hold a `std::optional` when
 * `InitializeOptionalAnalyticSolution` is used. In that case, the analytic
 * solution is only evaluated and stored in the DataBox if the `BackgroundTag`
 * holds a type that inherits from the `AnalyticSolutionType`.
 *
 * Uses:
 * - DataBox:
 *   - `AnalyticSolutionTag` or `BackgroundTag`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `::Tags::AnalyticSolutionsBase`
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename AnalyticSolutionTag, typename AnalyticSolutionFields>
struct InitializeAnalyticSolution {
 private:
  using analytic_fields_tag = ::Tags::AnalyticSolutions<AnalyticSolutionFields>;

 public:
  using simple_tags = tmpl::list<analytic_fields_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto& analytic_solution = get<AnalyticSolutionTag>(box);
    auto analytic_fields = variables_from_tagged_tuple(
        analytic_solution.variables(inertial_coords, AnalyticSolutionFields{}));
    Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                               std::move(analytic_fields));
    return {std::move(box)};
  }
};

template <typename BackgroundTag, typename AnalyticSolutionFields,
          typename AnalyticSolutionType>
struct InitializeOptionalAnalyticSolution {
 private:
  using analytic_fields_tag =
      ::Tags::AnalyticSolutionsOptional<AnalyticSolutionFields>;

 public:
  using simple_tags = tmpl::list<analytic_fields_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto analytic_solution =
        dynamic_cast<const AnalyticSolutionType*>(&db::get<BackgroundTag>(box));
    if (analytic_solution != nullptr) {
      const auto& inertial_coords =
          get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
      auto analytic_fields =
          variables_from_tagged_tuple(analytic_solution->variables(
              inertial_coords, AnalyticSolutionFields{}));
      Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                                 std::move(analytic_fields));
    }
    return {std::move(box)};
  }
};
// @}
}  // namespace elliptic::Actions
