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
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Utilities/GetAnalyticData.hpp"
#include "Parallel/AlgorithmExecution.hpp"
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
template <typename BackgroundTag, typename AnalyticSolutionFields,
          typename AnalyticSolutionType>
struct InitializeOptionalAnalyticSolution {
 private:
  using analytic_fields_tag = ::Tags::AnalyticSolutions<AnalyticSolutionFields>;

 public:
  using simple_tags = tmpl::list<analytic_fields_tag>;
  using compute_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<BackgroundTag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto analytic_solution =
        dynamic_cast<const AnalyticSolutionType*>(&db::get<BackgroundTag>(box));
    if (analytic_solution != nullptr) {
      const auto& inertial_coords =
          get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
      auto analytic_fields =
          elliptic::util::get_analytic_data<AnalyticSolutionFields>(
              *analytic_solution, box, inertial_coords);
      Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                                 std::move(analytic_fields));
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
/// @}
}  // namespace elliptic::Actions
