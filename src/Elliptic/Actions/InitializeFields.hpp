// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Utilities/GetAnalyticData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic::Actions {

/*!
 * \brief Initialize the dynamic fields of the elliptic system, i.e. those we
 * solve for.
 *
 * Uses:
 * - System:
 *   - `primal_fields`
 * - DataBox:
 *   - `InitialGuessTag`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `primal_fields`
 */
template <typename System, typename InitialGuessTag>
struct InitializeFields {
 private:
  using fields_tag = ::Tags::Variables<typename System::primal_fields>;

 public:
  using simple_tags = tmpl::list<fields_tag>;
  using compute_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<InitialGuessTag>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto& initial_guess = db::get<InitialGuessTag>(box);
    auto initial_fields =
        elliptic::util::get_analytic_data<typename fields_tag::tags_list>(
            initial_guess, box, inertial_coords);
    ::Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                                 std::move(initial_fields));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace elliptic::Actions
