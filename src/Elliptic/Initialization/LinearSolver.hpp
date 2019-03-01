// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace Elliptic {
namespace Initialization {

/*!
 * \brief Initializes the DataBox tag related to the linear solver
 *
 * Uses the analytic solution to compute the linear solver source.
 * Assumes the system fields are initialized to zero (see
 * `Elliptic::Initialization::System`) so we don't have to apply the operator to
 * the initial guess.
 *
 * With:
 * - `sources_tag` = `db::add_tag_prefix<Tags::Source, system::fields_tag>`
 *
 * Uses:
 * - Metavariables:
 *   - `linear_solver`
 * - System:
 *   - `fields_tag`
 * - DataBox:
 *   - `sources_tag`
 *
 * DataBox:
 * - Adds:
 *   - All items in `linear_solver::tags`
 */
template <typename Metavariables>
struct LinearSolver {
  using linear_solver_tags = typename Metavariables::linear_solver::tags;
  using system = typename Metavariables::system;

  using sources_tag =
      db::add_tag_prefix<Tags::Source, typename system::fields_tag>;
  using fields_operator_tag =
      db::add_tag_prefix<::LinearSolver::Tags::OperatorAppliedTo,
                         typename system::fields_tag>;

  using simple_tags = typename linear_solver_tags::simple_tags;
  using compute_tags = typename linear_solver_tags::compute_tags;

  template <typename TagsList, typename ArrayIndex, typename ParallelComponent>
  static auto initialize(
      db::DataBox<TagsList>&& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index,
      const ParallelComponent* const parallel_component_meta) noexcept {
    const auto& sources = get<sources_tag>(box);

    // Starting with x_0 = 0 initial guess, so Ax_0=0
    auto fields_operator =
        make_with_value<db::item_type<fields_operator_tag>>(sources, 0.);

    return linear_solver_tags::initialize(std::move(box), cache, array_index,
                                          parallel_component_meta, sources,
                                          std::move(fields_operator));
  }
};
}  // namespace Initialization
}  // namespace Elliptic
