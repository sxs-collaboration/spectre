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
#include "NumericalAlgorithms/LinearSolver/IterationId.hpp"
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
 * Uses:
 * - Metavariables:
 *   - `linear_solver`
 *   - `analytic_solution_tag`
 * - System:
 *   - `volume_dim`
 *   - `fields_tag`
 * - DataBox:
 *   - `Tags::Mesh<volume_dim>`
 *   - `Tags::Coordinates<volume_dim, Frame::Inertial>`
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
    const auto& inertial_coords =
        get<Tags::Coordinates<system::volume_dim, Frame::Inertial>>(box);
    const auto num_grid_points =
        get<Tags::Mesh<system::volume_dim>>(box).number_of_grid_points();

    db::item_type<sources_tag> sources(num_grid_points, 0.);
    sources.assign_subset(
        Parallel::get<typename Metavariables::analytic_solution_tag>(cache)
            .source_variables(inertial_coords));

    // Starting with x_0 = 0 initial guess, so Ax_0=0
    db::item_type<fields_operator_tag> fields_operator(num_grid_points, 0.);

    return linear_solver_tags::initialize(
        std::move(box), cache, array_index, parallel_component_meta,
        std::move(sources), std::move(fields_operator));
  }
};
}  // namespace Initialization
}  // namespace Elliptic
