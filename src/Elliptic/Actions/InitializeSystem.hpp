// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace Actions {

/*!
 * \brief Initializes the DataBox tags related to the elliptic system
 *
 * The system fields are initially set to zero.
 *
 * \note Currently the sources are always retrieved from an analytic solution.
 *
 * Uses:
 * - Metavariables:
 *   - `analytic_solution_tag`
 * - System:
 *   - `fields_tag`
 *   - `primal_fields`
 * - DataBox:
 *   - `Tags::Mesh<Dim>`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *
 * DataBox:
 * - Adds:
 *   - `fields_tag`
 *   - `db::add_tag_prefix<::Tags::FixedSource, fields_tag>`
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the `Initialization`
 * phase action list prior to this action.
 */
template <typename System>
struct InitializeSystem {
  using fields_tag = typename System::fields_tag;
  using fixed_sources_tag =
      db::add_tag_prefix<::Tags::FixedSource, fields_tag>;

  using simple_tags = tmpl::list<fields_tag, fixed_sources_tag>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ElementId<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using linear_operator_applied_to_fields_tag =
        db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;

    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const size_t num_grid_points = mesh.number_of_grid_points();
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);

    // Set initial data to zero (for now).
    typename fields_tag::type fields{num_grid_points, 0.};
    // Since the initial data is zero we don't need to apply the DG operator but
    // may just set it to zero as well. Once this condition is relaxed we will
    // have to add a communication step to the initialization that computes the
    // DG operator to the initial data.
    db::mutate<linear_operator_applied_to_fields_tag>(
        make_not_null(&box),
        [&num_grid_points](
            const gsl::not_null<
                typename linear_operator_applied_to_fields_tag::type*>
                linear_operator_applied_to_fields) noexcept {
          *linear_operator_applied_to_fields =
              typename linear_operator_applied_to_fields_tag::type{
                  num_grid_points, 0.};
        });

    // Retrieve the sources of the elliptic system from the analytic solution,
    // which defines the problem we want to solve.
    // We need only retrieve sources for the primal fields, since the auxiliary
    // fields will never be sourced.
    typename fixed_sources_tag::type fixed_sources{num_grid_points, 0.};
    fixed_sources.assign_subset(
        Parallel::get<typename Metavariables::analytic_solution_tag>(cache)
            .variables(inertial_coords,
                       db::wrap_tags_in<::Tags::FixedSource,
                                        typename System::primal_fields>{}));
    Initialization::mutate_assign<simple_tags>(
        make_not_null(&box), std::move(fields), std::move(fixed_sources));
    return std::make_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace elliptic
