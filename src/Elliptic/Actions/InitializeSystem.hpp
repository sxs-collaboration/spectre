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
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace Actions {

/*!
 * \brief Initializes the DataBox tags related to the elliptic system
 *
 * The system fields are initially set to zero. The linear solver operand and
 * the linear operator applied to it are also added, but initialized to
 * undefined values.
 *
 * \note Currently the sources are always retrieved from an analytic solution.
 *
 * Uses:
 * - Metavariables:
 *   - `analytic_solution_tag`
 * - System:
 *   - `volume_dim`
 *   - `fields_tag`
 *   - `gradient_tags`
 * - DataBox:
 *   - `Tags::Mesh<volume_dim>`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *   - All items required by the added compute tags
 *
 * DataBox:
 * - Adds:
 *   - `fields_tag`
 *   - `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>`
 *   - `db::add_tag_prefix<::Tags::FixedSource, fields_tag>`
 *   - `db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>`
 *   - `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
 *   db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>>`
 *   - `Tags::deriv<variables_tag, volume_dim, Frame::Inertial>>`
 */
struct InitializeSystem {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using fields_tag = typename system::fields_tag;
    using linear_operator_applied_to_fields_tag =
        db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
    using fixed_sources_tag =
        db::add_tag_prefix<::Tags::FixedSource, fields_tag>;
    using linear_operand_tag =
        db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
    using linear_operator_applied_to_operand_tag =
        db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
                           linear_operand_tag>;
    using inv_jacobian_tag =
        ::Tags::InverseJacobian<::Tags::ElementMap<Dim>,
                                ::Tags::Coordinates<Dim, Frame::Logical>>;

    using simple_tags =
        db::AddSimpleTags<fields_tag, linear_operator_applied_to_fields_tag,
                          fixed_sources_tag, linear_operand_tag,
                          linear_operator_applied_to_operand_tag>;
    using compute_tags = db::AddComputeTags<
        // The gradients are needed by the elliptic operator
        ::Tags::DerivCompute<linear_operand_tag, inv_jacobian_tag,
                             typename system::gradient_tags>>;

    const auto& mesh = db::get<::Tags::Mesh<Dim>>(box);
    const size_t num_grid_points = mesh.number_of_grid_points();
    const auto& inertial_coords =
        get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);

    // Set initial data to zero. Non-zero initial data would require us to also
    // compute the linear operator applied to the the initial data.
    db::item_type<fields_tag> fields{num_grid_points, 0.};
    db::item_type<linear_operator_applied_to_fields_tag>
        linear_operator_applied_to_fields{num_grid_points, 0.};

    // Retrieve the sources of the elliptic system from the analytic solution,
    // which defines the problem we want to solve.
    // We need only retrieve sources for the primal fields, since the auxiliary
    // fields will never be sourced.
    db::item_type<fixed_sources_tag> fixed_sources{num_grid_points, 0.};
    fixed_sources.assign_subset(
        Parallel::get<typename Metavariables::analytic_solution_tag>(cache)
            .variables(inertial_coords,
                       db::wrap_tags_in<::Tags::FixedSource,
                                        typename system::primal_fields>{}));

    // Initialize the variables for the elliptic solve. Their initial value is
    // determined by the linear solver. The value is also updated by the linear
    // solver in every step.
    db::item_type<linear_operand_tag> linear_operand{num_grid_points};

    // Initialize the linear operator applied to the variables. It needs no
    // initial value, but is computed in every step of the elliptic solve.
    db::item_type<linear_operator_applied_to_operand_tag>
        linear_operator_applied_to_operand{num_grid_points};

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeSystem, simple_tags,
                                             compute_tags>(
            std::move(box), std::move(fields),
            std::move(linear_operator_applied_to_fields),
            std::move(fixed_sources), std::move(linear_operand),
            std::move(linear_operator_applied_to_operand)));
  }
};

}  // namespace Actions
}  // namespace elliptic
