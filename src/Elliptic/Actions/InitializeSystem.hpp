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
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/TMPL.hpp"

namespace elliptic {
namespace Actions {

/*!
 * \brief Initializes the DataBox tags related to the system
 *
 * The system fields are initially set to zero here.
 *
 * Uses:
 * - Metavariables:
 *   - `analytic_solution_tag`
 * - System:
 *   - `volume_dim`
 *   - `fields_tag`
 *   - `variables_tag`
 *   - `gradient_tags`
 * - DataBox:
 *   - `Tags::Mesh<volume_dim>`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *   - All items required by the added compute tags
 *
 * DataBox:
 * - Adds:
 *   - `fields_tag`
 *   - `db::add_tag_prefix<::LinearSolver::Tags::OperatorAppliedTo, fields_tag>`
 *   - `db::add_tag_prefix<::Tags::Source, fields_tag>`
 *   - `variables_tag`
 *   - `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
 *   variables_tag>`
 *   - `Tags::deriv<variables_tag, volume_dim, Frame::Inertial>>`
 */
struct InitializeSystem {
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using fields_tag = typename system::fields_tag;
    using operator_applied_to_fields_tag =
        db::add_tag_prefix<::LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
    using sources_tag = db::add_tag_prefix<::Tags::Source, fields_tag>;
    using vars_tag = typename system::variables_tag;
    using operator_applied_to_vars_tag =
        db::add_tag_prefix<::LinearSolver::Tags::OperatorAppliedTo, vars_tag>;
    using inv_jacobian_tag =
        ::Tags::InverseJacobian<::Tags::ElementMap<Dim>,
                                ::Tags::Coordinates<Dim, Frame::Logical>>;

    using simple_tags =
        db::AddSimpleTags<fields_tag, operator_applied_to_fields_tag,
                          sources_tag, vars_tag, operator_applied_to_vars_tag>;
    using compute_tags = db::AddComputeTags<
        // The gradients are needed by the elliptic operator
        ::Tags::DerivCompute<vars_tag, inv_jacobian_tag,
                             typename system::gradient_tags>>;

    const auto& mesh = db::get<Tags::Mesh<Dim>>(box);
    const size_t num_grid_points = mesh.number_of_grid_points();
    const auto& inertial_coords =
        get<::Tags::Coordinates<Dim, Frame::Inertial>>(box);

    // Set initial data to zero. Non-zero initial data would require us to also
    // compute the linear operator applied to the the initial data.
    db::item_type<fields_tag> fields{num_grid_points, 0.};
    db::item_type<operator_applied_to_fields_tag> operator_applied_to_fields{
        num_grid_points, 0.};

    db::item_type<sources_tag> sources(num_grid_points, 0.);
    // This actually sets the complete set of tags in the Variables, but there
    // is no Variables constructor from a TaggedTuple (yet)
    sources.assign_subset(
        Parallel::get<typename Metavariables::analytic_solution_tag>(cache)
            .variables(inertial_coords,
                       db::get_variables_tags_list<sources_tag>{}));

    // Initialize the variables for the elliptic solve. Their initial value is
    // determined by the linear solver. The value is also updated by the linear
    // solver in every step.
    db::item_type<vars_tag> vars{num_grid_points};

    // Initialize the linear operator applied to the variables. It needs no
    // initial value, but is computed in every step of the elliptic solve.
    db::item_type<operator_applied_to_vars_tag> operator_applied_to_vars{
        num_grid_points};

    return std::make_tuple(
        Initialization::merge_into_databox<InitializeSystem, simple_tags,
                                           compute_tags>(
            std::move(box), std::move(fields),
            std::move(operator_applied_to_fields), std::move(sources),
            std::move(vars), std::move(operator_applied_to_vars)));
  }
};

}  // namespace Actions
}  // namespace elliptic
