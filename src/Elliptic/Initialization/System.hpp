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

namespace Elliptic {
namespace Initialization {

/*!
 * \brief Initializes the DataBox tags related to the system
 *
 * The system fields are initially set to zero here.
 *
 * Uses:
 * - System:
 *   - `volume_dim`
 *   - `fields_tag`
 *   - `variables_tag`
 * - DataBox:
 *   - `Tags::Mesh<volume_dim>`
 *
 * DataBox:
 * - Adds:
 *   - `fields_tag`
 *   - `variables_tag`
 *   - `db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo,
 *   variables_tag>`
 */
template <typename SystemType>
struct System {
  static constexpr size_t Dim = SystemType::volume_dim;
  using simple_tags = db::AddSimpleTags<
      typename SystemType::fields_tag, typename SystemType::variables_tag,
      db::add_tag_prefix<::LinearSolver::Tags::OperatorAppliedTo,
                         typename SystemType::variables_tag>>;
  using compute_tags = db::AddComputeTags<>;

  template <typename TagsList>
  static auto initialize(db::DataBox<TagsList>&& box) noexcept {
    const size_t num_grid_points =
        db::get<Tags::Mesh<Dim>>(box).number_of_grid_points();

    // Set initial data to zero. Non-zero initial data would require the
    // linear solver initialization to also compute the Ax term.
    db::item_type<typename SystemType::fields_tag> fields{num_grid_points, 0.};

    // Initialize the variables for the elliptic solve. Their initial value is
    // determined by the linear solver. The value is also updated by the linear
    // solver in every step.
    db::item_type<typename SystemType::variables_tag> vars{num_grid_points};

    // Initialize the linear operator applied to the variables. It needs no
    // initial value, but is computed in every step of the elliptic solve.
    db::item_type<db::add_tag_prefix<::LinearSolver::Tags::OperatorAppliedTo,
                                     typename SystemType::variables_tag>>
        operator_applied_to_vars{num_grid_points};

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), std::move(fields), std::move(vars),
        std::move(operator_applied_to_vars));
  }
};

}  // namespace Initialization
}  // namespace Elliptic
