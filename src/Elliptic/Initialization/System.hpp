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

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

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
 * - DataBox:
 *   - `Tags::Mesh<volume_dim>`
 *
 * DataBox:
 * - Adds:
 *   - `fields_tag`
 */
template <typename SystemType>
struct System {
  static constexpr size_t Dim = SystemType::volume_dim;
  using simple_tags = db::AddSimpleTags<typename SystemType::fields_tag>;
  using compute_tags = db::AddComputeTags<>;

  template <typename TagsList>
  static auto initialize(db::DataBox<TagsList>&& box) noexcept {
    using FieldsVars = typename SystemType::fields_tag::type;

    const size_t num_grid_points =
        db::get<Tags::Mesh<Dim>>(box).number_of_grid_points();

    // Set initial data to zero. Non-zero initial data would require the
    // linear solver initialization to also compute the Ax term.
    FieldsVars fields{num_grid_points, 0.};

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), std::move(fields));
  }
};

}  // namespace Initialization
}  // namespace Elliptic
