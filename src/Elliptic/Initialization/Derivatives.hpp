// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
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
 * \brief Initializes the DataBox tags related to derivatives of the system
 * variables
 *
 * We have to initialize these separately from
 * `Elliptic::Initialization::System` since the `variables_tag` is the linear
 * solver operand, which is added by the linear solver initialization.
 *
 * With:
 * - `inv_jac_tag` = `Tags::InverseJacobian<
 * Tags::ElementMap<Dim>, Tags::Coordinates<Dim, Frame::Logical>>`
 *
 * Uses:
 * - System:
 *   - `volume_dim`
 *   - `variables_tag`
 *   - `gradient_tags`
 *
 * DataBox:
 * - Adds:
 *   - `Tags::DerivCompute<variables_tag, inv_jac_tag, gradient_tags>>`
 */
template <typename SystemType>
struct Derivatives {
  static constexpr size_t Dim = SystemType::volume_dim;
  using inv_jac_tag =
      Tags::InverseJacobian<Tags::ElementMap<Dim>,
                            Tags::Coordinates<Dim, Frame::Logical>>;

  using simple_tags = db::AddSimpleTags<>;
  using compute_tags = db::AddComputeTags<
      Tags::DerivCompute<typename SystemType::variables_tag, inv_jac_tag,
                         typename SystemType::gradient_tags>>;

  template <typename TagsList>
  static auto initialize(db::DataBox<TagsList>&& box) noexcept {
    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box));
  }
};
}  // namespace Initialization
}  // namespace Elliptic
