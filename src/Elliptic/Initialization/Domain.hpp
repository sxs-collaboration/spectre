// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/MinimumGridSpacing.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/Helpers.hpp"

/// \cond
template <size_t VolumeDim>
class ElementIndex;
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace Elliptic {
namespace Initialization {

/*!
 * \brief Initializes the DataBox tag related to the computational domain
 *
 * DataBox:
 * - Adds:
 *   - `Tags::Mesh<Dim>`
 *   - `Tags::Element<Dim>`
 *   - `Tags::ElementMap<Dim, Frame::Inertial>`
 *   - `Tags::Coordinates<Dim, Frame::Logical>`
 *   - `Tags::Coordinates<Dim, Frame::Inertial>`
 *   - `Tags::InverseJacobian<
 *   Tags::ElementMap<Dim>, Tags::Coordinates<Dim, Frame::Logical>>`
 */
template <size_t Dim>
struct Domain {
  using simple_tags = db::AddSimpleTags<Tags::Mesh<Dim>, Tags::Element<Dim>,
                                        Tags::ElementMap<Dim, Frame::Inertial>>;
  using compute_tags = db::AddComputeTags<
      Tags::LogicalCoordinates<Dim>,
      Tags::MappedCoordinates<Tags::ElementMap<Dim>,
                              Tags::Coordinates<Dim, Frame::Logical>>,
      Tags::InverseJacobian<Tags::ElementMap<Dim>,
                            Tags::Coordinates<Dim, Frame::Logical>>>;

  template <typename TagsList>
  static auto initialize(
      db::DataBox<TagsList>&& box, const ElementIndex<Dim>& array_index,
      const std::vector<std::array<size_t, Dim>>& initial_extents,
      const ::Domain<Dim, Frame::Inertial>& domain) noexcept {
    const ElementId<Dim> element_id{array_index};
    const auto& my_block = domain.blocks()[element_id.block_id()];
    Mesh<Dim> mesh =
        ::Initialization::element_mesh(initial_extents, element_id);
    Element<Dim> element = create_initial_element(element_id, my_block);
    ElementMap<Dim, Frame::Inertial> map{element_id,
                                         my_block.coordinate_map().get_clone()};

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), std::move(mesh), std::move(element), std::move(map));
  }
};

}  // namespace Initialization
}  // namespace Elliptic
