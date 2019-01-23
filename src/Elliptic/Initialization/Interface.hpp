// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/FaceNormal.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/TMPL.hpp"

namespace Elliptic {
namespace Initialization {

/*!
 * \brief Initializes the DataBox tags related to data on the element faces
 *
 * With:
 * - `face<Tag>` =
 * `Tags::Interface<Tags::InternalDirections<volume_dim>, Tag>` **and**
 * `Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>, Tag>`
 * - `ext<Tag>` =
 * `Tags::Interface<Tags::BoundaryDirectionsExterior<volume_dim>, Tag>`
 *
 * Uses:
 * - System:
 *   - `volume_dim`
 *   - `variables_tag`
 *   - `gradient_tags`
 *   - `magnitude_tag`
 * - DataBox:
 *   - All items needed by the added compute tags
 *
 * DataBox:
 * - Adds:
 *   - `Tags::InternalDirections<volume_dim>`
 *   - `Tags::BoundaryDirectionsInterior<volume_dim>`
 *   - `face<Tags::Direction<volume_dim>>`
 *   - `face<Tags::Mesh<volume_dim - 1>>`
 *   - `face<variables_tag>` (as a compute tag)
 *   - `face<db::add_tag_prefix<Tags::deriv, db::variables_tag_with_tags_list<
 *   variables_tag, gradient_tags>, tmpl::size_t<volume_dim>, Frame::Inertial>>`
 *   - `face<Tags::Coordinates<volume_dim, Frame::Inertial>>`
 *   - `face<Tags::UnnormalizedFaceNormal<volume_dim>>`
 *   - `face<magnitude_tag<Tags::UnnormalizedFaceNormal<volume_dim>>>`
 *   - `face<Tags::Normalized<Tags::UnnormalizedFaceNormal<volume_dim>>>`
 *   - `Tags::BoundaryDirectionsExterior<volume_dim>`
 *   - `ext<Tags::Direction<volume_dim>>`
 *   - `ext<Tags::Mesh<volume_dim - 1>>`
 *   - `ext<variables_tag>` (as a simple tag)
 *   - `ext<db::add_tag_prefix<Tags::deriv, db::variables_tag_with_tags_list<
 *   variables_tag, gradient_tags>, tmpl::size_t<volume_dim>, Frame::Inertial>>`
 *   - `ext<Tags::UnnormalizedFaceNormal<volume_dim>>`
 *   - `ext<magnitude_tag<Tags::UnnormalizedFaceNormal<volume_dim>>>`
 *   - `ext<Tags::Normalized<Tags::UnnormalizedFaceNormal<volume_dim>>>`
 */
template <typename System>
struct Interface {
  static constexpr size_t volume_dim = System::volume_dim;
  template <typename Directions>
  using face_tags = tmpl::list<
      Directions,
      Tags::InterfaceComputeItem<Directions, Tags::Direction<volume_dim>>,
      Tags::InterfaceComputeItem<Directions, Tags::InterfaceMesh<volume_dim>>,
      Tags::Slice<Directions, typename System::variables_tag>,
      Tags::Slice<Directions, db::add_tag_prefix<
                                  Tags::deriv,
                                  db::variables_tag_with_tags_list<
                                      typename System::variables_tag,
                                      typename System::gradient_tags>,
                                  tmpl::size_t<volume_dim>, Frame::Inertial>>,
      Tags::InterfaceComputeItem<
          Directions, Tags::BoundaryCoordinates<volume_dim, Frame::Inertial>>,
      Tags::InterfaceComputeItem<Directions,
                                 Tags::UnnormalizedFaceNormal<volume_dim>>,
      Tags::InterfaceComputeItem<Directions,
                                 typename System::template magnitude_tag<
                                     Tags::UnnormalizedFaceNormal<volume_dim>>>,
      Tags::InterfaceComputeItem<
          Directions,
          Tags::Normalized<Tags::UnnormalizedFaceNormal<volume_dim>>>>;

  using exterior_face_tags = tmpl::list<
      Tags::BoundaryDirectionsExterior<volume_dim>,
      Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<volume_dim>,
                                 Tags::Direction<volume_dim>>,
      Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<volume_dim>,
                                 Tags::InterfaceMesh<volume_dim>>,
      // We slice the gradients to the exterior faces, too. This may need to be
      // reconsidered when boundary conditions are reworked.
      Tags::Slice<
          Tags::BoundaryDirectionsExterior<volume_dim>,
          db::add_tag_prefix<
              Tags::deriv,
              db::variables_tag_with_tags_list<typename System::variables_tag,
                                               typename System::gradient_tags>,
              tmpl::size_t<volume_dim>, Frame::Inertial>>,
      Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<volume_dim>,
                                 Tags::UnnormalizedFaceNormal<volume_dim>>,
      Tags::InterfaceComputeItem<Tags::BoundaryDirectionsExterior<volume_dim>,
                                 typename System::template magnitude_tag<
                                     Tags::UnnormalizedFaceNormal<volume_dim>>>,
      Tags::InterfaceComputeItem<
          Tags::BoundaryDirectionsExterior<volume_dim>,
          Tags::Normalized<Tags::UnnormalizedFaceNormal<volume_dim>>>>;

  using simple_tags = db::AddSimpleTags<
      // We rely on the variable data on exterior faces being updated manually.
      // This may need to be reconsidered when boundary conditions are reworked.
      Tags::Interface<Tags::BoundaryDirectionsExterior<volume_dim>,
                      typename System::variables_tag>>;
  using compute_tags =
      tmpl::append<face_tags<Tags::InternalDirections<volume_dim>>,
                   face_tags<Tags::BoundaryDirectionsInterior<volume_dim>>,
                   exterior_face_tags>;

  template <typename TagsList>
  static auto initialize(db::DataBox<TagsList>&& box) noexcept {
    const auto& mesh = db::get<Tags::Mesh<volume_dim>>(box);
    std::unordered_map<Direction<volume_dim>,
                       db::item_type<typename System::variables_tag>>
        external_boundary_vars{};

    for (const auto& direction :
         db::get<Tags::Element<volume_dim>>(box).external_boundaries()) {
      external_boundary_vars[direction] =
          db::item_type<typename System::variables_tag>{
              mesh.slice_away(direction.dimension()).number_of_grid_points()};
    }

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), std::move(external_boundary_vars));
  }
};

}  // namespace Initialization
}  // namespace Elliptic
