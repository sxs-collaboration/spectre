// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Initialization {

/// \brief Initialize items related to the interfaces between Elements and on
/// external boundaries
///
/// DataBox changes:
/// - Adds:
///   * `face_tags<Tags::InternalDirections<Dim>>`
///   * `face_tags<Tags::BoundaryDirections<Dim>>`
///
/// - For face_tags:
///   * `Tags::InterfaceComputeItem<Directions, Tags::Direction<Dim>>`
///   * `Tags::InterfaceComputeItem<Directions, Tags::InterfaceMesh<Dim>>`
///   * `Tags::Slice<Directions, typename System::variables_tag>`
///   * `Tags::InterfaceComputeItem<Directions,
///                                Tags::UnnormalizedFaceNormal<Dim>>`
///   * Tags::InterfaceComputeItem<Directions,
///                                typename System::template magnitude_tag<
///                                    Tags::UnnormalizedFaceNormal<Dim>>>
///   * `Tags::InterfaceComputeItem<
///         Directions, Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>`
/// - Removes: nothing
/// - Modifies: nothing
template <typename System>
struct Interface {
  static constexpr size_t dim = System::volume_dim;
  using simple_tags = db::AddSimpleTags<>;

  template <typename Directions>
  using face_tags = tmpl::list<
      Directions, Tags::InterfaceComputeItem<Directions, Tags::Direction<dim>>,
      Tags::InterfaceComputeItem<Directions, Tags::InterfaceMesh<dim>>,
      Tags::Slice<Directions, typename System::variables_tag>,
      Tags::Slice<Directions, typename System::spacetime_variables_tag>,
      Tags::InterfaceComputeItem<Directions, Tags::UnnormalizedFaceNormal<dim>>,
      Tags::InterfaceComputeItem<Directions,
                                 typename System::template magnitude_tag<
                                     Tags::UnnormalizedFaceNormal<dim>>>,
      Tags::InterfaceComputeItem<
          Directions, Tags::Normalized<Tags::UnnormalizedFaceNormal<dim>>>>;

  using compute_tags = tmpl::append<face_tags<Tags::InternalDirections<dim>>,
                                    face_tags<Tags::BoundaryDirections<dim>>>;

  template <typename TagsList>
  static auto initialize(db::DataBox<TagsList>&& box) noexcept {
    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box));
  }
};

}  // namespace Initialization
