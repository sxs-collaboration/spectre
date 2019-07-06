// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Tags.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace dg {
namespace Initialization {
/// \ingroup InitializationGroup
/// %Tags that are to be sliced to the faces of the element
template <typename... Tags>
using slice_tags_to_face = tmpl::list<Tags...>;

/// \ingroup InitializationGroup
/// %Tags that are to be sliced to the exterior side of the faces of the element
template <typename... Tags>
using slice_tags_to_exterior = tmpl::list<Tags...>;

/// \ingroup InitializationGroup
/// Compute tags on the faces of the element
template <typename... Tags>
using face_compute_tags = tmpl::list<Tags...>;

/// \ingroup InitializationGroup
/// Compute tags on the exterior side of the faces of the element
template <typename... Tags>
using exterior_compute_tags = tmpl::list<Tags...>;
}  // namespace Initialization

namespace Actions {
/// \ingroup InitializationGroup
/// \brief Initialize items related to the interfaces between Elements and on
/// external boundaries
///
/// The `Initialization::slice_tags_to_face` and
/// `Initialization::slice_tags_to_exterior` types should be used for the
/// `SliceTagsToFace` and `SliceTagsToExterior` template parameters,
/// respectively. These are used to set additional volume quantities that are to
/// be sliced to the interfaces. Compute tags on the interfaces are controlled
/// using the `FaceComputeTags` and `ExteriorComputeTags` template parameters,
/// which should be passed the `Initialization::face_compute_tags` and
/// `Initialization::exterior_compute_tags` types, respectively.
///
/// Uses:
/// - System:
///   * `volume_dim`
///   * `variables_tag`
///   * `magnitude_tag`
///
/// DataBox changes:
/// - Adds:
///   * `Tags::Interface<Tags::BoundaryDirectionsExterior<volume_dim>,
///   variables_tag>` (as a simple tag)
///   * `face_tags<Tags::InternalDirections<Dim>>`
///   * `face_tags<Tags::BoundaryDirectionsInterior<Dim>>`
///   * `face_tags<Tags::BoundaryDirectionsExterior<Dim>>`
///
/// - For `face_tags<Directions>`:
///   * `Directions`
///   * `Tags::Interface<Directions, Tags::Directions<Dim>>`
///   * `Tags::Interface<Directions, Tags::Mesh<Dim - 1>>`
///   * `Tags::Interface<Directions, Tags::Coordinates<Dim, Frame::Inertial>>`
///   * `Tags::Interface<Directions, Tags::UnnormalizedFaceNormal<Dim>>`
///   * `Tags::Interface<Directions, Tags::Magnitude<
///   Tags::UnnormalizedFaceNormal<Dim>>>`
///   * `Tags::Interface<Directions, Tags::Normalized<
///   Tags::UnnormalizedFaceNormal<Dim>>>`
///
/// - Removes: nothing
/// - Modifies: nothing
template <
    typename System, typename SliceTagsToFace, typename SliceTagsToExterior,
    typename FaceComputeTags = Initialization::face_compute_tags<>,
    typename ExteriorComputeTags = Initialization::exterior_compute_tags<>>
struct InitializeInterfaces {
 private:
  static constexpr size_t dim = System::volume_dim;

  template <typename TagToSlice, typename Directions>
  struct make_slice_tag {
    using type = ::Tags::Slice<Directions, TagToSlice>;
  };

  template <typename ComputeTag, typename Directions>
  struct make_compute_tag {
    using type = ::Tags::InterfaceComputeItem<Directions, ComputeTag>;
  };

  template <typename Directions>
  using face_tags = tmpl::flatten<tmpl::list<
      Directions,
      ::Tags::InterfaceComputeItem<Directions, ::Tags::Direction<dim>>,
      ::Tags::InterfaceComputeItem<Directions, ::Tags::InterfaceMesh<dim>>,
      ::Tags::InterfaceComputeItem<
          Directions, ::Tags::BoundaryCoordinates<dim, Frame::Inertial>>,
      tmpl::transform<SliceTagsToFace,
                      make_slice_tag<tmpl::_1, tmpl::pin<Directions>>>,
      ::Tags::InterfaceComputeItem<Directions,
                                   ::Tags::UnnormalizedFaceNormal<dim>>,
      ::Tags::InterfaceComputeItem<Directions,
                                   typename System::template magnitude_tag<
                                       ::Tags::UnnormalizedFaceNormal<dim>>>,
      ::Tags::InterfaceComputeItem<
          Directions,
          ::Tags::NormalizedCompute<::Tags::UnnormalizedFaceNormal<dim>>>,
      tmpl::transform<FaceComputeTags,
                      make_compute_tag<tmpl::_1, tmpl::pin<Directions>>>>>;

  using exterior_face_tags = tmpl::flatten<tmpl::list<
      ::Tags::BoundaryDirectionsExterior<dim>,
      ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsExterior<dim>,
                                   ::Tags::Direction<dim>>,
      ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsExterior<dim>,
                                   ::Tags::InterfaceMesh<dim>>,
      ::Tags::InterfaceComputeItem<
          ::Tags::BoundaryDirectionsExterior<dim>,
          ::Tags::BoundaryCoordinates<dim, Frame::Inertial>>,
      tmpl::transform<
          SliceTagsToExterior,
          make_slice_tag<tmpl::_1,
                         tmpl::pin<::Tags::BoundaryDirectionsExterior<dim>>>>,
      ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsExterior<dim>,
                                   ::Tags::UnnormalizedFaceNormal<dim>>,
      ::Tags::InterfaceComputeItem<::Tags::BoundaryDirectionsExterior<dim>,
                                   typename System::template magnitude_tag<
                                       ::Tags::UnnormalizedFaceNormal<dim>>>,
      ::Tags::InterfaceComputeItem<
          ::Tags::BoundaryDirectionsExterior<dim>,
          ::Tags::NormalizedCompute<::Tags::UnnormalizedFaceNormal<dim>>>,
      tmpl::transform<
          ExteriorComputeTags,
          make_compute_tag<
              tmpl::_1, tmpl::pin<::Tags::BoundaryDirectionsExterior<dim>>>>>>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/, ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using simple_tags = db::AddSimpleTags<
        ::Tags::Interface<::Tags::BoundaryDirectionsExterior<dim>,
                          typename System::variables_tag>>;
    using compute_tags =
        tmpl::append<face_tags<::Tags::InternalDirections<dim>>,
                     face_tags<::Tags::BoundaryDirectionsInterior<dim>>,
                     exterior_face_tags>;

    const auto& mesh = db::get<::Tags::Mesh<dim>>(box);
    std::unordered_map<Direction<dim>,
                       db::item_type<typename System::variables_tag>>
        exterior_boundary_vars{};

    for (const auto& direction :
         db::get<::Tags::Element<dim>>(box).external_boundaries()) {
      exterior_boundary_vars[direction] =
          db::item_type<typename System::variables_tag>{
              mesh.slice_away(direction.dimension()).number_of_grid_points()};
    }

    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeInterfaces, simple_tags,
                                             compute_tags>(
            std::move(box), std::move(exterior_boundary_vars)));
  }
};
}  // namespace Actions
}  // namespace dg
