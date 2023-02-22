// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Domain/Tags/SurfaceJacobian.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic::dg {

/// Initialize the background-independent geometry for the elliptic DG operator
template <size_t Dim>
struct InitializeGeometry {
  using return_tags = tmpl::list<
      domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
      domain::Tags::ElementMap<Dim>,
      domain::Tags::Coordinates<Dim, Frame::ElementLogical>,
      domain::Tags::Coordinates<Dim, Frame::Inertial>,
      domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                    Frame::Inertial>,
      domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>>;
  using argument_tags = tmpl::list<domain::Tags::InitialExtents<Dim>,
                                   domain::Tags::InitialRefinementLevels<Dim>,
                                   domain::Tags::Domain<Dim>>;
  void operator()(
      gsl::not_null<Mesh<Dim>*> mesh, gsl::not_null<Element<Dim>*> element,
      gsl::not_null<ElementMap<Dim, Frame::Inertial>*> element_map,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::ElementLogical>*>
          logical_coords,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> inertial_coords,
      gsl::not_null<InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                    Frame::Inertial>*>
          inv_jacobian,
      gsl::not_null<Scalar<DataVector>*> det_inv_jacobian,
      const std::vector<std::array<size_t, Dim>>& initial_extents,
      const std::vector<std::array<size_t, Dim>>& initial_refinement,
      const Domain<Dim>& domain, const ElementId<Dim>& element_id) const;
};

namespace detail {
// Compute derivative of the Jacobian numerically
template <size_t Dim>
void deriv_unnormalized_face_normals_impl(
    gsl::not_null<DirectionMap<Dim, tnsr::ij<DataVector, Dim>>*>
        deriv_unnormalized_face_normals,
    const Mesh<Dim>& mesh, const Element<Dim>& element,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian);

// Get element-logical coordinates of the mortar collocation points
template <size_t Dim>
tnsr::I<DataVector, Dim, Frame::ElementLogical> mortar_logical_coordinates(
    const Mesh<Dim - 1>& mortar_mesh,
    const ::dg::MortarSize<Dim - 1>& mortar_size,
    const Direction<Dim>& direction);
}  // namespace detail

/// Initialize the geometry on faces and mortars for the elliptic DG operator
///
/// To normalize face normals this function needs the inverse background metric.
/// Pass the tag representing the inverse background metric to the
/// `InvMetricTag` template parameter, and pass the analytic background from
/// which it can be retrieved as additional argument to the call operator. Set
/// `InvMetricTag` to `void` to normalize face normals with the Euclidean
/// magnitude.
///
/// Mortar Jacobians are added only on nonconforming internal element
/// boundaries, i.e., when `Spectral::needs_projection()` is true.
///
/// The `::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<Dim>>` is only added
/// on external boundaries, for use by boundary conditions.
template <size_t Dim, typename InvMetricTag>
struct InitializeFacesAndMortars {
  using return_tags = tmpl::append<
      domain::make_faces_tags<
          Dim,
          tmpl::list<domain::Tags::Direction<Dim>,
                     domain::Tags::Coordinates<Dim, Frame::Inertial>,
                     domain::Tags::FaceNormal<Dim>,
                     domain::Tags::UnnormalizedFaceNormalMagnitude<Dim>,
                     domain::Tags::DetSurfaceJacobian<Frame::ElementLogical,
                                                      Frame::Inertial>,
                     // Possible optimization: The derivative of the face normal
                     // could be omitted for some systems, but its memory usage
                     // is probably insignificant since it's only added on
                     // external boundaries.
                     ::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<Dim>,
                                   tmpl::size_t<Dim>, Frame::Inertial>>>,
      tmpl::list<::Tags::Mortars<domain::Tags::Mesh<Dim - 1>, Dim>,
                 ::Tags::Mortars<::Tags::MortarSize<Dim - 1>, Dim>,
                 ::Tags::Mortars<domain::Tags::DetSurfaceJacobian<
                                     Frame::ElementLogical, Frame::Inertial>,
                                 Dim>>>;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                 domain::Tags::ElementMap<Dim>,
                 domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                               Frame::Inertial>>;
  template <typename Background = std::nullptr_t, typename... BackgroundClasses>
  void operator()(
      const gsl::not_null<DirectionMap<Dim, Direction<Dim>>*> face_directions,
      const gsl::not_null<DirectionMap<Dim, tnsr::I<DataVector, Dim>>*>
          faces_inertial_coords,
      const gsl::not_null<DirectionMap<Dim, tnsr::i<DataVector, Dim>>*>
          face_normals,
      const gsl::not_null<DirectionMap<Dim, Scalar<DataVector>>*>
          face_normal_magnitudes,
      const gsl::not_null<DirectionMap<Dim, Scalar<DataVector>>*>
          face_jacobians,
      const gsl::not_null<DirectionMap<Dim, tnsr::ij<DataVector, Dim>>*>
          deriv_unnormalized_face_normals,
      const gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*> mortar_meshes,
      const gsl::not_null<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>*>
          mortar_sizes,
      const gsl::not_null<::dg::MortarMap<Dim, Scalar<DataVector>>*>
          mortar_jacobians,
      const Mesh<Dim>& mesh, const Element<Dim>& element,
      const ElementMap<Dim, Frame::Inertial>& element_map,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      const std::vector<std::array<size_t, Dim>>& initial_extents,
      const Background& background = std::nullptr_t{},
      tmpl::list<BackgroundClasses...> /*meta*/ = tmpl::list<>{}) const {
    static_assert(std::is_same_v<InvMetricTag, void> or
                      not(std::is_same_v<Background, std::nullptr_t>),
                  "Supply an analytic background from which the 'InvMetricTag' "
                  "can be retrieved");
    [[maybe_unused]] const auto get_inv_metric =
        [&background]([[maybe_unused]] const tnsr::I<DataVector, Dim>&
                          local_inertial_coords) {
          if constexpr (not std::is_same_v<InvMetricTag, void>) {
            return call_with_dynamic_type<tnsr::II<DataVector, Dim>,
                                          tmpl::list<BackgroundClasses...>>(
                &background,
                [&local_inertial_coords](const auto* const derived) {
                  return get<InvMetricTag>(derived->variables(
                      local_inertial_coords, tmpl::list<InvMetricTag>{}));
                });
          } else {
            (void)background;
          }
        };
    const Spectral::Quadrature quadrature = mesh.quadrature(0);
    ASSERT(std::equal(mesh.quadrature().begin() + 1, mesh.quadrature().end(),
                      mesh.quadrature().begin()),
           "This function is implemented assuming the quadrature is isotropic");
    // Faces
    for (const auto& direction : Direction<Dim>::all_directions()) {
      const auto face_mesh = mesh.slice_away(direction.dimension());
      (*face_directions)[direction] = direction;
      // Possible optimization: Not all systems need the coordinates on internal
      // faces.
      const auto face_logical_coords =
          interface_logical_coordinates(face_mesh, direction);
      auto& face_inertial_coords = (*faces_inertial_coords)[direction];
      face_inertial_coords = element_map.operator()(face_logical_coords);
      auto& face_normal = (*face_normals)[direction];
      auto& face_normal_magnitude = (*face_normal_magnitudes)[direction];
      face_normal = unnormalized_face_normal(face_mesh, element_map, direction);
      if constexpr (std::is_same_v<InvMetricTag, void>) {
        magnitude(make_not_null(&face_normal_magnitude), face_normal);
      } else {
        const auto inv_metric_on_face = get_inv_metric(face_inertial_coords);
        magnitude(make_not_null(&face_normal_magnitude), face_normal,
                  inv_metric_on_face);
      }
      for (size_t d = 0; d < Dim; ++d) {
        face_normal.get(d) /= get(face_normal_magnitude);
      }
      get((*face_jacobians)[direction]) =
          get(determinant(element_map.jacobian(face_logical_coords))) *
          get(face_normal_magnitude);
    }
    // Compute the Jacobian derivative numerically, because our coordinate maps
    // currently don't provide it analytically.
    detail::deriv_unnormalized_face_normals_impl(
        deriv_unnormalized_face_normals, mesh, element, inv_jacobian);
    // Mortars (internal directions)
    const auto& element_id = element.id();
    for (const auto& [direction, neighbors] : element.neighbors()) {
      const auto face_mesh = mesh.slice_away(direction.dimension());
      const auto& orientation = neighbors.orientation();
      for (const auto& neighbor_id : neighbors) {
        const ::dg::MortarId<Dim> mortar_id{direction, neighbor_id};
        mortar_meshes->emplace(
            mortar_id,
            ::dg::mortar_mesh(
                face_mesh,
                domain::Initialization::create_initial_mesh(
                    initial_extents, neighbor_id, quadrature, orientation)
                    .slice_away(direction.dimension())));
        mortar_sizes->emplace(
            mortar_id, ::dg::mortar_size(element_id, neighbor_id,
                                         direction.dimension(), orientation));
        // Mortar Jacobian
        const auto& mortar_mesh = mortar_meshes->at(mortar_id);
        const auto& mortar_size = mortar_sizes->at(mortar_id);
        if (Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size)) {
          const auto mortar_logical_coords = detail::mortar_logical_coordinates(
              mortar_mesh, mortar_size, direction);
          auto& mortar_jacobian = (*mortar_jacobians)[mortar_id];
          mortar_jacobian =
              determinant(element_map.jacobian(mortar_logical_coords));
          // These factors of two account for the mortar size
          for (const auto& mortar_size_i : mortar_size) {
            if (mortar_size_i != Spectral::MortarSize::Full) {
              get(mortar_jacobian) *= 0.5;
            }
          }
          const auto inv_jacobian_on_mortar =
              element_map.inv_jacobian(mortar_logical_coords);
          const auto unnormalized_mortar_normal = unnormalized_face_normal(
              mortar_mesh, inv_jacobian_on_mortar, direction);
          Scalar<DataVector> mortar_normal_magnitude{};
          if constexpr (std::is_same_v<InvMetricTag, void>) {
            magnitude(make_not_null(&mortar_normal_magnitude),
                      unnormalized_mortar_normal);
          } else {
            const auto mortar_inertial_coords =
                element_map(mortar_logical_coords);
            const auto inv_metric_on_mortar =
                get_inv_metric(mortar_inertial_coords);
            magnitude(make_not_null(&mortar_normal_magnitude),
                      unnormalized_mortar_normal, inv_metric_on_mortar);
          }
          get(mortar_jacobian) *= get(mortar_normal_magnitude);
        }
      }  // neighbors
    }    // internal directions
    // Mortars (external directions)
    for (const auto& direction : element.external_boundaries()) {
      const auto face_mesh = mesh.slice_away(direction.dimension());
      const auto mortar_id =
          std::make_pair(direction, ElementId<Dim>::external_boundary_id());
      mortar_meshes->emplace(mortar_id, face_mesh);
      mortar_sizes->emplace(mortar_id,
                            make_array<Dim - 1>(Spectral::MortarSize::Full));
    }  // external directions
  }
};

/// Initialize background quantities for the elliptic DG operator, possibly
/// including the metric necessary for normalizing face normals
template <size_t Dim, typename BackgroundFields>
struct InitializeBackground {
  using return_tags =
      tmpl::list<::Tags::Variables<BackgroundFields>,
                 domain::Tags::Faces<Dim, ::Tags::Variables<BackgroundFields>>>;
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 domain::Tags::Mesh<Dim>,
                 domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                               Frame::Inertial>>;

  template <typename BackgroundBase, typename... BackgroundClasses>
  void operator()(
      const gsl::not_null<Variables<BackgroundFields>*> background_fields,
      const gsl::not_null<DirectionMap<Dim, Variables<BackgroundFields>>*>
          face_background_fields,
      const tnsr::I<DataVector, Dim>& inertial_coords, const Mesh<Dim>& mesh,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      const BackgroundBase& background,
      tmpl::list<BackgroundClasses...> /*meta*/) const {
    *background_fields =
        call_with_dynamic_type<Variables<BackgroundFields>,
                               tmpl::list<BackgroundClasses...>>(
            &background, [&inertial_coords, &mesh,
                          &inv_jacobian](const auto* const derived) {
              return variables_from_tagged_tuple(derived->variables(
                  inertial_coords, mesh, inv_jacobian, BackgroundFields{}));
            });
    ASSERT(mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto,
           "Only Gauss-Lobatto quadrature is currently implemented for "
           "slicing background fields to faces.");
    for (const auto& direction : Direction<Dim>::all_directions()) {
      // Possible optimization: Only the background fields in the
      // System::fluxes_computer::argument_tags are needed on internal faces.
      data_on_slice(
          make_not_null(&(*face_background_fields)[direction]),
          *background_fields, mesh.extents(), direction.dimension(),
          index_to_slice_at(mesh.extents(), direction));
    }
  }
};

}  // namespace elliptic::dg
