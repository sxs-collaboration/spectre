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
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Domain/Tags/SurfaceJacobian.hpp"
#include "Elliptic/DiscontinuousGalerkin/Penalty.hpp"
#include "Elliptic/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace elliptic::dg {

namespace detail {
template <size_t Dim>
void initialize_coords_and_jacobians(
    gsl::not_null<tnsr::I<DataVector, Dim, Frame::ElementLogical>*>
        logical_coords,
    gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> inertial_coords,
    gsl::not_null<InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                  Frame::Inertial>*>
        inv_jacobian,
    gsl::not_null<Scalar<DataVector>*> det_inv_jacobian,
    gsl::not_null<Scalar<DataVector>*> det_jacobian,
    gsl::not_null<InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                  Frame::Inertial>*>
        det_times_inv_jacobian,
    const Mesh<Dim>& mesh, const ElementMap<Dim, Frame::Inertial>& element_map,
    const domain::FunctionsOfTimeMap& functions_of_time);
}  // namespace detail

/*!
 * \brief Initialize the background-independent geometry for the elliptic DG
 * operator
 *
 * ## Geometric aliasing
 *
 * The geometric quantities such as Jacobians are evaluated on the DG grid.
 * Since we know them analytically, we could also evaluate them on a
 * higher-order grid or with a stronger quadrature (Gauss instead of
 * Gauss-Lobatto) to combat geometric aliasing. See discussions in
 * \cite Vincent2019qpd and \cite Fischer2021voj .
 */
template <size_t Dim>
struct InitializeGeometry {
  using return_tags = tmpl::list<
      domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
      domain::Tags::NeighborMesh<Dim>, domain::Tags::ElementMap<Dim>,
      domain::Tags::Coordinates<Dim, Frame::ElementLogical>,
      domain::Tags::Coordinates<Dim, Frame::Inertial>,
      domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                    Frame::Inertial>,
      domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>,
      domain::Tags::DetJacobian<Frame::ElementLogical, Frame::Inertial>,
      domain::Tags::DetTimesInvJacobian<Dim, Frame::ElementLogical,
                                        Frame::Inertial>>;
  using argument_tags =
      tmpl::list<domain::Tags::InitialExtents<Dim>,
                 domain::Tags::InitialRefinementLevels<Dim>,
                 domain::Tags::Domain<Dim>, domain::Tags::FunctionsOfTime,
                 elliptic::dg::Tags::Quadrature>;
  using volume_tags = argument_tags;
  static void apply(
      gsl::not_null<Mesh<Dim>*> mesh, gsl::not_null<Element<Dim>*> element,
      gsl::not_null<DirectionalIdMap<Dim, Mesh<Dim>>*> neighbor_meshes,
      gsl::not_null<ElementMap<Dim, Frame::Inertial>*> element_map,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::ElementLogical>*>
          logical_coords,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> inertial_coords,
      gsl::not_null<InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                    Frame::Inertial>*>
          inv_jacobian,
      gsl::not_null<Scalar<DataVector>*> det_inv_jacobian,
      gsl::not_null<Scalar<DataVector>*> det_jacobian,
      gsl::not_null<InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                    Frame::Inertial>*>
          det_times_inv_jacobian,
      const std::vector<std::array<size_t, Dim>>& initial_extents,
      const std::vector<std::array<size_t, Dim>>& initial_refinement,
      const Domain<Dim>& domain,
      const domain::FunctionsOfTimeMap& functions_of_time,
      Spectral::Quadrature quadrature, const ElementId<Dim>& element_id);
};

template <size_t Dim>
struct ProjectGeometry : tt::ConformsTo<::amr::protocols::Projector> {
  using return_tags = tmpl::list<
      domain::Tags::ElementMap<Dim>,
      domain::Tags::Coordinates<Dim, Frame::ElementLogical>,
      domain::Tags::Coordinates<Dim, Frame::Inertial>,
      domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                    Frame::Inertial>,
      domain::Tags::DetInvJacobian<Frame::ElementLogical, Frame::Inertial>,
      domain::Tags::DetJacobian<Frame::ElementLogical, Frame::Inertial>,
      domain::Tags::DetTimesInvJacobian<Dim, Frame::ElementLogical,
                                        Frame::Inertial>>;
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                 domain::Tags::Domain<Dim>, domain::Tags::FunctionsOfTime>;
  using volume_tags =
      tmpl::list<domain::Tags::Domain<Dim>, domain::Tags::FunctionsOfTime>;

  // p-refinement
  static void apply(
      const gsl::not_null<ElementMap<Dim, Frame::Inertial>*> element_map,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::ElementLogical>*>
          logical_coords,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          inertial_coords,
      const gsl::not_null<InverseJacobian<
          DataVector, Dim, Frame::ElementLogical, Frame::Inertial>*>
          inv_jacobian,
      const gsl::not_null<Scalar<DataVector>*> det_inv_jacobian,
      const gsl::not_null<Scalar<DataVector>*> det_jacobian,
      const gsl::not_null<InverseJacobian<
          DataVector, Dim, Frame::ElementLogical, Frame::Inertial>*>
          det_times_inv_jacobian,
      const Mesh<Dim>& mesh, const Element<Dim>& /*element*/,
      const Domain<Dim>& /*domain*/,
      const domain::FunctionsOfTimeMap& functions_of_time,
      const std::pair<Mesh<Dim>, Element<Dim>>& old_mesh_and_element) {
    if (mesh == old_mesh_and_element.first) {
      // Neighbors may have changed, but this element hasn't. Nothing to do.
      return;
    }
    // The element map doesn't change with p-refinement
    detail::initialize_coords_and_jacobians(
        logical_coords, inertial_coords, inv_jacobian, det_inv_jacobian,
        det_jacobian, det_times_inv_jacobian, mesh, *element_map,
        functions_of_time);
  }

  // h-refinement
  template <typename... ParentOrChildrenItemsType>
  static void apply(
      const gsl::not_null<ElementMap<Dim, Frame::Inertial>*> element_map,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::ElementLogical>*>
          logical_coords,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          inertial_coords,
      const gsl::not_null<InverseJacobian<
          DataVector, Dim, Frame::ElementLogical, Frame::Inertial>*>
          inv_jacobian,
      const gsl::not_null<Scalar<DataVector>*> det_inv_jacobian,
      const gsl::not_null<Scalar<DataVector>*> det_jacobian,
      const gsl::not_null<InverseJacobian<
          DataVector, Dim, Frame::ElementLogical, Frame::Inertial>*>
          det_times_inv_jacobian,
      const Mesh<Dim>& mesh, const Element<Dim>& element,
      const Domain<Dim>& domain,
      const domain::FunctionsOfTimeMap& functions_of_time,
      const ParentOrChildrenItemsType&... /*parent_or_children_items*/) {
    const auto& element_id = element.id();
    const auto& block = domain.blocks()[element_id.block_id()];
    *element_map = ElementMap<Dim, Frame::Inertial>{element_id, block};
    detail::initialize_coords_and_jacobians(
        logical_coords, inertial_coords, inv_jacobian, det_inv_jacobian,
        det_jacobian, det_times_inv_jacobian, mesh, *element_map,
        functions_of_time);
  }
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

template <size_t Dim>
void mortar_jacobian(
    gsl::not_null<Scalar<DataVector>*> mortar_jacobian,
    gsl::not_null<Scalar<DataVector>*> perpendicular_element_size,
    const Mesh<Dim - 1>& mortar_mesh,
    const ::dg::MortarSize<Dim - 1>& mortar_size,
    const Direction<Dim>& direction,
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>&
        mortar_logical_coords,
    const std::optional<tnsr::II<DataVector, Dim>>& inv_metric_on_mortar,
    const ElementMap<Dim, Frame::Inertial>& element_map,
    const domain::FunctionsOfTimeMap& functions_of_time);
}  // namespace detail

/// Initialize the geometry on faces and mortars for the elliptic DG operator
///
/// To normalize face normals this function needs the inverse background metric.
/// Pass the tag representing the inverse background metric to the
/// `InvMetricTag` template parameter, and the tag representing the analytic
/// background from which it can be retrieved to the `BackgroundTag` template
/// parameter. Set `InvMetricTag` and `BackgroundTag` to `void` to normalize
/// face normals with the Euclidean magnitude.
///
/// Mortar Jacobians are added only on nonconforming internal element
/// boundaries, i.e., when `Spectral::needs_projection()` is true.
///
/// The `::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<Dim>>` is only added
/// on external boundaries, for use by boundary conditions.
template <size_t Dim, typename InvMetricTag, typename BackgroundTag>
struct InitializeFacesAndMortars : tt::ConformsTo<::amr::protocols::Projector> {
  using return_tags = tmpl::append<
      domain::make_faces_tags<
          Dim,
          tmpl::list<domain::Tags::Direction<Dim>,
                     domain::Tags::Coordinates<Dim, Frame::Inertial>,
                     domain::Tags::FaceNormal<Dim>,
                     domain::Tags::FaceNormalVector<Dim>,
                     domain::Tags::UnnormalizedFaceNormalMagnitude<Dim>,
                     domain::Tags::DetSurfaceJacobian<Frame::ElementLogical,
                                                      Frame::Inertial>,
                     // This is the volume inverse Jacobian on the face grid
                     // points, multiplied by the determinant of the _face_
                     // Jacobian (the tag above)
                     domain::Tags::DetTimesInvJacobian<
                         Dim, Frame::ElementLogical, Frame::Inertial>,
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
                                 Dim>,
                 ::Tags::Mortars<elliptic::dg::Tags::PenaltyFactor, Dim>>>;
  using argument_tags = tmpl::append<
      tmpl::list<domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                 domain::Tags::NeighborMesh<Dim>, domain::Tags::ElementMap<Dim>,
                 domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                               Frame::Inertial>,
                 domain::Tags::Domain<Dim>, domain::Tags::FunctionsOfTime,
                 elliptic::dg::Tags::PenaltyParameter>,
      tmpl::conditional_t<
          std::is_same_v<BackgroundTag, void>, tmpl::list<>,
          tmpl::list<BackgroundTag, Parallel::Tags::Metavariables>>>;
  using volume_tags = tmpl::append<
      tmpl::list<domain::Tags::Domain<Dim>, domain::Tags::FunctionsOfTime,
                 elliptic::dg::Tags::PenaltyParameter>,
      tmpl::conditional_t<
          std::is_same_v<BackgroundTag, void>, tmpl::list<>,
          tmpl::list<BackgroundTag, Parallel::Tags::Metavariables>>>;
  template <typename... AmrData>
  static void apply(
      const gsl::not_null<DirectionMap<Dim, Direction<Dim>>*> face_directions,
      const gsl::not_null<DirectionMap<Dim, tnsr::I<DataVector, Dim>>*>
          faces_inertial_coords,
      const gsl::not_null<DirectionMap<Dim, tnsr::i<DataVector, Dim>>*>
          face_normals,
      const gsl::not_null<DirectionMap<Dim, tnsr::I<DataVector, Dim>>*>
          face_normal_vectors,
      const gsl::not_null<DirectionMap<Dim, Scalar<DataVector>>*>
          face_normal_magnitudes,
      const gsl::not_null<DirectionMap<Dim, Scalar<DataVector>>*>
          face_jacobians,
      const gsl::not_null<DirectionMap<
          Dim, InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                               Frame::Inertial>>*>
          face_jacobian_times_inv_jacobian,
      const gsl::not_null<DirectionMap<Dim, tnsr::ij<DataVector, Dim>>*>
          deriv_unnormalized_face_normals,
      const gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*> mortar_meshes,
      const gsl::not_null<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>*>
          mortar_sizes,
      const gsl::not_null<::dg::MortarMap<Dim, Scalar<DataVector>>*>
          mortar_jacobians,
      const gsl::not_null<::dg::MortarMap<Dim, Scalar<DataVector>>*>
          penalty_factors,
      const Mesh<Dim>& mesh, const Element<Dim>& element,
      const DirectionalIdMap<Dim, Mesh<Dim>>& neighbor_meshes,
      const ElementMap<Dim, Frame::Inertial>& element_map,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      const Domain<Dim>& domain,
      const domain::FunctionsOfTimeMap& functions_of_time,
      const double penalty_parameter, const AmrData&... amr_data) {
    apply(face_directions, faces_inertial_coords, face_normals,
          face_normal_vectors, face_normal_magnitudes, face_jacobians,
          face_jacobian_times_inv_jacobian, deriv_unnormalized_face_normals,
          mortar_meshes, mortar_sizes, mortar_jacobians, penalty_factors, mesh,
          element, neighbor_meshes, element_map, inv_jacobian, domain,
          functions_of_time, penalty_parameter, nullptr, nullptr, amr_data...);
  }
  template <typename Background, typename Metavariables, typename... AmrData>
  static void apply(
      const gsl::not_null<DirectionMap<Dim, Direction<Dim>>*> face_directions,
      const gsl::not_null<DirectionMap<Dim, tnsr::I<DataVector, Dim>>*>
          faces_inertial_coords,
      const gsl::not_null<DirectionMap<Dim, tnsr::i<DataVector, Dim>>*>
          face_normals,
      const gsl::not_null<DirectionMap<Dim, tnsr::I<DataVector, Dim>>*>
          face_normal_vectors,
      const gsl::not_null<DirectionMap<Dim, Scalar<DataVector>>*>
          face_normal_magnitudes,
      const gsl::not_null<DirectionMap<Dim, Scalar<DataVector>>*>
          face_jacobians,
      const gsl::not_null<DirectionMap<
          Dim, InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                               Frame::Inertial>>*>
          face_jacobian_times_inv_jacobian,
      const gsl::not_null<DirectionMap<Dim, tnsr::ij<DataVector, Dim>>*>
          deriv_unnormalized_face_normals,
      const gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*> mortar_meshes,
      const gsl::not_null<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>*>
          mortar_sizes,
      const gsl::not_null<::dg::MortarMap<Dim, Scalar<DataVector>>*>
          mortar_jacobians,
      const gsl::not_null<::dg::MortarMap<Dim, Scalar<DataVector>>*>
          penalty_factors,
      const Mesh<Dim>& mesh, const Element<Dim>& element,
      const DirectionalIdMap<Dim, Mesh<Dim>>& neighbor_meshes,
      const ElementMap<Dim, Frame::Inertial>& element_map,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      const Domain<Dim>& domain,
      const domain::FunctionsOfTimeMap& functions_of_time,
      const double penalty_parameter, const Background& background,
      const Metavariables& /*meta*/, const AmrData&... /*amr_data*/) {
    static_assert(std::is_same_v<InvMetricTag, void> or
                      not(std::is_same_v<Background, std::nullptr_t>),
                  "Supply an analytic background from which the 'InvMetricTag' "
                  "can be retrieved");
    [[maybe_unused]] const auto get_inv_metric =
        [&background]([[maybe_unused]] const tnsr::I<DataVector, Dim>&
                          local_inertial_coords)
        -> std::optional<tnsr::II<DataVector, Dim>> {
      if constexpr (not std::is_same_v<InvMetricTag, void>) {
        using factory_classes = typename std::decay_t<
            Metavariables>::factory_creation::factory_classes;
        return call_with_dynamic_type<tnsr::II<DataVector, Dim>,
                                      tmpl::at<factory_classes, Background>>(
            &background, [&local_inertial_coords](const auto* const derived) {
              return get<InvMetricTag>(derived->variables(
                  local_inertial_coords, tmpl::list<InvMetricTag>{}));
            });
      } else {
        (void)background;
        return std::nullopt;
      }
    };
    ASSERT(
        alg::all_of(mesh.quadrature(),
                    [&mesh](const auto t) { return t == mesh.quadrature(0); }),
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
      face_inertial_coords =
          element_map(face_logical_coords, 0., functions_of_time);
      auto& face_normal = (*face_normals)[direction];
      auto& face_normal_vector = (*face_normal_vectors)[direction];
      auto& face_normal_magnitude = (*face_normal_magnitudes)[direction];
      // Buffer the inv Jacobian on the face here, then multiply by the face
      // Jacobian below
      auto& inv_jacobian_on_face =
          (*face_jacobian_times_inv_jacobian)[direction];
      inv_jacobian_on_face =
          element_map.inv_jacobian(face_logical_coords, 0., functions_of_time);
      unnormalized_face_normal(make_not_null(&face_normal), face_mesh,
                               inv_jacobian_on_face, direction);
      if constexpr (std::is_same_v<InvMetricTag, void>) {
        magnitude(make_not_null(&face_normal_magnitude), face_normal);
        for (size_t d = 0; d < Dim; ++d) {
          face_normal.get(d) /= get(face_normal_magnitude);
          face_normal_vector.get(d) = face_normal.get(d);
        }
      } else {
        const auto inv_metric_on_face = *get_inv_metric(face_inertial_coords);
        magnitude(make_not_null(&face_normal_magnitude), face_normal,
                  inv_metric_on_face);
        for (size_t d = 0; d < Dim; ++d) {
          face_normal.get(d) /= get(face_normal_magnitude);
        }
        raise_or_lower_index(make_not_null(&face_normal_vector), face_normal,
                             inv_metric_on_face);
      }
      auto& face_jacobian = (*face_jacobians)[direction];
      get(face_jacobian) =
          get(face_normal_magnitude) / get(determinant(inv_jacobian_on_face));
      for (auto& component : inv_jacobian_on_face) {
        component *= get(face_jacobian);
      }
    }
    // Compute the Jacobian derivative numerically, because our coordinate maps
    // currently don't provide it analytically.
    detail::deriv_unnormalized_face_normals_impl(
        deriv_unnormalized_face_normals, mesh, element, inv_jacobian);
    // Mortars (internal directions)
    mortar_meshes->clear();
    mortar_sizes->clear();
    mortar_jacobians->clear();
    penalty_factors->clear();
    const auto& element_id = element.id();
    for (const auto& [direction, neighbors] : element.neighbors()) {
      const auto face_mesh = mesh.slice_away(direction.dimension());
      const auto& orientation = neighbors.orientation();
      for (const auto& neighbor_id : neighbors) {
        const ::dg::MortarId<Dim> mortar_id{direction, neighbor_id};
        const auto& neighbor_mesh = neighbor_meshes.at(mortar_id);
        mortar_meshes->emplace(
            mortar_id, ::dg::mortar_mesh(
                           face_mesh, orientation.inverse_map()(neighbor_mesh)
                                          .slice_away(direction.dimension())));
        mortar_sizes->emplace(
            mortar_id, ::dg::mortar_size(element_id, neighbor_id,
                                         direction.dimension(), orientation));
        // Mortar Jacobian
        const auto& mortar_mesh = mortar_meshes->at(mortar_id);
        const auto& mortar_size = mortar_sizes->at(mortar_id);
        const auto mortar_logical_coords = detail::mortar_logical_coordinates(
            mortar_mesh, mortar_size, direction);
        const auto mortar_inertial_coords =
            element_map(mortar_logical_coords, 0., functions_of_time);
        Scalar<DataVector> perpendicular_element_size{};
        if (Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size)) {
          auto& mortar_jacobian = (*mortar_jacobians)[mortar_id];
          detail::mortar_jacobian(make_not_null(&mortar_jacobian),
                                  make_not_null(&perpendicular_element_size),
                                  mortar_mesh, mortar_size, direction,
                                  mortar_logical_coords,
                                  get_inv_metric(mortar_inertial_coords),
                                  element_map, functions_of_time);
        } else {
          // Mortar is identical to face, and we have computed the face normal
          // magnitude already above
          get(perpendicular_element_size) =
              2. / get(face_normal_magnitudes->at(direction));
        }
        // Penalty factor
        // The penalty factor (like all quantities on mortars) must agree when
        // calculated on both sides of the mortar. So we switch perspective to
        // the neighbor here.
        const auto direction_in_neighbor = orientation(direction).opposite();
        const auto reoriented_mortar_mesh = ::dg::mortar_mesh(
            orientation(mesh).slice_away(direction_in_neighbor.dimension()),
            neighbor_mesh.slice_away(direction_in_neighbor.dimension()));
        const auto mortar_size_in_neighbor = ::dg::mortar_size(
            neighbor_id, element_id, direction_in_neighbor.dimension(),
            orientation.inverse_map());
        const auto mortar_logical_coords_in_neighbor =
            detail::mortar_logical_coordinates(reoriented_mortar_mesh,
                                               mortar_size_in_neighbor,
                                               direction_in_neighbor);
        const ElementMap<Dim, Frame::Inertial> neighbor_element_map{
            neighbor_id, domain.blocks()[neighbor_id.block_id()]};
        const auto reoriented_mortar_inertial_coords = neighbor_element_map(
            mortar_logical_coords_in_neighbor, 0., functions_of_time);
        Scalar<DataVector> buffer{};
        Scalar<DataVector> reoriented_neighbor_element_size{};
        detail::mortar_jacobian(
            make_not_null(&buffer),
            make_not_null(&reoriented_neighbor_element_size),
            reoriented_mortar_mesh, mortar_size_in_neighbor,
            direction_in_neighbor, mortar_logical_coords_in_neighbor,
            get_inv_metric(reoriented_mortar_inertial_coords),
            neighbor_element_map, functions_of_time);
        // Orient the result back to the perspective of this element
        const auto neighbor_element_size = orient_variables_on_slice(
            get(reoriented_neighbor_element_size),
            reoriented_mortar_mesh.extents(), direction_in_neighbor.dimension(),
            orientation.inverse_map());
        penalty_factors->emplace(
            mortar_id, elliptic::dg::penalty(
                           blaze::min(get(perpendicular_element_size),
                                      neighbor_element_size),
                           std::max(mesh.extents(direction.dimension()),
                                    neighbor_mesh.extents(
                                        direction_in_neighbor.dimension())),
                           penalty_parameter));
      }  // neighbors
    }    // internal directions
    // Mortars (external directions)
    for (const auto& direction : element.external_boundaries()) {
      const auto face_mesh = mesh.slice_away(direction.dimension());
      const auto mortar_id =
          DirectionalId<Dim>{direction, ElementId<Dim>::external_boundary_id()};
      mortar_meshes->emplace(mortar_id, face_mesh);
      mortar_sizes->emplace(mortar_id,
                            make_array<Dim - 1>(Spectral::MortarSize::Full));
      penalty_factors->emplace(
          mortar_id,
          elliptic::dg::penalty(2. / get(face_normal_magnitudes->at(direction)),
                                mesh.extents(direction.dimension()),
                                penalty_parameter));
    }  // external directions
  }
};

/// Initialize background quantities for the elliptic DG operator, possibly
/// including the metric necessary for normalizing face normals
template <size_t Dim, typename BackgroundFields, typename BackgroundTag>
struct InitializeBackground : tt::ConformsTo<::amr::protocols::Projector> {
  using return_tags =
      tmpl::list<::Tags::Variables<BackgroundFields>,
                 domain::Tags::Faces<Dim, ::Tags::Variables<BackgroundFields>>>;
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 domain::Tags::Mesh<Dim>,
                 domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                               Frame::Inertial>,
                 BackgroundTag, Parallel::Tags::Metavariables>;

  template <typename BackgroundBase, typename Metavariables,
            typename... AmrData>
  static void apply(
      const gsl::not_null<Variables<BackgroundFields>*> background_fields,
      const gsl::not_null<DirectionMap<Dim, Variables<BackgroundFields>>*>
          face_background_fields,
      const tnsr::I<DataVector, Dim>& inertial_coords, const Mesh<Dim>& mesh,
      const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      const BackgroundBase& background, const Metavariables& /*meta*/,
      const AmrData&... amr_data) {
    if constexpr (sizeof...(AmrData) == 1) {
      if constexpr (std::is_same_v<AmrData...,
                                   std::pair<Mesh<Dim>, Element<Dim>>>) {
        if (((mesh == amr_data.first) and ...)) {
          // This element hasn't changed during AMR. Nothing to do.
          return;
        }
      }
    }

    // Background fields in the volume
    using factory_classes =
        typename std::decay_t<Metavariables>::factory_creation::factory_classes;
    *background_fields =
        call_with_dynamic_type<Variables<BackgroundFields>,
                               tmpl::at<factory_classes, BackgroundBase>>(
            &background, [&inertial_coords, &mesh,
                          &inv_jacobian](const auto* const derived) {
              return variables_from_tagged_tuple(derived->variables(
                  inertial_coords, mesh, inv_jacobian, BackgroundFields{}));
            });
    // Background fields on faces
    for (const auto& direction : Direction<Dim>::all_directions()) {
      // Possible optimization: Only the background fields in the
      // System::fluxes_computer::argument_tags are needed on internal faces.
      // On Gauss grids we could evaluate the background directly on the faces
      // instead of projecting the data. However, we need to take derivatives of
      // the background fields, so we could evaluate them on a Gauss-Lobatto
      // grid in the volume. We could even evaluate the background fields on a
      // higher-order grid and project down to get more accurate derivatives if
      // needed.
      (*face_background_fields)[direction].initialize(
          mesh.slice_away(direction.dimension()).number_of_grid_points());
      ::dg::project_contiguous_data_to_boundary(
          make_not_null(&(*face_background_fields)[direction]),
          *background_fields, mesh, direction);
    }
  }
};

}  // namespace elliptic::dg
