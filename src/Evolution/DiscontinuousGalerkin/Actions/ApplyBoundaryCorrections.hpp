// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrectionTags.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Evolution/DiscontinuousGalerkin/LiftFromBoundary.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace evolution::dg::Actions {
namespace detail {
template <typename... BoundaryCorrectionTags, typename... Tags,
          typename BoundaryCorrection>
void boundary_correction(
    const gsl::not_null<Variables<tmpl::list<BoundaryCorrectionTags...>>*>
        boundary_corrections_on_mortar,
    const Variables<tmpl::list<Tags...>>& local_boundary_data,
    const Variables<tmpl::list<Tags...>>& neighbor_boundary_data,
    const BoundaryCorrection& boundary_correction,
    const ::dg::Formulation dg_formulation) noexcept {
  boundary_correction.dg_boundary_terms(
      make_not_null(
          &get<BoundaryCorrectionTags>(*boundary_corrections_on_mortar))...,
      get<Tags>(local_boundary_data)..., get<Tags>(neighbor_boundary_data)...,
      dg_formulation);
}
}  // namespace detail

/*!
 * \brief Computes the boundary corrections and lifts them to the volume.
 *
 * Given the data from both sides of each mortar, computes the boundary
 * correction on each mortar and then lifts it into the volume.
 *
 * Future additions include:
 * - boundary conditions, both through ghost cells and by changing the time
 *   derivatives.
 * - support local time stepping (shouldn't be very difficult)
 */
template <typename Metavariables>
struct ApplyBoundaryCorrections {
  using inbox_tags =
      tmpl::list<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          Metavariables::volume_dim>>;
  using const_global_cache_tags = tmpl::list<
      evolution::Tags::BoundaryCorrection<typename Metavariables::system>,
      ::dg::Tags::Formulation>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    constexpr size_t volume_dim = Metavariables::system::volume_dim;

    if (UNLIKELY(db::get<domain::Tags::Element<volume_dim>>(box)
                     .number_of_neighbors() == 0)) {
      // We have no neighbors, yay!
      return {std::move(box)};
    }

    if constexpr (Metavariables::local_time_stepping) {
      ERROR("Local time stepping not supported");
    } else {
      apply_global_time_stepping(make_not_null(&box), make_not_null(&inboxes));
    }
    return {std::move(box)};
  }

  template <typename DbTags, typename... InboxTags, typename ArrayIndex>
  static bool is_ready(const db::DataBox<DbTags>& box,
                       const tuples::TaggedTuple<InboxTags...>& inboxes,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ArrayIndex& /*array_index*/) noexcept;

 private:
  template <typename DbTagsList, typename... InboxTags>
  static void apply_global_time_stepping(
      gsl::not_null<db::DataBox<DbTagsList>*> box,
      gsl::not_null<tuples::TaggedTuple<InboxTags...>*> inboxes) noexcept;
};

template <typename Metavariables>
template <typename DbTagsList, typename... InboxTags>
void ApplyBoundaryCorrections<Metavariables>::apply_global_time_stepping(
    const gsl::not_null<db::DataBox<DbTagsList>*> box,
    const gsl::not_null<tuples::TaggedTuple<InboxTags...>*> inboxes) noexcept {
  constexpr size_t volume_dim = Metavariables::system::volume_dim;

  // Move inbox contents into the DataBox
  using Key = std::pair<Direction<volume_dim>, ElementId<volume_dim>>;
  std::map<
      TimeStepId,
      FixedHashMap<
          maximum_number_of_neighbors(volume_dim), Key,
          std::tuple<Mesh<volume_dim - 1>, std::optional<std::vector<double>>,
                     std::optional<std::vector<double>>, ::TimeStepId>,
          boost::hash<Key>>>& inbox =
      tuples::get<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
          volume_dim>>(*inboxes);
  const auto& temporal_id = get<::Tags::TimeStepId>(*box);
  const auto received_temporal_id_and_data = inbox.find(temporal_id);
  db::mutate<evolution::dg::Tags::MortarData<volume_dim>>(
      box,
      [&received_temporal_id_and_data](
          const gsl::not_null<std::unordered_map<
              Key, evolution::dg::MortarData<volume_dim>, boost::hash<Key>>*>
              mortar_data) noexcept {
        for (auto& received_mortar_data :
             received_temporal_id_and_data->second) {
          const auto& mortar_id = received_mortar_data.first;
          mortar_data->at(mortar_id).insert_neighbor_mortar_data(
              received_temporal_id_and_data->first,
              std::get<0>(received_mortar_data.second),
              std::move(*std::get<2>(received_mortar_data.second)));
        }
      });
  inbox.erase(received_temporal_id_and_data);

  // Set up helper lambda that will compute and lift the boundary corrections
  const Mesh<volume_dim>& volume_mesh =
      db::get<domain::Tags::Mesh<volume_dim>>(*box);
  ASSERT(
      volume_mesh.quadrature() ==
          make_array<volume_dim>(volume_mesh.quadrature(0)),
      "Must have isotropic quadrature, but got volume mesh: " << volume_mesh);
  const bool using_gauss_lobatto_points =
      volume_mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto;

  const Scalar<DataVector>& volume_det_inv_jacobian =
      db::get<domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>>(
          *box);
  const DataVector volume_det_jacobian = 1.0 / get(volume_det_inv_jacobian);

  const auto& mortar_meshes = db::get<Tags::MortarMesh<volume_dim>>(*box);
  const auto& mortar_sizes = db::get<Tags::MortarSize<volume_dim>>(*box);
  const ::dg::Formulation dg_formulation =
      db::get<::dg::Tags::Formulation>(*box);

  const DirectionMap<volume_dim,
                     std::optional<Variables<tmpl::list<
                         evolution::dg::Tags::MagnitudeOfNormal,
                         evolution::dg::Tags::NormalCovector<volume_dim>>>>>&
      face_normal_covector_and_magnitude =
          db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<volume_dim>>(
              *box);

  using variables_tag = typename Metavariables::system::variables_tag;
  using variables_tags = typename variables_tag::tags_list;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

  const auto compute_and_lift_boundary_corrections =
      [&dg_formulation, &face_normal_covector_and_magnitude, &mortar_meshes,
       &mortar_sizes, using_gauss_lobatto_points, &volume_det_inv_jacobian,
       &volume_det_jacobian,
       &volume_mesh](const auto dt_variables_ptr, const auto mortar_data_ptr,
                     const auto& boundary_correction) noexcept {
        using mortar_tags_list = typename std::decay_t<decltype(
            boundary_correction)>::dg_package_field_tags;

        Variables<db::wrap_tags_in<::Tags::dt, variables_tags>>
            dt_boundary_correction_on_mortar{};
        Variables<db::wrap_tags_in<::Tags::dt, variables_tags>>
            dt_boundary_correction_projected_onto_face{};

        // Iterate over all mortars
        for (auto& [mortar_id, mortar_data] : *mortar_data_ptr) {
          const auto& direction = mortar_id.first;
          const size_t dimension = direction.dimension();
          if (UNLIKELY(mortar_id.second ==
                       ElementId<volume_dim>::external_boundary_id())) {
            ERROR(
                "Cannot impose boundary conditions on external boundary in "
                "direction "
                << direction
                << " in the ApplyBoundaryCorrections action. Boundary "
                   "conditions are applied in the ComputeTimeDerivative action "
                   "instead. You may have unintentionally added external "
                   "mortars in one of the initialization actions.");
          }

          const Mesh<volume_dim - 1>& mortar_mesh = mortar_meshes.at(mortar_id);

          // Extract local and neighbor data, copy into Variables because we
          // store them in a std::vector for type erasure.
          const auto extracted_mortar_data = mortar_data.extract();
          const auto& [local_mesh_and_data, neighbor_mesh_and_data] =
              extracted_mortar_data;
          Variables<mortar_tags_list> local_data_on_mortar{
              mortar_mesh.number_of_grid_points()};
          Variables<mortar_tags_list> neighbor_data_on_mortar{
              mortar_mesh.number_of_grid_points()};
          std::copy(local_mesh_and_data.second.begin(),
                    local_mesh_and_data.second.end(),
                    local_data_on_mortar.data());
          std::copy(neighbor_mesh_and_data.second.begin(),
                    neighbor_mesh_and_data.second.end(),
                    neighbor_data_on_mortar.data());

          // The boundary computations and lifting can be further optimized by
          // in the h-refinement case having only one allocation for the face
          // and having the projection from the mortar to the face be done in
          // place. E.g. local_data_on_mortar and neighbor_data_on_mortar could
          // be allocated fewer times, as well as `needs_projection` section
          // below could do an in-place projection.
          if (dt_boundary_correction_on_mortar.number_of_grid_points() !=
              mortar_mesh.number_of_grid_points()) {
            dt_boundary_correction_on_mortar.initialize(
                mortar_mesh.number_of_grid_points());
          }

          detail::boundary_correction(
              make_not_null(&dt_boundary_correction_on_mortar),
              local_data_on_mortar, neighbor_data_on_mortar,
              boundary_correction, dg_formulation);

          const std::array<Spectral::MortarSize, volume_dim - 1>& mortar_size =
              mortar_sizes.at(mortar_id);
          const Mesh<volume_dim - 1> face_mesh =
              volume_mesh.slice_away(dimension);

          auto& dt_boundary_correction =
              [&dt_boundary_correction_on_mortar,
               &dt_boundary_correction_projected_onto_face, &face_mesh,
               &mortar_mesh, &mortar_size]() noexcept
              -> Variables<db::wrap_tags_in<::Tags::dt, variables_tags>>& {
            if (::dg::needs_projection(face_mesh, mortar_mesh, mortar_size)) {
              dt_boundary_correction_projected_onto_face =
                  ::dg::project_from_mortar(dt_boundary_correction_on_mortar,
                                            face_mesh, mortar_mesh,
                                            mortar_size);
              return dt_boundary_correction_projected_onto_face;
            }
            return dt_boundary_correction_on_mortar;
          }();

          ASSERT(
              face_normal_covector_and_magnitude.count(direction) == 1 and
                  face_normal_covector_and_magnitude.at(direction).has_value(),
              "Face normal covector and magnitude not set in direction: "
                  << direction);
          const auto& magnitude_of_face_normal =
              get<evolution::dg::Tags::MagnitudeOfNormal>(
                  *face_normal_covector_and_magnitude.at(direction));
          if (using_gauss_lobatto_points) {
            // The lift_flux function lifts only on the slice, it does not add
            // the contribution to the volume.
            ::dg::lift_flux(make_not_null(&dt_boundary_correction),
                            volume_mesh.extents(dimension),
                            magnitude_of_face_normal);

            // Add the flux contribution to the volume data
            add_slice_to_data(
                dt_variables_ptr, dt_boundary_correction, volume_mesh.extents(),
                dimension, index_to_slice_at(volume_mesh.extents(), direction));
          } else {
            // We are using Gauss points.
            //
            // Notes:
            // - We should really lift both sides simultaneously since this
            //   reduces memory accesses. Lifting all sides at the same time is
            //   unlikely to improve performance since we lift by jumping
            //   through slices.
            // - If we lift both sides at the same time we first need to deal
            //   with projecting from mortars to the face, then lift off the
            //   faces. With non-owning Variables memory allocations could be
            //   significantly reduced in this code.

            // Project the determinant of the Jacobian to the face. This could
            // be optimized by caching in the time-independent case.
            Scalar<DataVector> face_det_jacobian{
                face_mesh.number_of_grid_points()};
            const Matrix identity{};
            auto interpolation_matrices =
                make_array<volume_dim>(std::cref(identity));
            const std::pair<Matrix, Matrix>& matrices =
                Spectral::boundary_interpolation_matrices(
                    volume_mesh.slice_through(direction.dimension()));
            gsl::at(interpolation_matrices, direction.dimension()) =
                direction.side() == Side::Upper ? matrices.second
                                                : matrices.first;
            apply_matrices(make_not_null(&get(face_det_jacobian)),
                           interpolation_matrices, volume_det_jacobian,
                           volume_mesh.extents());

            evolution::dg::lift_boundary_terms_gauss_points(
                dt_variables_ptr, volume_det_inv_jacobian, volume_mesh,
                direction, dt_boundary_correction, magnitude_of_face_normal,
                face_det_jacobian);
          }
        }
      };

  // Now compute the boundary contribution to this element using the helper
  // lambda
  const auto& boundary_correction = db::get<
      evolution::Tags::BoundaryCorrection<typename Metavariables::system>>(
      *box);
  using derived_boundary_corrections =
      typename std::decay_t<decltype(boundary_correction)>::creatable_classes;

  static_assert(
      tmpl::all<derived_boundary_corrections, std::is_final<tmpl::_1>>::value,
      "All createable classes for boundary corrections must be marked "
      "final.");
  tmpl::for_each<derived_boundary_corrections>(
      [&boundary_correction, &box, &compute_and_lift_boundary_corrections](
          auto derived_correction_v) noexcept {
        using DerivedCorrection =
            tmpl::type_from<decltype(derived_correction_v)>;
        if (typeid(boundary_correction) == typeid(DerivedCorrection)) {
          // Compute internal boundary quantities on the mortar for sides of
          // the element that have neighbors, i.e. they are not an external
          // side.
          db::mutate<dt_variables_tag,
                     evolution::dg::Tags::MortarData<volume_dim>>(
              box, compute_and_lift_boundary_corrections,
              dynamic_cast<const DerivedCorrection&>(boundary_correction));
        }
      });
}

template <typename Metavariables>
template <typename DbTags, typename... InboxTags, typename ArrayIndex>
bool ApplyBoundaryCorrections<Metavariables>::is_ready(
    const db::DataBox<DbTags>& box,
    const tuples::TaggedTuple<InboxTags...>& inboxes,
    const Parallel::GlobalCache<Metavariables>& /*cache*/,
    const ArrayIndex& /*array_index*/) noexcept {
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  const Element<volume_dim>& element =
      db::get<domain::Tags::Element<volume_dim>>(box);

  if (UNLIKELY(element.number_of_neighbors() == 0)) {
    return true;
  }

  if constexpr (not Metavariables::local_time_stepping) {
    const TimeStepId& temporal_id = get<::Tags::TimeStepId>(box);
    const auto& inbox =
        tuples::get<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
            Metavariables::volume_dim>>(inboxes);
    const auto received_temporal_id_and_data = inbox.find(temporal_id);
    if (received_temporal_id_and_data == inbox.end()) {
      return false;
    }
    const auto& received_neighbor_data = received_temporal_id_and_data->second;
    for (const auto& [direction, neighbors] : element.neighbors()) {
      for (const auto& neighbor : neighbors) {
        const auto neighbor_received =
            received_neighbor_data.find(std::pair{direction, neighbor});
        if (neighbor_received == received_neighbor_data.end()) {
          return false;
        }
      }
    }
  } else {
    const auto& inbox =
        tuples::get<evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
            Metavariables::volume_dim>>(inboxes);

    const auto& local_next_temporal_id =
        db::get<::Tags::Next<::Tags::TimeStepId>>(box);
    const auto& mortars_next_temporal_id = db::get<
        evolution::dg::Tags::MortarNextTemporalId<Metavariables::volume_dim>>(
        box);

    for (const auto& [mortar_id, mortar_next_temporal_id] :
         mortars_next_temporal_id) {
      // If on an external boundary
      if (UNLIKELY(mortar_id.second ==
                   ElementId<volume_dim>::external_boundary_id())) {
        continue;
      }
      auto next_temporal_id = mortar_next_temporal_id;
      while (next_temporal_id < local_next_temporal_id) {
        const auto temporal_received = inbox.find(next_temporal_id);
        if (temporal_received == inbox.end()) {
          return false;
        }
        const auto mortar_received = temporal_received->second.find(mortar_id);
        if (mortar_received == temporal_received->second.end()) {
          return false;
        }
        next_temporal_id = std::get<3>(mortar_received->second);
      }
    }
  }
  return true;
}
}  // namespace evolution::dg::Actions
