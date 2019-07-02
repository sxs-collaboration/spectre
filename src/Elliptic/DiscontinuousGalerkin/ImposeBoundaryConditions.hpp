// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InterfaceActionHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare db::DataBox

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace elliptic {
namespace dg {
namespace Actions {

/*!
 * \brief Adds Dirichlet boundary contributions to the sources
 *
 * Imposes Dirichlet boundary conditions by adding contributions to the sources
 * of the elliptic equations. With the source modified, we may then assume
 * homogeneous (i.e. zero) Dirichlet boundary conditions throughout the elliptic
 * solve.
 *
 * \note Only Dirichlet boundary conditions retrieved from the analytic solution
 * are currently supported.
 *
 * \warning Call this action during the initialization of a linear solve, not in
 * each of its iterations.
 *
 * With:
 * - `sources_tag` = `db::add_tag_prefix<Tags::Source, system::fields_tag>`
 *
 * Uses:
 * - Metavariables:
 *   - `analytic_solution_tag`
 *   - `normal_dot_numerical_flux`
 * - System:
 *   - `volume_dim`
 *   - `fields_tag`
 *   - `impose_boundary_conditions_on_fields`
 * - DataBox:
 *   - `Tags::Mesh<volume_dim>`
 *   - `Tags::Coordinates<volume_dim, Frame::Inertial>`
 *   - `Tags::BoundaryDirectionsInterior<volume_dim>`
 *   - `Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>,
 *   Tags::Coordinates<volume_dim, Frame::Inertial>>`
 *   - `Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>,
 *   Tags::Normalized<Tags::UnnormalizedFaceNormal<volume_dim>>>`
 *   - `Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>,
 *   Tags::Magnitude<Tags::UnnormalizedFaceNormal<volume_dim>>>`
 *
 * DataBox:
 * - Modifies:
 *   - `sources_tag`
 */
struct ImposeInhomogeneousBoundaryConditionsOnSource {
 private:
  template <typename NormalDotNumericalFluxComputer,
            typename... NumericalFluxTags, typename... BoundaryDataTags,
            size_t Dim>
  static void compute_dirichlet_boundary_normal_dot_numerical_flux(
      const gsl::not_null<Variables<tmpl::list<NumericalFluxTags...>>*>
          numerical_fluxes,
      const NormalDotNumericalFluxComputer& normal_dot_numerical_flux_computer,
      const Variables<tmpl::list<BoundaryDataTags...>>& boundary_data,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normalized_face_normal) noexcept {
    normal_dot_numerical_flux_computer.compute_dirichlet_boundary(
        make_not_null(&get<NumericalFluxTags>(*numerical_fluxes))...,
        get<BoundaryDataTags>(boundary_data)..., normalized_face_normal);
  }

 public:
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using sources_tag =
        db::add_tag_prefix<Tags::Source,
                           typename Metavariables::system::fields_tag>;

    const auto& analytic_solution =
        Parallel::get<typename Metavariables::analytic_solution_tag>(cache);
    const auto& normal_dot_numerical_flux_computer =
        Parallel::get<typename Metavariables::normal_dot_numerical_flux>(cache);

    db::mutate<sources_tag>(
        make_not_null(&box),
        [&analytic_solution, &normal_dot_numerical_flux_computer ](
            const gsl::not_null<db::item_type<sources_tag>*> sources,
            const Mesh<Dim>& mesh,
            const db::item_type<Tags::BoundaryDirectionsInterior<Dim>>&
                boundary_directions,
            const db::item_type<Tags::Interface<
                Tags::BoundaryDirectionsInterior<Dim>,
                Tags::Coordinates<Dim, Frame::Inertial>>>& boundary_coordinates,
            const db::item_type<Tags::Interface<
                Tags::BoundaryDirectionsInterior<Dim>,
                Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>>&
                normalized_face_normals,
            const db::item_type<Tags::Interface<
                Tags::BoundaryDirectionsInterior<Dim>,
                Tags::Magnitude<Tags::UnnormalizedFaceNormal<Dim>>>>&
                magnitude_of_face_normals) noexcept {
          // Impose Dirichlet boundary conditions as contributions to the source
          for (const auto& direction : boundary_directions) {
            const size_t dimension = direction.dimension();
            const auto mortar_mesh = mesh.slice_away(dimension);
            // Compute Dirichlet data on mortar
            Variables<typename system::impose_boundary_conditions_on_fields>
                dirichlet_boundary_data{mortar_mesh.number_of_grid_points(),
                                        0.};
            dirichlet_boundary_data.assign_subset(analytic_solution.variables(
                boundary_coordinates.at(direction),
                typename system::impose_boundary_conditions_on_fields{}));
            // Compute the numerical flux contribution from the Dirichlet data
            db::item_type<
                db::add_tag_prefix<Tags::NormalDotNumericalFlux, sources_tag>>
                boundary_normal_dot_numerical_fluxes{
                    mortar_mesh.number_of_grid_points(), 0.};
            compute_dirichlet_boundary_normal_dot_numerical_flux(
                make_not_null(&boundary_normal_dot_numerical_fluxes),
                normal_dot_numerical_flux_computer,
                std::move(dirichlet_boundary_data),
                normalized_face_normals.at(direction));
            // Flip sign of the boundary contributions, making them
            // contributions to the source
            db::item_type<sources_tag> lifted_boundary_data{
                -1. *
                ::dg::lift_flux(std::move(boundary_normal_dot_numerical_fluxes),
                                mesh.extents(dimension),
                                magnitude_of_face_normals.at(direction))};
            add_slice_to_data(sources, std::move(lifted_boundary_data),
                              mesh.extents(), dimension,
                              index_to_slice_at(mesh.extents(), direction));
          }
        },
        get<Tags::Mesh<Dim>>(box),
        get<Tags::BoundaryDirectionsInterior<Dim>>(box),
        get<Tags::Interface<Tags::BoundaryDirectionsInterior<Dim>,
                            Tags::Coordinates<Dim, Frame::Inertial>>>(box),
        get<Tags::Interface<
            Tags::BoundaryDirectionsInterior<Dim>,
            Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>>(box),
        get<Tags::Interface<
            Tags::BoundaryDirectionsInterior<Dim>,
            Tags::Magnitude<Tags::UnnormalizedFaceNormal<Dim>>>>(box));

    return std::forward_as_tuple(std::move(box));
  }
};

/*!
 * \brief Packages data on external boundaries so that they represent
 * homogeneous (zero) Dirichlet boundary conditions.
 *
 * This action imposes homogeneous boundary conditions on all fields in
 * `system::impose_boundary_conditions_on_fields`. The fields are wrapped in
 * `LinearSolver::Tags::Operand`. The result should be a subset of the
 * `system::variables`. Because we are working with the linear solver operand,
 * we cannot impose non-zero boundary conditions here. Instead, non-zero
 * boundary conditions are handled as contributions to the linear solver source
 * during initialization.
 *
 * \warning This actions works only for scalar fields right now. It should be
 * considered a temporary solution and will have to be reworked for more
 * involved boundary conditions.
 *
 * With:
 * - `interior<Tag> =
 *   Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>, Tag>`
 * - `exterior<Tag> =
 *   Tags::Interface<Tags::BoundaryDirectionsExterior<volume_dim>, Tag>`
 *
 * Uses:
 * - Metavariables:
 *   - `normal_dot_numerical_flux`
 *   - `temporal_id`
 * - System:
 *   - `volume_dim`
 *   - `variables_tag`
 *   - `impose_boundary_conditions_on_fields`
 * - ConstGlobalCache:
 *   - `normal_dot_numerical_flux`
 * - DataBox:
 *   - `Tags::Element<volume_dim>`
 *   - `temporal_id`
 *   - `Tags::BoundaryDirectionsInterior<volume_dim>`
 *   - `Tags::BoundaryDirectionsExterior<volume_dim>`
 *   - `interior<variables_tag>`
 *   - `exterior<variables_tag>`
 *   - `interior<normal_dot_numerical_flux::type::argument_tags>`
 *   - `exterior<normal_dot_numerical_flux::type::argument_tags>`
 *
 * DataBox changes:
 * - Modifies:
 *   - `exterior<variables_tag>`
 *   - `Tags::VariablesBoundaryData`
 */
struct ImposeHomogeneousDirichletBoundaryConditions {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTags>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using dirichlet_tags =
        typename system::impose_boundary_conditions_on_fields;
    constexpr size_t volume_dim = system::volume_dim;

    // Set the data on exterior (ghost) faces to impose the boundary conditions
    db::mutate<Tags::Interface<Tags::BoundaryDirectionsExterior<volume_dim>,
                               typename system::variables_tag>>(
        make_not_null(&box),
        // Need to use system::volume_dim below instead of just
        // volume_dim to avoid an ICE on gcc 7.
        [](const gsl::not_null<db::item_type<Tags::Interface<
               Tags::BoundaryDirectionsExterior<system::volume_dim>,
               typename system::variables_tag>>*>
               exterior_boundary_vars,
           const db::item_type<Tags::Interface<
               Tags::BoundaryDirectionsInterior<system::volume_dim>,
               typename system::variables_tag>>& interior_vars) noexcept {
          for (auto& exterior_direction_and_vars : *exterior_boundary_vars) {
            auto& direction = exterior_direction_and_vars.first;
            auto& exterior_vars = exterior_direction_and_vars.second;

            // By default, use the variables on the external boundary for the
            // exterior
            exterior_vars = interior_vars.at(direction);

            // For those variables where we have boundary conditions, impose
            // zero Dirichlet b.c. here. The non-zero boundary conditions are
            // handled as contributions to the source during initialization.
            // Imposing them here would not work because we are working with the
            // linear solver operand.
            tmpl::for_each<dirichlet_tags>([
              &interior_vars, &exterior_vars, &direction
            ](auto dirichlet_tag_val) noexcept {
              using dirichlet_tag =
                  tmpl::type_from<decltype(dirichlet_tag_val)>;
              using dirichlet_operand_tag =
                  LinearSolver::Tags::Operand<dirichlet_tag>;
              // Use mirror principle. This only works for scalars right now.
              get(get<dirichlet_operand_tag>(exterior_vars)) =
                  -1. *
                  get<dirichlet_operand_tag>(interior_vars.at(direction)).get();
            });
          }
        },
        get<Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>,
                            typename system::variables_tag>>(box));

    const auto& element = db::get<Tags::Element<volume_dim>>(box);
    const auto& temporal_id = db::get<typename Metavariables::temporal_id>(box);

    const auto& normal_dot_numerical_flux_computer =
        get<typename Metavariables::normal_dot_numerical_flux>(cache);

    // Store local and packaged data on the mortars
    for (const auto& direction : element.external_boundaries()) {
      const auto mortar_id = std::make_pair(
          direction, ElementId<volume_dim>::external_boundary_id());

      auto interior_data = DgActions_detail::compute_local_mortar_data(
          box, direction, normal_dot_numerical_flux_computer,
          Tags::BoundaryDirectionsInterior<volume_dim>{}, Metavariables{});

      auto exterior_data = DgActions_detail::compute_packaged_data(
          box, direction, normal_dot_numerical_flux_computer,
          Tags::BoundaryDirectionsExterior<volume_dim>{}, Metavariables{});

      db::mutate<Tags::VariablesBoundaryData>(
          make_not_null(&box),
          [&mortar_id, &temporal_id, &interior_data,
           &exterior_data ](const gsl::not_null<
                            db::item_type<Tags::VariablesBoundaryData, DbTags>*>
                                mortar_data) noexcept {
            mortar_data->at(mortar_id).local_insert(temporal_id,
                                                    std::move(interior_data));
            mortar_data->at(mortar_id).remote_insert(temporal_id,
                                                     std::move(exterior_data));
          });
    }
    return std::forward_as_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace dg
}  // namespace elliptic
