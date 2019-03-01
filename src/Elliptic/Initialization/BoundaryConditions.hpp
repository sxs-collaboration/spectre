// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
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
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace Elliptic {
namespace Initialization {

/*!
 * \brief Adds boundary contributions to the sources
 *
 * Imposes boundary conditions by adding contributions to the sources of the
 * elliptic equations. With the source modified, we may then assume homogeneous
 * (i.e. zero) Dirichlet boundary conditions throughout the elliptic solve.
 *
 * \note Only Dirichlet boundary conditions retrieved from the analytic solution
 * are currently supported.
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
template <typename Metavariables>
struct BoundaryConditions {
  using system = typename Metavariables::system;

  using sources_tag =
      db::add_tag_prefix<Tags::Source, typename system::fields_tag>;

  using simple_tags = db::AddSimpleTags<>;
  using compute_tags = db::AddComputeTags<>;

  template <typename NormalDotNumericalFluxComputer,
            typename... NumericalFluxTags, typename... BoundaryDataTags>
  static void compute_dirichlet_boundary_normal_dot_numerical_flux(
      const gsl::not_null<Variables<tmpl::list<NumericalFluxTags...>>*>
          numerical_fluxes,
      const NormalDotNumericalFluxComputer& normal_dot_numerical_flux_computer,
      const Variables<tmpl::list<BoundaryDataTags...>>& boundary_data,
      const tnsr::i<DataVector, system::volume_dim, Frame::Inertial>&
          normalized_face_normal) noexcept {
    normal_dot_numerical_flux_computer.compute_dirichlet_boundary(
        make_not_null(&get<NumericalFluxTags>(*numerical_fluxes))...,
        get<BoundaryDataTags>(boundary_data)..., normalized_face_normal);
  }

  template <typename TagsList>
  static auto initialize(
      db::DataBox<TagsList>&& box,
      const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
    static constexpr const size_t volume_dim = system::volume_dim;

    const auto& analytic_solution =
        Parallel::get<typename Metavariables::analytic_solution_tag>(cache);
    const auto& normal_dot_numerical_flux_computer =
        Parallel::get<typename Metavariables::normal_dot_numerical_flux>(cache);

    db::mutate<sources_tag>(
        make_not_null(&box),
        [
          &analytic_solution, &normal_dot_numerical_flux_computer
        ](const gsl::not_null<db::item_type<sources_tag>*> sources,
          const Mesh<volume_dim>& mesh,
          const db::item_type<Tags::BoundaryDirectionsInterior<volume_dim>>&
              boundary_directions,
          const db::item_type<
              Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>,
                              Tags::Coordinates<volume_dim, Frame::Inertial>>>&
              boundary_coordinates,
          const db::item_type<Tags::Interface<
              Tags::BoundaryDirectionsInterior<volume_dim>,
              Tags::Normalized<Tags::UnnormalizedFaceNormal<volume_dim>>>>&
              normalized_face_normals,
          const db::item_type<Tags::Interface<
              Tags::BoundaryDirectionsInterior<volume_dim>,
              Tags::Magnitude<Tags::UnnormalizedFaceNormal<volume_dim>>>>&
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
                dg::lift_flux(std::move(boundary_normal_dot_numerical_fluxes),
                              mesh.extents(dimension),
                              magnitude_of_face_normals.at(direction))};
            add_slice_to_data(sources, std::move(lifted_boundary_data),
                              mesh.extents(), dimension,
                              index_to_slice_at(mesh.extents(), direction));
          }
        },
        get<Tags::Mesh<volume_dim>>(box),
        get<Tags::BoundaryDirectionsInterior<volume_dim>>(box),
        get<Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>,
                            Tags::Coordinates<volume_dim, Frame::Inertial>>>(
            box),
        get<Tags::Interface<
            Tags::BoundaryDirectionsInterior<volume_dim>,
            Tags::Normalized<Tags::UnnormalizedFaceNormal<volume_dim>>>>(box),
        get<Tags::Interface<
            Tags::BoundaryDirectionsInterior<volume_dim>,
            Tags::Magnitude<Tags::UnnormalizedFaceNormal<volume_dim>>>>(box));

    return std::move(box);
  }
};
}  // namespace Initialization
}  // namespace Elliptic
