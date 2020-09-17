// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/SurfaceJacobian.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace elliptic {
namespace dg {
namespace Actions {

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
 * - `fixed_sources_tag` = `db::add_tag_prefix<Tags::FixedSource, fields_tag>`
 *
 * Uses:
 * - Metavariables:
 *   - `analytic_solution_tag`
 *   - `normal_dot_numerical_flux`
 * - System:
 *   - `volume_dim`
 *   - `fields_tag`
 *   - `primal_fields`
 * - DataBox:
 *   - `Tags::Mesh<volume_dim>`
 *   - `Tags::Coordinates<volume_dim, Frame::Inertial>`
 *   - `Tags::BoundaryDirectionsInterior<volume_dim>`
 *   - `Tags::Interface<Tags::BoundaryDirectionsExterior<volume_dim>,
 *   Tags::Coordinates<volume_dim, Frame::Inertial>>`
 *   - `Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>,
 *   Tags::Normalized<Tags::UnnormalizedFaceNormal<volume_dim>>>`
 *   - `Tags::Interface<Tags::BoundaryDirectionsInterior<volume_dim>,
 *   Tags::Magnitude<Tags::UnnormalizedFaceNormal<volume_dim>>>`
 *
 * DataBox:
 * - Modifies:
 *   - `fixed_sources_tag`
 */
template <typename Metavariables>
struct ImposeInhomogeneousBoundaryConditionsOnSource {
  using system = typename Metavariables::system;
  static constexpr size_t volume_dim = system::volume_dim;
  using FluxesType = typename system::fluxes;
  using fluxes_computer_tag = elliptic::Tags::FluxesComputer<FluxesType>;

  using fixed_sources_tag =
      db::add_tag_prefix<::Tags::FixedSource, typename system::fields_tag>;

  template <typename NormalDotNumericalFluxComputer,
            typename... NumericalFluxTags, typename... BoundaryDataTags,
            typename... FluxesArgs>
  static void compute_dirichlet_boundary_normal_dot_numerical_flux(
      const gsl::not_null<Variables<tmpl::list<NumericalFluxTags...>>*>
          numerical_fluxes,
      const NormalDotNumericalFluxComputer& normal_dot_numerical_flux_computer,
      const Variables<tmpl::list<BoundaryDataTags...>>& boundary_data,
      const Mesh<volume_dim>& volume_mesh,
      const Direction<volume_dim>& direction,
      const tnsr::i<DataVector, volume_dim, Frame::Inertial>&
          normalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
      const FluxesType& fluxes_computer,
      const FluxesArgs&... fluxes_args) noexcept {
    normal_dot_numerical_flux_computer.compute_dirichlet_boundary(
        make_not_null(&get<NumericalFluxTags>(*numerical_fluxes))...,
        get<BoundaryDataTags>(boundary_data)..., volume_mesh, direction,
        normalized_face_normal, face_normal_magnitude, fluxes_computer,
        fluxes_args...);
  }

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& analytic_solution =
        Parallel::get<typename Metavariables::analytic_solution_tag>(cache);
    const auto& normal_dot_numerical_flux_computer =
        Parallel::get<typename Metavariables::normal_dot_numerical_flux>(cache);

    // Compute the contribution to the fixed sources from all Dirichlet
    // boundaries
    auto dirichlet_boundary_contributions = interface_apply<
        domain::Tags::BoundaryDirectionsExterior<volume_dim>,
        tmpl::push_front<
            typename FluxesType::argument_tags, domain::Tags::Mesh<volume_dim>,
            domain::Tags::Mesh<volume_dim - 1>,
            domain::Tags::Direction<volume_dim>,
            domain::Tags::Coordinates<volume_dim, Frame::Inertial>,
            ::Tags::Normalized<
                domain::Tags::UnnormalizedFaceNormal<volume_dim>>,
            ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<volume_dim>>,
            domain::Tags::SurfaceJacobian<Frame::Logical, Frame::Inertial>,
            fluxes_computer_tag>,
        tmpl::push_front<get_volume_tags<FluxesType>,
                         domain::Tags::Mesh<volume_dim>, fluxes_computer_tag>>(
        [&analytic_solution, &normal_dot_numerical_flux_computer](
            const Mesh<volume_dim>& volume_mesh,
            const Mesh<volume_dim - 1>& face_mesh,
            const Direction<volume_dim>& direction,
            const tnsr::I<DataVector, volume_dim, Frame::Inertial>&
                boundary_coordinates,
            tnsr::i<DataVector, volume_dim> normalized_face_normal,
            const Scalar<DataVector>& magnitude_of_face_normal,
            const Scalar<DataVector>& surface_jacobian,
            const FluxesType& fluxes_computer, const auto&... fluxes_args) {
          const size_t dimension = direction.dimension();
          // We get the exterior face normal out of the box, so we flip its sign
          // to obtain the interior face normal
          for (size_t d = 0; d < volume_dim; d++) {
            normalized_face_normal.get(d) *= -1.;
          }
          // Compute Dirichlet data on mortar
          Variables<typename system::primal_fields> dirichlet_boundary_data{
              face_mesh.number_of_grid_points(), 0.};
          dirichlet_boundary_data.assign_subset(analytic_solution.variables(
              boundary_coordinates, typename system::primal_fields{}));
          // Compute the numerical flux contribution from the Dirichlet data
          typename db::add_tag_prefix<::Tags::NormalDotNumericalFlux,
                                      fixed_sources_tag>::type
              boundary_normal_dot_numerical_fluxes{
                  face_mesh.number_of_grid_points(), 0.};
          compute_dirichlet_boundary_normal_dot_numerical_flux(
              make_not_null(&boundary_normal_dot_numerical_fluxes),
              normal_dot_numerical_flux_computer,
              std::move(dirichlet_boundary_data), volume_mesh, direction,
              normalized_face_normal, magnitude_of_face_normal, fluxes_computer,
              fluxes_args...);
          // Flip sign of the boundary contributions, making them
          // contributions to the source
          if constexpr (Metavariables::massive_operator) {
            return typename fixed_sources_tag::type{
                -1. * ::dg::lift_flux_massive_no_mass_lumping(
                          std::move(boundary_normal_dot_numerical_fluxes),
                          face_mesh, surface_jacobian)};
          } else {
            return typename fixed_sources_tag::type{
                -1. *
                ::dg::lift_flux(std::move(boundary_normal_dot_numerical_fluxes),
                                volume_mesh.extents(dimension),
                                magnitude_of_face_normal)};
          }
        },
        box);

    // Add the boundary contributions to the fixed sources
    db::mutate<fixed_sources_tag>(
        make_not_null(&box),
        [&dirichlet_boundary_contributions](
            const gsl::not_null<typename fixed_sources_tag::type*>
                fixed_sources,
            const Mesh<volume_dim>& mesh,
            const std::unordered_set<::Direction<volume_dim>>&
                directions) noexcept {
          for (const auto& direction : directions) {
            add_slice_to_data(
                fixed_sources,
                std::move(dirichlet_boundary_contributions.at(direction)),
                mesh.extents(), direction.dimension(),
                index_to_slice_at(mesh.extents(), direction));
          }
        },
        get<domain::Tags::Mesh<volume_dim>>(box),
        get<domain::Tags::BoundaryDirectionsInterior<volume_dim>>(box));

    return {std::move(box)};
  }
};
}  // namespace Actions
}  // namespace dg
}  // namespace elliptic
