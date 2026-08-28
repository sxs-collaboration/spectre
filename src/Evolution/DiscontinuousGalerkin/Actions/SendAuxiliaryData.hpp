// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/BoundaryConditions/Cartoon.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/BoundaryConditionsImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/UsingSubcell.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Time/LtsMode.hpp"
#include "Time/Tags/LtsMode.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace evolution::dg::Actions {
namespace detail {
// Per-boundary-condition tags projected from the volume temporaries
// (`dg_interior_temporary_tags`). The inertial coordinates are exempt: the
// boundary-condition path computes them on the face directly from the
// coordinate map instead of projecting them from the volume temporaries.
template <typename DimType, typename BoundaryCondition>
struct boundary_condition_volume_temporary_tags {
  using type =
      tmpl::remove<typename BoundaryCondition::dg_interior_temporary_tags,
                   domain::Tags::Coordinates<DimType::value, Frame::Inertial>>;
};

// Per-boundary-condition tags projected from the volume partial derivatives
// (`dg_interior_deriv_vars_tags`, optional) and from the volume time
// derivative (`dg_interior_dt_vars_tags`, optional).
template <typename BoundaryCondition>
struct boundary_condition_deriv_tags {
  using type = get_deriv_vars_from_boundary_condition<BoundaryCondition>;
};
template <typename BoundaryCondition>
struct boundary_condition_dt_tags {
  using type = get_dt_vars_from_boundary_condition<BoundaryCondition>;
};

// Verify that the LDG auxiliary send reads no uninitialized face data, i.e. it
// never projects any of the volume time-derivative buffers (volume
// temporaries, fluxes, partial derivatives), the inverse spatial metric, or
// the volume time derivative to the boundary.
template <typename EvolutionSystem, typename Metavariables>
struct auxiliary_send_projects_no_uninitialized_data_to_face {
 private:
  using derived_boundary_corrections =
      tmpl::at<typename Metavariables::factory_creation::factory_classes,
               evolution::BoundaryCorrection>;

  // The Cartoon/None/Periodic marker conditions are handled specially and do
  // not declare the interior-data interface (e.g. dg_interior_temporary_tags),
  // so they are excluded before inspecting that interface.
  using derived_boundary_conditions = tmpl::remove_if<
      tmpl::at<typename Metavariables::factory_creation::factory_classes,
               typename EvolutionSystem::boundary_conditions_base>,
      tmpl::or_<
          std::is_base_of<domain::BoundaryConditions::MarkAsCartoon, tmpl::_1>,
          std::is_base_of<domain::BoundaryConditions::MarkAsNone, tmpl::_1>,
          std::is_base_of<domain::BoundaryConditions::MarkAsPeriodic,
                          tmpl::_1>>>;

  // Volume temporaries the boundary corrections would need on the face, from
  // either packaging interface.
  using all_correction_temporary_tags = tmpl::flatten<tmpl::append<
      tmpl::transform<derived_boundary_corrections,
                      get_dg_auxiliary_package_temporary_tags<tmpl::_1>>,
      tmpl::transform<derived_boundary_corrections,
                      get_dg_package_temporary_tags<tmpl::_1>>>>;
  // Volume temporaries the boundary conditions would project.
  using all_condition_volume_temporary_tags = tmpl::flatten<tmpl::transform<
      derived_boundary_conditions,
      boundary_condition_volume_temporary_tags<
          tmpl::pin<tmpl::size_t<EvolutionSystem::volume_dim>>, tmpl::_1>>>;
  // Partial derivatives the boundary conditions would project.
  using all_condition_deriv_tags =
      tmpl::flatten<tmpl::transform<derived_boundary_conditions,
                                    boundary_condition_deriv_tags<tmpl::_1>>>;
  // Volume time derivatives the boundary conditions would project.
  using all_condition_dt_tags =
      tmpl::flatten<tmpl::transform<derived_boundary_conditions,
                                    boundary_condition_dt_tags<tmpl::_1>>>;

 public:
  static constexpr bool value =
      // (1) No fluxes are projected (system is flux-free).
      tmpl::size<typename EvolutionSystem::flux_variables>::value == 0 and
      // (2) The system has no inverse spatial metric to project.
      not has_inverse_spatial_metric_tag_v<EvolutionSystem> and
      // (3) No boundary correction declares temporaries in either packaging
      //     interface.
      tmpl::size<all_correction_temporary_tags>::value == 0 and
      // (4) No boundary condition projects volume temporaries.
      tmpl::size<all_condition_volume_temporary_tags>::value == 0 and
      // (5) No boundary condition projects partial derivatives.
      tmpl::size<all_condition_deriv_tags>::value == 0 and
      // (6) No boundary condition projects the volume time derivative.
      tmpl::size<all_condition_dt_tags>::value == 0;
};

template <typename EvolutionSystem, typename Metavariables>
constexpr bool auxiliary_send_projects_no_uninitialized_data_to_face_v =
    auxiliary_send_projects_no_uninitialized_data_to_face<EvolutionSystem,
                                                          Metavariables>::value;
}  // namespace detail

/*!
 * \brief Packages and sends the auxiliary-pass data for the local
 * discontinuous Galerkin (LDG) method for auxiliary boundary corrections.
 *
 * This action performs the *send* half of the LDG auxiliary communication. It
 * mirrors the boundary-data portion of `ComputeTimeDerivative::apply`
 * (without the time-derivative computation) but uses the auxiliary package-data
 * interface (`dg_auxiliary_package_field_tags`, `dg_auxiliary_package_data`,
 * `dg_auxiliary_package_data_volume_tags`) and sends to the auxiliary inbox
 * channel `evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox` with
 * `IsAuxiliary` set to `true`.
 *
 * The action does NOT compute the volume time derivative, fluxes, flux
 * divergence, or partial derivatives. It only:
 *
 * 1. Computes the internal mortar data in auxiliary mode
 *    (`detail::internal_mortar_data` with `ComputeAuxiliary` set to `true`).
 * 2. Applies the auxiliary external boundary conditions
 *    (`detail::apply_boundary_conditions_on_all_external_faces` with
 *    `ComputeAuxiliary` set to `true`).
 * 3. Sends the packaged auxiliary mortar data to the neighbors on the auxiliary
 *    inbox channel by the shared implementation
 *    (`ComputeTimeDerivative_detail::Impl` with `IsAuxiliary` set to `true`).
 *
 * Only pure-DG executables (no DG-FD subcell) and global time stepping are
 * supported for now. The subcell restriction is enforced by a `static_assert`;
 * global time stepping is enforced by a runtime error unless `Tags::LtsMode`
 * is `LtsMode::Off`. Nonconforming meshes flow through the same shared mortar
 * machinery as the physical pass, but are not yet exercised by tests; until
 * they are, a runtime error rejects elements with nonconforming neighbors.
 */
template <size_t Dim, typename EvolutionSystem, bool UseNodegroupDgElements,
          typename VariablesTag = typename EvolutionSystem::variables_tag>
struct SendAuxiliaryData
    : ComputeTimeDerivative_detail::Impl<Dim, EvolutionSystem, tmpl::list<>,
                                         UseNodegroupDgElements, true,
                                         VariablesTag> {
 private:
  using base =
      ComputeTimeDerivative_detail::Impl<Dim, EvolutionSystem, tmpl::list<>,
                                         UseNodegroupDgElements, true,
                                         VariablesTag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList action_list_meta,
      const ParallelComponent* const parallel_component_meta) {
    static_assert(not evolution::dg::using_subcell_v<Metavariables>,
                  "The LDG auxiliary pass does not support executables using "
                  "DG-subcell (the DG-FD hybrid scheme) yet.");
    static_assert(not EvolutionSystem::has_primitive_and_conservative_vars,
                  "The LDG auxiliary pass does not support systems with "
                  "primitive and conservative variables.");
    static_assert(
        detail::auxiliary_send_projects_no_uninitialized_data_to_face_v<
            EvolutionSystem, Metavariables>,
        "The LDG auxiliary pass does not compute the volume time derivative, "
        "so it cannot supply volume temporaries, fluxes, partial derivatives, "
        "an inverse spatial metric, or interior time derivatives to the "
        "auxiliary boundary corrections/conditions. An auxiliary boundary "
        "correction or condition "
        "registered for this system requires one of these on the boundary, "
        "which the auxiliary send does not support.");

    if (db::get<::Tags::LtsMode>(box) != LtsMode::Off) {
      ERROR("Local time stepping is not supported for the LDG auxiliary pass.");
    }

    const Element<Dim>& element = db::get<domain::Tags::Element<Dim>>(box);
    for (const auto& direction_and_neighbors : element.neighbors()) {
      if (not direction_and_neighbors.second.are_conforming()) {
        ERROR(
            "The LDG auxiliary send has not been tested with nonconforming "
            "meshes yet. Element "
            << element.id() << " has nonconforming neighbors in direction "
            << direction_and_neighbors.first << ".");
      }
    }
    return base::apply(box, inboxes, cache, array_index, action_list_meta,
                       parallel_component_meta);
  }
};
}  // namespace evolution::dg::Actions
