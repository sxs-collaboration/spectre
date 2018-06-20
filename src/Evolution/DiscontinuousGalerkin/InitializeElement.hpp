// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>  // IWYU pragma: keep
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Variables.hpp"                   // IWYU pragma: keep
// IWYU pragma: no_include "DataStructures/VariablesHelpers.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
// IWYU pragma: no_include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
template <size_t VolumeDim>
class ElementIndex;
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace tuples {
template <typename... Tags>
class TaggedTuple;  // IWYU pragma: keep
}  // namespace tuples
/// \endcond

namespace dg {
namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Initialize a dG element with analytic initial data
///
/// Uses:
/// - ConstGlobalCache:
///   * A tag deriving off of Cache::AnalyticSolutionBase
///   * CacheTags::TimeStepper
///
/// DataBox changes:
/// - Adds:
///   * Tags::TimeId
///   * Tags::Time
///   * Tags::TimeStep
///   * Tags::LogicalCoordinates<Dim>
///   * Tags::Extents<Dim>
///   * Tags::Element<Dim>
///   * Tags::ElementMap<Dim>
///   * System::variables_tag
///   * Tags::HistoryEvolvedVariables<System::variables_tag,
///                  db::add_tag_prefix<Tags::dt, System::variables_tag>>
///   * Tags::Coordinates<Tags::ElementMap<Dim>,
///                       Tags::LogicalCoordinates<Dim>>
///   * Tags::InverseJacobian<Tags::ElementMap<Dim>,
///                           Tags::LogicalCoordinates<Dim>>
///   * Tags::deriv<System::variables_tag::tags_list,
///                 System::gradients_tags,
///                 Tags::InverseJacobian<Tags::ElementMap<Dim>,
///                                       Tags::LogicalCoordinates<Dim>>>
///   * db::add_tag_prefix<Tags::dt, System::variables_tag>
///   * Tags::UnnormalizedFaceNormal<Dim>
/// - Removes: nothing
/// - Modifies: nothing
template <size_t Dim>
struct InitializeElement {
  template <typename Tag>
  using interface_tag = Tags::Interface<Tags::InternalDirections<Dim>, Tag>;

  template <typename Tag>
  using boundary_tag = Tags::Interface<Tags::BoundaryDirections<Dim>, Tag>;

  template <class Metavariables>
  using return_tag_list = tmpl::list<
      // Simple items
      Tags::TimeId, Tags::Next<Tags::TimeId>, Tags::TimeStep,
      Tags::Extents<Dim>, Tags::Element<Dim>, Tags::ElementMap<Dim>,
      typename Metavariables::system::variables_tag,
      Tags::HistoryEvolvedVariables<
          typename Metavariables::system::variables_tag,
          db::add_tag_prefix<Tags::dt,
                             typename Metavariables::system::variables_tag>>,
      db::add_tag_prefix<Tags::dt,
                         typename Metavariables::system::variables_tag>,
      typename dg::FluxCommunicationTypes<
          Metavariables>::global_time_stepping_mortar_data_tag,
      Tags::Mortars<Tags::Next<Tags::TimeId>, Dim>,
      // Compute items
      Tags::Time, Tags::LogicalCoordinates<Dim>,
      Tags::Coordinates<Tags::ElementMap<Dim>, Tags::LogicalCoordinates<Dim>>,
      Tags::InverseJacobian<Tags::ElementMap<Dim>,
                            Tags::LogicalCoordinates<Dim>>,
      Tags::deriv<typename Metavariables::system::variables_tag::tags_list,
                  typename Metavariables::system::gradients_tags,
                  Tags::InverseJacobian<Tags::ElementMap<Dim>,
                                        Tags::LogicalCoordinates<Dim>>>,
      Tags::InternalDirections<Dim>, interface_tag<Tags::Direction<Dim>>,
      interface_tag<Tags::Extents<Dim - 1>>,
      interface_tag<Tags::UnnormalizedFaceNormal<Dim>>,
      interface_tag<typename Metavariables::system::template magnitude_tag<
          Tags::UnnormalizedFaceNormal<Dim>>>,
      interface_tag<Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>,
      // This should be the boundary forms of
      // System::normal_dot_fluxes::argument_tags, but it is not clear
      // how to get that.  System::variables_tag should generally be a
      // superset.
      interface_tag<typename Metavariables::system::variables_tag>,
      Tags::BoundaryDirections<Dim>, boundary_tag<Tags::Direction<Dim>>,
      boundary_tag<Tags::Extents<Dim - 1>>,
      boundary_tag<Tags::UnnormalizedFaceNormal<Dim>>,
      boundary_tag<typename Metavariables::system::template magnitude_tag<
          Tags::UnnormalizedFaceNormal<Dim>>>,
      boundary_tag<Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>,
      boundary_tag<typename Metavariables::system::variables_tag>>;

  template <typename... InboxTags, typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static auto apply(const db::DataBox<tmpl::list<>>& /*box*/,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ElementIndex<Dim>& array_index,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    std::vector<std::array<size_t, Dim>> initial_extents,
                    Domain<Dim, Frame::Inertial> domain, Time initial_time,
                    TimeDelta initial_dt) noexcept {
    using solution_tag = CacheTags::AnalyticSolutionBase;
    using variables_tags =
        typename Metavariables::system::variables_tag::tags_list;

    ElementId<Dim> element_id{array_index};

    const auto& my_block = domain.blocks()[element_id.block_id()];

    ElementMap<Dim, Frame::Inertial> map{element_id,
                                         my_block.coordinate_map().get_clone()};

    Element<Dim> element = create_initial_element(element_id, my_block);

    ::Index<Dim> mesh{initial_extents[element_id.block_id()]};
    const auto num_grid_points = mesh.product();
    auto logical_coords = logical_coordinates(mesh);
    auto inertial_coords = map(logical_coords);

    // Set up initial time
    const auto& time_stepper = Parallel::get<CacheTags::TimeStepper>(cache);
    const TimeId time_id(initial_dt.is_positive(), 0, initial_time);
    TimeId next_time_id = time_stepper.next_time_id(time_id, initial_dt);
    if (next_time_id.is_at_slab_boundary()) {
      const auto next_time = next_time_id.step_time();
      next_time_id = TimeId(
          initial_dt.is_positive(), 1,
          next_time.with_slab(next_time.slab().advance_towards(initial_dt)));
    }

    // Set initial data from analytic solution
    Variables<variables_tags> vars{num_grid_points};
    vars.assign_subset(Parallel::get<solution_tag>(cache).variables(
        inertial_coords, initial_time.value(), variables_tags{}));

    typename Tags::HistoryEvolvedVariables<
        Tags::Variables<variables_tags>,
        Tags::dt<Tags::Variables<db::wrap_tags_in<Tags::dt, variables_tags>>>>::
        type history_dt_vars;
    if (not time_stepper.is_self_starting()) {
      // We currently just put initial points at past slab boundaries.
      Time past_t = initial_time;
      TimeDelta past_dt = initial_dt;
      for (size_t i = time_stepper.number_of_past_steps(); i > 0; --i) {
        past_dt = past_dt.with_slab(past_dt.slab().advance_towards(-past_dt));
        past_t -= past_dt;
        Variables<variables_tags> hist_vars{num_grid_points};
        Variables<db::wrap_tags_in<Tags::dt, variables_tags>> dt_hist_vars{
            num_grid_points};
        hist_vars.assign_subset(Parallel::get<solution_tag>(cache).variables(
            inertial_coords, past_t.value(), variables_tags{}));
        dt_hist_vars.assign_subset(Parallel::get<solution_tag>(cache).variables(
            inertial_coords, past_t.value(),
            db::wrap_tags_in<Tags::dt, variables_tags>{}));
        history_dt_vars.insert_initial(past_t, std::move(hist_vars),
                                       std::move(dt_hist_vars));
      }
    }

    // Set up boundary information
    db::item_type<typename dg::FluxCommunicationTypes<
        Metavariables>::global_time_stepping_mortar_data_tag>
        boundary_data{};
    typename Tags::Mortars<Tags::Next<Tags::TimeId>, Dim>::type
        mortar_next_time_ids{};
    for (const auto& direction_neighbors : element.neighbors()) {
      const auto& direction = direction_neighbors.first;
      const auto& neighbors = direction_neighbors.second;
      for (const auto& neighbor : neighbors) {
        const auto mortar_id = std::make_pair(direction, neighbor);
        boundary_data.insert({mortar_id, {}});
        mortar_next_time_ids.insert({mortar_id, time_id});
      }
    }

    db::compute_databox_type<return_tag_list<Metavariables>> outbox =
        db::create<db::get_items<return_tag_list<Metavariables>>,
                   db::get_compute_items<return_tag_list<Metavariables>>>(
            time_id, next_time_id, initial_dt, std::move(mesh),
            std::move(element), std::move(map), std::move(vars),
            std::move(history_dt_vars),
            Variables<db::wrap_tags_in<Tags::dt, variables_tags>>{
                num_grid_points, 0.0},
            std::move(boundary_data), std::move(mortar_next_time_ids));

    return std::make_tuple(std::move(outbox));
  }
};
}  // namespace Actions
}  // namespace dg
