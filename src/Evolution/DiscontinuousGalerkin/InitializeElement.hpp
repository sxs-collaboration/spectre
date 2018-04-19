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
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
// IWYU pragma: no_include "DataStructures/VariablesHelpers.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Tags.hpp"
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

  template <class System>
  using return_tag_list = tmpl::list<
      // Simple items
      Tags::TimeId, Tags::TimeStep, Tags::Extents<Dim>, Tags::Element<Dim>,
      Tags::ElementMap<Dim>, typename System::variables_tag,
      Tags::HistoryEvolvedVariables<
          typename System::variables_tag,
          db::add_tag_prefix<Tags::dt, typename System::variables_tag>>,
      db::add_tag_prefix<Tags::dt, typename System::variables_tag>,
      // Compute items
      Tags::Time, Tags::LogicalCoordinates<Dim>,
      Tags::Coordinates<Tags::ElementMap<Dim>, Tags::LogicalCoordinates<Dim>>,
      Tags::InverseJacobian<Tags::ElementMap<Dim>,
                            Tags::LogicalCoordinates<Dim>>,
      Tags::deriv<typename System::variables_tag::tags_list,
                  typename System::gradients_tags,
                  Tags::InverseJacobian<Tags::ElementMap<Dim>,
                                        Tags::LogicalCoordinates<Dim>>>,
      Tags::InternalDirections<Dim>, interface_tag<Tags::Direction<Dim>>,
      interface_tag<Tags::Extents<Dim - 1>>,
      interface_tag<Tags::UnnormalizedFaceNormal<Dim>>,
      interface_tag<typename System::template magnitude_tag<
          Tags::UnnormalizedFaceNormal<Dim>>>,
      interface_tag<Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>,
                                     typename System::template magnitude_tag<
                                         Tags::UnnormalizedFaceNormal<Dim>>>>,
      // This should be the boundary forms of
      // System::normal_dot_fluxes::argument_tags, but it is not clear
      // how to get that.  System::variables_tag should generally be a
      // superset.
      interface_tag<typename System::variables_tag>>;

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

    // Set initial data from analytic solution
    Variables<variables_tags> vars{num_grid_points};
    vars.assign_subset(Parallel::get<solution_tag>(cache).variables(
        inertial_coords, initial_time.value(), variables_tags{}));

    typename Tags::HistoryEvolvedVariables<
        Tags::Variables<variables_tags>,
        Tags::dt<Tags::Variables<db::wrap_tags_in<Tags::dt, variables_tags>>>>::
        type history_dt_vars;
    const auto& time_stepper = Parallel::get<CacheTags::TimeStepper>(cache);
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

    // Set up initial time
    TimeId time_id{};
    time_id.time = initial_time;

    db::compute_databox_type<return_tag_list<typename Metavariables::system>>
        outbox = db::create<
            db::get_items<return_tag_list<typename Metavariables::system>>,
            db::get_compute_items<
                return_tag_list<typename Metavariables::system>>>(
            time_id, initial_dt, std::move(mesh), std::move(element),
            std::move(map), std::move(vars), std::move(history_dt_vars),
            Variables<db::wrap_tags_in<Tags::dt, variables_tags>>{
                num_grid_points, 0.0});

    return std::make_tuple(std::move(outbox));
  }
};
}  // namespace Actions
}  // namespace dg
