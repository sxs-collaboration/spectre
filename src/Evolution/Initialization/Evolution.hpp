// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/TaggedVariant.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Structure/ChildSize.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeSteppingPolicy.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/ChangeSlabSize/Tags.hpp"
#include "Time/ChooseLtsStepSize.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel::Tags {
template <typename Index>
struct ArrayIndex;
}  // namespace Parallel::Tags
namespace Tags {
template <typename Tag>
struct StepperErrors;
}  // namespace Tags
namespace amr::Tags {
template <size_t VolumeDim>
struct Info;
}  // namespace amr::Tags
namespace domain::Tags {
template <size_t VolumeDim>
struct Element;
template <size_t VolumeDim>
struct Mesh;
}  // namespace domain::Tags
namespace evolution::dg::Tags {
template <size_t Dim>
struct MortarInfo;
}  // namespace evolution::dg::Tags
/// \endcond

namespace Initialization {

namespace detail {
inline Time initial_time(const bool time_runs_forward,
                         const double initial_time_value,
                         const double initial_slab_size) {
  const Slab initial_slab =
      time_runs_forward
          ? Slab::with_duration_from_start(initial_time_value,
                                           initial_slab_size)
          : Slab::with_duration_to_end(initial_time_value, initial_slab_size);
  return time_runs_forward ? initial_slab.start() : initial_slab.end();
}

template <typename TimeStepper>
void set_next_time_step_id(const gsl::not_null<TimeStepId*> next_time_step_id,
                           const Time& initial_time,
                           const bool time_runs_forward,
                           const TimeStepper& time_stepper) {
  *next_time_step_id = TimeStepId(
      time_runs_forward,
      -static_cast<int64_t>(time_stepper.number_of_past_steps()), initial_time);
}
}  // namespace detail

/// \ingroup InitializationGroup
/// \brief Initialize items related to time stepping
///
/// \details See the type aliases defined below for what items are added to the
/// GlobalCache and DataBox and how they are initialized
///
/// Since the evolution has not started yet, initialize the state
/// _before_ the initial time. So `Tags::TimeStepId` is undefined at this point,
/// and `Tags::Next<Tags::TimeStepId>` is the initial time.
template <typename Metavariables, typename TimeStepperBase>
struct TimeStepping {
  /// Tags for constant items added to the GlobalCache.  These items are
  /// initialized from input file options.
  using const_global_cache_tags =
      tmpl::list<::Tags::ConcreteTimeStepper<TimeStepperBase>>;

  /// Tags for mutable items added to the GlobalCache.  These items are
  /// initialized from input file options.
  using mutable_global_cache_tags = tmpl::list<>;

  /// Tags for items fetched by the DataBox and passed to the apply function
  using argument_tags =
      tmpl::list<::Tags::Time, Tags::InitialTimeDelta,
                 Tags::InitialSlabSize<TimeStepperBase::local_time_stepping>,
                 ::Tags::ConcreteTimeStepper<TimeStepperBase>>;

  /// Tags for simple DataBox items that are initialized from input file options
  using simple_tags_from_options =
      tmpl::list<::Tags::Time, Tags::InitialTimeDelta,
                 Tags::InitialSlabSize<TimeStepperBase::local_time_stepping>>;

  /// Tags for simple DataBox items that are default initialized.
  using default_initialized_simple_tags =
      tmpl::push_back<StepChoosers::step_chooser_simple_tags<
                          Metavariables, TimeStepperBase::local_time_stepping>,
                      ::Tags::TimeStepId, ::Tags::StepNumberWithinSlab,
                      ::Tags::AdaptiveSteppingDiagnostics>;

  /// Tags for items in the DataBox that are mutated by the apply function
  using return_tags =
      tmpl::list<::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
                 ::Tags::ChangeSlabSize::SlabSizeGoal>;

  /// Tags for mutable DataBox items that are either default initialized or
  /// initialized by the apply function
  using simple_tags =
      tmpl::append<default_initialized_simple_tags, return_tags>;

  /// Tags for immutable DataBox items (compute items or reference items) added
  /// to the DataBox.
  using compute_tags = time_stepper_ref_tags<TimeStepperBase>;

  /// Given the items fetched from a DataBox by the argument_tags when using
  /// LTS, mutate the items in the DataBox corresponding to return_tags
  static void apply(const gsl::not_null<TimeStepId*> next_time_step_id,
                    const gsl::not_null<TimeDelta*> time_step,
                    const gsl::not_null<double*> slab_size_goal,
                    const double initial_time_value,
                    const double initial_dt_value,
                    const double initial_slab_size,
                    const LtsTimeStepper& time_stepper) {
    const bool time_runs_forward = initial_dt_value > 0.0;
    const Time initial_time = detail::initial_time(
        time_runs_forward, initial_time_value, initial_slab_size);
    detail::set_next_time_step_id(next_time_step_id, initial_time,
                                  time_runs_forward, time_stepper);
    *time_step = choose_lts_step_size(initial_time, initial_dt_value);
    *slab_size_goal =
        time_runs_forward ? initial_slab_size : -initial_slab_size;
  }

  /// Given the items fetched from a DataBox by the argument_tags, when not
  /// using LTS, mutate the items in the DataBox corresponding to return_tags
  static void apply(const gsl::not_null<TimeStepId*> next_time_step_id,
                    const gsl::not_null<TimeDelta*> time_step,
                    const gsl::not_null<double*> slab_size_goal,
                    const double initial_time_value,
                    const double initial_dt_value,
                    const double initial_slab_size,
                    const TimeStepper& time_stepper) {
    const bool time_runs_forward = initial_dt_value > 0.0;
    const Time initial_time = detail::initial_time(
        time_runs_forward, initial_time_value, initial_slab_size);
    detail::set_next_time_step_id(next_time_step_id, initial_time,
                                  time_runs_forward, time_stepper);
    *time_step = (time_runs_forward ? 1 : -1) * initial_time.slab().duration();
    *slab_size_goal =
        time_runs_forward ? initial_slab_size : -initial_slab_size;
  }
};

/// \brief Initialize/update items related to time stepping after an AMR change
template <size_t Dim>
struct ProjectTimeStepping : tt::ConformsTo<amr::protocols::Projector> {
  using return_tags =
      tmpl::list<::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
                 ::Tags::TimeStep, ::Tags::Time, ::Tags::StepNumberWithinSlab,
                 ::Tags::AdaptiveSteppingDiagnostics,
                 ::Tags::ChangeSlabSize::SlabSizeGoal>;
  using argument_tags = tmpl::list<Parallel::Tags::ArrayIndex<ElementId<Dim>>>;

  static void apply(
      const gsl::not_null<TimeStepId*> /*time_step_id*/,
      const gsl::not_null<TimeStepId*> /*next_time_step_id*/,
      const gsl::not_null<TimeDelta*> /*time_step*/,
      const gsl::not_null<double*> /*time*/,
      const gsl::not_null<uint64_t*> /*step_number_within_slab*/,
      const gsl::not_null<AdaptiveSteppingDiagnostics*>
      /*adaptive_stepping_diagnostics*/,
      const gsl::not_null<double*> /*slab_size_goal*/,
      const ElementId<Dim>& /*element_id*/,
      const std::pair<Mesh<Dim>, Element<Dim>>& /*old_mesh_and_element*/) {
    // Do not change anything for p-refinement
  }

  template <typename... Tags>
  static void apply(const gsl::not_null<TimeStepId*> time_step_id,
                    const gsl::not_null<TimeStepId*> next_time_step_id,
                    const gsl::not_null<TimeDelta*> time_step,
                    const gsl::not_null<double*> time,
                    const gsl::not_null<uint64_t*> step_number_within_slab,
                    const gsl::not_null<AdaptiveSteppingDiagnostics*>
                        adaptive_stepping_diagnostics,
                    const gsl::not_null<double*> slab_size_goal,
                    const ElementId<Dim>& element_id,
                    const tuples::TaggedTuple<Tags...>& parent_items) {
    *time_step_id = get<::Tags::TimeStepId>(parent_items);
    *next_time_step_id = get<::Tags::Next<::Tags::TimeStepId>>(parent_items);
    *time_step = get<::Tags::TimeStep>(parent_items);
    *time = get<::Tags::Time>(parent_items);
    *slab_size_goal = get<::Tags::ChangeSlabSize::SlabSizeGoal>(parent_items);
    *step_number_within_slab = get<::Tags::StepNumberWithinSlab>(parent_items);

    // Since AdaptiveSteppingDiagnostics are reduced over all elements, we
    // set the slab quantities to the same value over all children, and the
    // step quantities to belong to the first child
    const auto& parent_diagnostics =
        get<::Tags::AdaptiveSteppingDiagnostics>(parent_items);
    const auto& parent_amr_flags =
        get<amr::Tags::Info<Dim>>(parent_items).flags;
    const auto& parent_id =
        get<Parallel::Tags::ArrayIndex<ElementId<Dim>>>(parent_items);
    auto children_ids = amr::ids_of_children(parent_id, parent_amr_flags);
    if (element_id == children_ids.front()) {
      *adaptive_stepping_diagnostics = parent_diagnostics;
    } else {
      adaptive_stepping_diagnostics->number_of_slabs =
          parent_diagnostics.number_of_slabs;
      adaptive_stepping_diagnostics->number_of_slab_size_changes =
          parent_diagnostics.number_of_slab_size_changes;
    }
  }

  template <typename... Tags>
  static void apply(
      const gsl::not_null<TimeStepId*> time_step_id,
      const gsl::not_null<TimeStepId*> next_time_step_id,
      const gsl::not_null<TimeDelta*> time_step,
      const gsl::not_null<double*> time,
      const gsl::not_null<uint64_t*> step_number_within_slab,
      const gsl::not_null<AdaptiveSteppingDiagnostics*>
          adaptive_stepping_diagnostics,
      const gsl::not_null<double*> slab_size_goal,
      const ElementId<Dim>& /*element_id*/,
      const std::unordered_map<ElementId<Dim>, tuples::TaggedTuple<Tags...>>&
          children_items) {
    const auto slowest_child =
        alg::min_element(children_items, [](const auto& a, const auto& b) {
          const auto& time_step_a = get<::Tags::TimeStep>(a.second);
          const auto& time_step_b = get<::Tags::TimeStep>(b.second);
          ASSERT(time_step_a.is_positive() == time_step_b.is_positive(),
                 "Elements are not taking time steps in the same direction!");
          return time_step_a.is_positive() ? (time_step_a < time_step_b)
                                           : (time_step_a > time_step_b);
        });
    const auto& slowest_child_items = (*slowest_child).second;
    *time_step_id = get<::Tags::TimeStepId>(slowest_child_items);
    *next_time_step_id =
        get<::Tags::Next<::Tags::TimeStepId>>(slowest_child_items);
    *time_step = get<::Tags::TimeStep>(slowest_child_items);
    *time = get<::Tags::Time>(slowest_child_items);
    *slab_size_goal =
        get<::Tags::ChangeSlabSize::SlabSizeGoal>(slowest_child_items);
    *step_number_within_slab =
        get<::Tags::StepNumberWithinSlab>(slowest_child_items);
    const auto& slowest_child_diagnostics =
        get<::Tags::AdaptiveSteppingDiagnostics>(slowest_child_items);

    adaptive_stepping_diagnostics->number_of_slabs =
        slowest_child_diagnostics.number_of_slabs;
    adaptive_stepping_diagnostics->number_of_slab_size_changes =
        slowest_child_diagnostics.number_of_slab_size_changes;
    for (const auto& [_, child_items] : children_items) {
      *adaptive_stepping_diagnostics +=
          get<::Tags::AdaptiveSteppingDiagnostics>(child_items);
    }
  }
};

/// \ingroup InitializationGroup
/// \brief Initialize time-stepper items
///
/// DataBox changes:
/// - Adds:
///   * `db::add_tag_prefix<Tags::dt, variables_tag>`
///   * `Tags::HistoryEvolvedVariables<variables_tag>`
/// - Removes: nothing
/// - Modifies: nothing
///
/// \note HistoryEvolvedVariables is allocated, but needs to be initialized
template <typename Metavariables>
struct TimeStepperHistory {
  static constexpr size_t dim = Metavariables::volume_dim;
  using variables_tag = typename Metavariables::system::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;

  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<>;
  using simple_tags =
      tmpl::list<dt_variables_tag,
                 ::Tags::HistoryEvolvedVariables<variables_tag>>;
  using compute_tags = tmpl::list<>;

  using argument_tags =
      tmpl::list<::Tags::TimeStepper<TimeStepper>, domain::Tags::Mesh<dim>>;
  using return_tags = simple_tags;

  static void apply(
      const gsl::not_null<typename dt_variables_tag::type*> dt_vars,
      const gsl::not_null<TimeSteppers::History<typename variables_tag::type>*>
          history,
      const TimeStepper& time_stepper, const Mesh<dim>& mesh) {
    // Will be overwritten before use
    dt_vars->initialize(mesh.number_of_grid_points());

    // All steppers we have that need to start at low order require
    // one additional point per order, so this is the order that
    // requires no initial past steps.
    const size_t starting_order =
        visit(
            []<typename Tag>(
                const std::pair<tmpl::type_<Tag>, typename Tag::type&&> order) {
              if constexpr (std::is_same_v<Tag,
                                           TimeSteppers::Tags::FixedOrder>) {
                return order.second;
              } else {
                return order.second.minimum;
              }
            },
            time_stepper.order()) -
        time_stepper.number_of_past_steps();
    history->integration_order(starting_order);
  }
};

/// \brief Initialize/update items related to time stepper history after an AMR
/// change
///
/// \note `Tags::TimeStep` and `Tags::Next<Tags::TimeStepId>` are not
/// initially set by this projector.  They are only updated if the
/// time stepper must be restarted because of LTS h-refinement.
template <typename Metavariables>
struct ProjectTimeStepperHistory : tt::ConformsTo<amr::protocols::Projector> {
  static constexpr size_t dim = Metavariables::volume_dim;
  using variables_tag = typename Metavariables::system::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using history_tag = ::Tags::HistoryEvolvedVariables<variables_tag>;

  using return_tags =
      tmpl::list<dt_variables_tag, history_tag,
                 ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep>;
  using argument_tags = tmpl::list<
      domain::Tags::Mesh<dim>, Parallel::Tags::ArrayIndex<ElementId<dim>>,
      ::Tags::TimeStepper<TimeStepper>, evolution::dg::Tags::MortarInfo<dim>>;

  static void apply(
      const gsl::not_null<typename dt_variables_tag::type*> dt_vars,
      const gsl::not_null<typename history_tag::type*> history,
      const gsl::not_null<TimeStepId*> /*next_time_step_id*/,
      const gsl::not_null<TimeDelta*> /*time_step*/, const Mesh<dim>& new_mesh,
      const ElementId<dim>& /*element_id*/, const TimeStepper& /*time_stepper*/,
      const DirectionalIdMap<dim, evolution::dg::MortarInfo<dim>>&
      /*mortar_info*/,
      const std::pair<Mesh<dim>, Element<dim>>& old_mesh_and_element) {
    const auto& old_mesh = old_mesh_and_element.first;
    if (old_mesh == new_mesh) {
      return;  // mesh was not refined, so no projection needed
    }
    const auto projection_matrices =
        Spectral::p_projection_matrices(old_mesh, new_mesh);
    const auto& old_extents = old_mesh.extents();
    history->map_entries(
        [&projection_matrices, &old_extents](const auto entry) {
          *entry = apply_matrices(projection_matrices, *entry, old_extents);
        });
    dt_vars->initialize(new_mesh.number_of_grid_points());
  }

  template <typename... Tags>
  static void apply(
      const gsl::not_null<typename dt_variables_tag::type*> dt_vars,
      const gsl::not_null<typename history_tag::type*> history,
      const gsl::not_null<TimeStepId*> next_time_step_id,
      const gsl::not_null<TimeDelta*> time_step, const Mesh<dim>& new_mesh,
      const ElementId<dim>& element_id, const TimeStepper& time_stepper,
      const DirectionalIdMap<dim, evolution::dg::MortarInfo<dim>>& mortar_info,
      const tuples::TaggedTuple<Tags...>& parent_items) {
    dt_vars->initialize(new_mesh.number_of_grid_points());
    ASSERT(element_id.refinement_levels() == make_array<dim>(0_st) or
               not mortar_info.empty(),
           "Element has no neighbors but is not a full block");
    const auto time_stepping_policy = time_stepping_policy_for_h_refinement(
        mortar_info, &get<evolution::dg::Tags::MortarInfo<dim>>(parent_items),
        {});
    switch (time_stepping_policy) {
      case evolution::dg::TimeSteppingPolicy::Conservative: {
        if (time_stepper.number_of_past_steps() != 0) {
          ERROR_NO_TRACE(
              "Cannot perform h-refinement with LTS steppers requiring "
              "initialization.");
        }
        const auto integrator_order = time_stepper.order();
        if (variants::holds_alternative<TimeSteppers::Tags::FixedOrder>(
                integrator_order)) {
          *history = typename history_tag::type{
              get<TimeSteppers::Tags::FixedOrder>(integrator_order)};
          return;
        }
        const auto start_order =
            get<TimeSteppers::Tags::VariableOrder>(integrator_order).minimum;
        *history = typename history_tag::type{start_order};

        const auto reduced_step = restart_time_step(parent_items, start_order);
        if (abs(reduced_step) < abs(*time_step)) {
          *time_step = reduced_step;
          *next_time_step_id = time_stepper.next_time_id(
              get<::Tags::TimeStepId>(parent_items), *time_step);
        }
        break;
      }
      case evolution::dg::TimeSteppingPolicy::EqualRate: {
        const auto& parent_id =
            get<domain::Tags::Element<dim>>(parent_items).id();
        const auto& parent_mesh = get<domain::Tags::Mesh<dim>>(parent_items);
        const auto child_sizes = domain::child_size(element_id.segment_ids(),
                                                    parent_id.segment_ids());
        const auto projection_matrices =
            Spectral::projection_matrix_parent_to_child(parent_mesh, new_mesh,
                                                        child_sizes);
        transform(history, get<history_tag>(parent_items),
                  [&](const auto& source_entry) {
                    return apply_matrices(projection_matrices, source_entry,
                                          parent_mesh.extents());
                  });
        break;
      }
      default:
        ERROR("Unhandled time-stepping policy: " << time_stepping_policy);
    }
  }

  template <typename... Tags>
  static void apply(
      const gsl::not_null<typename dt_variables_tag::type*> dt_vars,
      const gsl::not_null<typename history_tag::type*> history,
      const gsl::not_null<TimeStepId*> next_time_step_id,
      const gsl::not_null<TimeDelta*> time_step, const Mesh<dim>& new_mesh,
      const ElementId<dim>& element_id, const TimeStepper& time_stepper,
      const DirectionalIdMap<dim, evolution::dg::MortarInfo<dim>>& mortar_info,
      const std::unordered_map<ElementId<dim>, tuples::TaggedTuple<Tags...>>&
          children_items) {
    dt_vars->initialize(new_mesh.number_of_grid_points());
    ASSERT(element_id.refinement_levels() == make_array<dim>(0_st) or
               not mortar_info.empty(),
           "Element has no neighbors but is not a full block");
    const auto time_stepping_policy =
        time_stepping_policy_for_h_refinement<Tags...>(mortar_info, {},
                                                       {&children_items});
    switch (time_stepping_policy) {
      case evolution::dg::TimeSteppingPolicy::Conservative: {
        if (time_stepper.number_of_past_steps() != 0) {
          ERROR_NO_TRACE(
              "Cannot perform h-refinement with LTS steppers requiring "
              "initialization.");
        }
        const auto integrator_order = time_stepper.order();
        if (variants::holds_alternative<TimeSteppers::Tags::FixedOrder>(
                integrator_order)) {
          *history = typename history_tag::type{
              get<TimeSteppers::Tags::FixedOrder>(integrator_order)};
          return;
        }
        const auto start_order =
            get<TimeSteppers::Tags::VariableOrder>(integrator_order).minimum;
        *history = typename history_tag::type{start_order};

        for (const auto& [child, child_items] : children_items) {
          const auto reduced_step = restart_time_step(child_items, start_order);
          if (abs(reduced_step) < abs(*time_step)) {
            *time_step = reduced_step;
            *next_time_step_id = time_stepper.next_time_id(
                get<::Tags::TimeStepId>(children_items.begin()->second),
                *time_step);
          }
        }
        break;
      }
      case evolution::dg::TimeSteppingPolicy::EqualRate: {
        bool first_child = true;
        for (const auto& [child_id, child_items] : children_items) {
          const auto& child_mesh = get<domain::Tags::Mesh<dim>>(child_items);
          const auto child_sizes = domain::child_size(child_id.segment_ids(),
                                                      element_id.segment_ids());
          const auto projection_matrices =
              Spectral::projection_matrix_child_to_parent(child_mesh, new_mesh,
                                                          child_sizes);
          if (first_child) {
            transform(history, get<history_tag>(child_items),
                      [&](const auto& source_entry) {
                        return apply_matrices(projection_matrices, source_entry,
                                              child_mesh.extents());
                      });
            first_child = false;
          } else {
            transform_mutate(
                history, get<history_tag>(child_items),
                [&](const auto dest_entry, const auto& source_entry) {
                  *dest_entry += apply_matrices(
                      projection_matrices, source_entry, child_mesh.extents());
                });
          }
        }
        break;
      }
      default:
        ERROR("Unhandled time-stepping policy: " << time_stepping_policy);
    }
  }

 private:
  template <typename... Tags>
  static evolution::dg::TimeSteppingPolicy
  time_stepping_policy_for_h_refinement(
      const DirectionalIdMap<dim, evolution::dg::MortarInfo<dim>>&
          element_infos,
      const std::optional<gsl::not_null<
          const DirectionalIdMap<dim, evolution::dg::MortarInfo<dim>>*>>&
          parent_infos,
      const std::optional<gsl::not_null<const std::unordered_map<
          ElementId<dim>, tuples::TaggedTuple<Tags...>>*>>& children_items) {
    std::optional<evolution::dg::TimeSteppingPolicy> policy{};
    const auto process_infos =
        [&policy](const DirectionalIdMap<dim, evolution::dg::MortarInfo<dim>>&
                      infos) {
          for (const auto& info : infos) {
            const auto mortar_policy = info.second.time_stepping_policy();
            ASSERT(not policy.has_value() or *policy == mortar_policy,
                   "Inconsistent policies: "
                       << *policy << " and " << mortar_policy
                       << "\nWe currently require all mortars to have the "
                          "same policy when doing h-refinement.  This might "
                          "be relaxed when we add an LTS policy other than "
                          "Conservative.");
            policy.emplace(mortar_policy);
          }
        };

    process_infos(element_infos);
    if (parent_infos.has_value()) {
      process_infos(**parent_infos);
    }
    if (children_items.has_value()) {
      if constexpr (sizeof...(Tags) > 0) {
        for (const auto& child : **children_items) {
          process_infos(
              get<evolution::dg::Tags::MortarInfo<dim>>(child.second));
        }
      } else {
        ERROR("Children but no child data");
      }
    }

    ASSERT(policy.has_value(),
           "Found no mortars, either before or after h-refinement.  A mortar "
           "should have been created or destroyed, so this is not possible.");
    return *policy;
  }

  template <typename... Tags>
  static TimeDelta restart_time_step(
      const tuples::TaggedTuple<Tags...>& old_items, const size_t start_order) {
    const auto& old_history = get<history_tag>(old_items);
    if (old_history.integration_order() == start_order) {
      return get<::Tags::TimeStep>(old_items);
    }

    const auto& time_step_id = get<::Tags::TimeStepId>(old_items);
    const auto& errors = get<::Tags::StepperErrors<variables_tag>>(old_items);
    if (not errors[1].has_value()) {
      ERROR_NO_TRACE(
          "Evolutions performing h-refinement with variable-order local "
          "time-stepping must use ErrorControl for step size choosing.");
    }
    ASSERT(errors[1]->errors[start_order - 1].has_value(),
           "Start-order estimate not available.");

    // At this low an order, we are almost certainly step-size-limited
    // by accuracy rather than stability, so ignore things like the
    // change to the grid spacing.
    return choose_lts_step_size(
        time_step_id.step_time(),
        errors[1]->step_size.value() *
            pow(1.0 / std::max(*errors[1]->errors[start_order - 1], 1e-14),
                1.0 / static_cast<double>(start_order)));
  }
};
}  // namespace Initialization
