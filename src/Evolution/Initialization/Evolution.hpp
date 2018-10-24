// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Time/Slab.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace Initialization {

/// \brief Initialize items related to time-evolution of the system
///
/// DataBox changes:
/// - Adds:
///   * Tags::TimeId
///   * `Tags::Next<Tags::TimeId>`
///   * Tags::TimeStep
///   * `db::add_tag_prefix<Tags::dt, variables_tag>`
///   * `Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>`
///   * Tags::Time
///   * Tags::ComputeDeriv  (for non-conservative systems)
///   * Tags::ComputeDiv (for conservative systems)
/// - Removes: nothing
/// - Modifies: nothing
///
/// \note HistoryEvolvedVariables is allocated, but needs to be initialized
template <typename System>
struct Evolution {
  static constexpr size_t dim = System::volume_dim;
  using variables_tag = typename System::variables_tag;
  using dt_variables_tag = db::add_tag_prefix<Tags::dt, variables_tag>;

  using simple_tags = db::AddSimpleTags<
      Tags::TimeId, Tags::Next<Tags::TimeId>, Tags::TimeStep, dt_variables_tag,
      Tags::HistoryEvolvedVariables<variables_tag, dt_variables_tag>>;

  template <typename LocalSystem, bool IsInFluxConservativeForm =
                                      LocalSystem::is_in_flux_conservative_form>
  struct ComputeTags {
    using type = db::AddComputeTags<
        Tags::Time,
        Tags::DerivCompute<variables_tag,
                           Tags::InverseJacobian<Tags::ElementMap<dim>,
                                                 Tags::LogicalCoordinates<dim>>,
                           typename System::gradients_tags>>;
  };

  template <typename LocalSystem>
  struct ComputeTags<LocalSystem, true> {
    using type = db::AddComputeTags<
        Tags::Time,
        Tags::DivCompute<db::add_tag_prefix<Tags::Flux, variables_tag,
                                            tmpl::size_t<dim>, Frame::Inertial>,
                         Tags::InverseJacobian<Tags::ElementMap<dim>,
                                               Tags::LogicalCoordinates<dim>>>>;
  };

  using compute_tags = typename ComputeTags<System>::type;

  // Global time stepping
  template <typename Metavariables,
            Requires<not Metavariables::local_time_stepping> = nullptr>
  static TimeDelta get_initial_time_step(
      const Time& initial_time, const double initial_dt_value,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/) noexcept {
    return (initial_dt_value > 0.0 ? 1 : -1) * initial_time.slab().duration();
  }

  // Local time stepping
  template <typename Metavariables,
            Requires<Metavariables::local_time_stepping> = nullptr>
  static TimeDelta get_initial_time_step(
      const Time& initial_time, const double initial_dt_value,
      const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
    const auto& step_controller =
        Parallel::get<OptionTags::StepController>(cache);
    return step_controller.choose_step(initial_time, initial_dt_value);
  }
  template <typename TagsList, typename Metavariables>
  static auto initialize(db::DataBox<TagsList>&& box,
                         const Parallel::ConstGlobalCache<Metavariables>& cache,
                         const double initial_time_value,
                         const double initial_dt_value,
                         const double initial_slab_size) noexcept {
    using DtVars = typename dt_variables_tag::type;

    const bool time_runs_forward = initial_dt_value > 0.0;
    const Slab initial_slab =
        time_runs_forward
            ? Slab::with_duration_from_start(initial_time_value,
                                             initial_slab_size)
            : Slab::with_duration_to_end(initial_time_value, initial_slab_size);
    const Time initial_time =
        time_runs_forward ? initial_slab.start() : initial_slab.end();
    const TimeDelta initial_dt =
        get_initial_time_step(initial_time, initial_dt_value, cache);

    const size_t num_grid_points =
        db::get<Tags::Mesh<dim>>(box).number_of_grid_points();

    // Will be overwritten before use
    DtVars dt_vars{num_grid_points};
    typename Tags::HistoryEvolvedVariables<variables_tag,
                                           dt_variables_tag>::type history;

    // The slab number is increased in the self-start phase each
    // time one order of accuracy is obtained, and the evolution
    // proper starts with slab 0.
    const auto& time_stepper = Parallel::get<OptionTags::TimeStepper>(cache);

    const TimeId time_id(
        time_runs_forward,
        -static_cast<int64_t>(time_stepper.number_of_past_steps()),
        initial_time);

    return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
        std::move(box), TimeId{}, time_id, initial_dt, std::move(dt_vars),
        std::move(history));
  }
};

}  // namespace Initialization
