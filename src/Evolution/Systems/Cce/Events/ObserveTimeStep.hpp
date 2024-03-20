// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <tuple>
#include <vector>

#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Time.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct TimeStep;
}  // namespace Tags
/// \endcond

namespace Cce::Events {

/*!
 * \brief %Observe the size of the time steps on the characteristic evolution.
 *
 * Writes reduction quantities:
 * - `%Time`
 * - `Time Step`
 *
 * The subfile will be written into the `/Cce` subgroup.
 */
class ObserveTimeStep : public Event {
 public:
  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'. The subfile will be written into the "
        "subgroup '/Cce'."};
  };

  struct PrintTimeToTerminal {
    using type = bool;
    static constexpr Options::String help = {
        "Whether to print the time to screen."};
  };

  /// \cond
  explicit ObserveTimeStep(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveTimeStep);  // NOLINT
  /// \endcond

  using options = tmpl::list<SubfileName, PrintTimeToTerminal>;
  static constexpr Options::String help =
      "Observe the size of the time step for the characteristic evolution.\n"
      "\n"
      "Writes quantities:\n"
      "- Time\n"
      "- Time Step\n"
      "\n"
      "The subfile will be written into the subgroup '/Cce'.";

  ObserveTimeStep() = default;
  explicit ObserveTimeStep(const std::string& subfile_name,
                           const bool output_time);

  using observed_reduction_data_tags = tmpl::list<>;

  using compute_tags_for_observation_box = tmpl::list<>;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<::Tags::TimeStep>;

  template <typename ArrayIndex, typename ParallelComponent,
            typename Metavariables>
  void operator()(const TimeDelta& time_step,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& /*array_index*/,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& observation_value) const {
    std::vector<double> data_to_write{observation_value.value,
                                      time_step.value()};

    auto& writer = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);

    Parallel::threaded_action<
        observers::ThreadedActions::WriteReductionDataRow>(
        writer[0], subfile_path_, legend_,
        std::make_tuple(std::move(data_to_write)));

    if (output_time_) {
      Parallel::printf(
          "Simulation time: %s\n"
          "  Wall time: %s\n",
          std::to_string(observation_value.value), sys::pretty_wall_time());
    }
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return false; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event::pup(p);
    p | subfile_path_;
    p | output_time_;
    p | legend_;
  }

 private:
  std::string subfile_path_;
  bool output_time_;
  std::vector<std::string> legend_;
};

ObserveTimeStep::ObserveTimeStep(const std::string& subfile_name,
                                 const bool output_time)
    : subfile_path_("/Cce/" + subfile_name),
      output_time_(output_time),
      legend_(std::vector<std::string>{"Time", "Time Step"}) {}

/// \cond
PUP::able::PUP_ID ObserveTimeStep::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Cce::Events
