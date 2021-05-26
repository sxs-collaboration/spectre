// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/Protocols/ReductionDataFormatter.hpp"
#include "IO/Observer/ReductionActions.hpp"  // IWYU pragma: keep
#include "IO/Observer/TypeOfObservation.hpp"
#include "Options/Options.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
struct TimeStep;
}  // namespace Tags
/// \endcond

namespace Events {
namespace detail {
using ObserveTimeStepReductionData = Parallel::ReductionData<
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    Parallel::ReductionDatum<size_t, funcl::Plus<>>,
    Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
    Parallel::ReductionDatum<double, funcl::Min<>>,
    Parallel::ReductionDatum<double, funcl::Max<>>,
    Parallel::ReductionDatum<
        double, funcl::Plus<>,
        funcl::Divides<funcl::Literal<1, double>, funcl::Divides<>>,
        std::index_sequence<1>>,
    Parallel::ReductionDatum<double, funcl::Min<>>,
    Parallel::ReductionDatum<double, funcl::Max<>>>;

struct FormatTimeOutput
    : tt::ConformsTo<observers::protocols::ReductionDataFormatter> {
  using reduction_data = ObserveTimeStepReductionData;
  std::string operator()(const double time, const size_t /* num_points */,
                         const double /* slab_size */,
                         const double /* min_time_step */,
                         const double /* max_time_step */,
                         const double /* effective_time_step */,
                         const double min_wall_time,
                         const double max_wall_time) const
      noexcept {
    std::stringstream ss;
    ss  << "Simulation time: " << std::to_string(time)
        << "s\n  Wall time: " << std::to_string(min_wall_time)
        << "s (min) - "
        << std::to_string(max_wall_time) << "s (max)";
    return ss.str();
  }
  // NOLINTNEXTLINE
  void pup(PUP::er& /*p*/) noexcept {}
};
}  // namespace detail

/*!
 * \brief %Observe the size of the time steps.
 *
 * Writes reduction quantities:
 * - `%Time`
 * - `NumberOfPoints`
 * - `%Slab size`
 * - `Minimum time step`
 * - `Maximum time step`
 * - `Effective time step`
 *
 * The effective time step is the step size of a global-time-stepping
 * method that would perform a similar amount of work.  This is the
 * harmonic mean of the step size over all grid points:
 *
 * \f{equation}
 * (\Delta t)_{\text{eff}}^{-1} =
 * \frac{\sum_{i \in \text{points}} (\Delta t)_i^{-1}}{N_{\text{points}}}.
 * \f}
 *
 * This corresponds to averaging the number of steps per unit time
 * taken by all points.
 *
 * All values are reported as positive numbers, even for backwards
 * evolutions.
 */
template <typename Metavariables>
class ObserveTimeStep : public Event {
 private:
  using ReductionData = Events::detail::ObserveTimeStepReductionData;

 public:
  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'."};
  };

  struct PrintTimeToTerminal {
    using type = bool;
    static constexpr Options::String help = {
        "Whether to print the time to screen."};
  };

  /// \cond
  explicit ObserveTimeStep(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveTimeStep);  // NOLINT
  /// \endcond

  using options = tmpl::list<SubfileName, PrintTimeToTerminal>;
  static constexpr Options::String help =
      "Observe the size of the time steps.\n"
      "\n"
      "Writes reduction quantities:\n"
      "- Time\n"
      "- NumberOfPoints\n"
      "- Slab size\n"
      "- Minimum time step\n"
      "- Maximum time step\n"
      "- Effective time step\n"
      "\n"
      "The effective time step is the step size of a global-time-stepping\n"
      "method that would perform a similar amount of work.\n"
      "\n"
      "All values are reported as positive numbers, even for backwards\n"
      "evolutions.";

  ObserveTimeStep() = default;
  explicit ObserveTimeStep(const std::string& subfile_name,
                           const bool output_time) noexcept;

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  // We obtain the grid size from the variables, rather than the mesh,
  // so that this observer is not DG-specific.
  using argument_tags =
      tmpl::list<Tags::Time, Tags::TimeStep,
                 typename Metavariables::system::variables_tag>;

  template <typename ArrayIndex, typename ParallelComponent>
  void operator()(
      const double& time, const TimeDelta& time_step,
      const typename Metavariables::system::variables_tag::type& variables,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index,
      const ParallelComponent* const /*meta*/) const noexcept {
    const size_t number_of_grid_points = variables.number_of_grid_points();
    const double slab_size = time_step.slab().duration().value();
    const double step_size = abs(time_step.value());
    const double wall_time = sys::wall_time();

    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    auto formatter =
        output_time_ ? std::make_optional(Events::detail::FormatTimeOutput{})
                     : std::nullopt;
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer, observers::ObservationId(time, subfile_path_ + ".dat"),
        observers::ArrayComponentId{
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ArrayIndex>(array_index)},
        subfile_path_,
        std::vector<std::string>{"Time", "NumberOfPoints", "Slab size",
                                 "Minimum time step", "Maximum time step",
                                 "Effective time step", "Minimum Walltime",
                                 "Maximum Walltime"},
        ReductionData{time, number_of_grid_points, slab_size, step_size,
                      step_size, number_of_grid_points / step_size, wall_time,
                      wall_time},
        std::move(formatter));
  }

  using observation_registration_tags = tmpl::list<>;
  std::pair<observers::TypeOfObservation, observers::ObservationKey>
  get_observation_type_and_key_for_registration() const noexcept {
    return {observers::TypeOfObservation::Reduction,
            observers::ObservationKey(subfile_path_ + ".dat")};
  }

  bool needs_evolved_variables() const noexcept override { return false; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Event::pup(p);
    p | subfile_path_;
    p | output_time_;
  }

 private:
  std::string subfile_path_;
  bool output_time_;
};

template <typename Metavariables>
ObserveTimeStep<Metavariables>::ObserveTimeStep(const std::string& subfile_name,
                                                const bool output_time) noexcept
    : subfile_path_("/" + subfile_name), output_time_(output_time) {}

/// \cond
template <typename Metavariables>
PUP::able::PUP_ID ObserveTimeStep<Metavariables>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
