// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Protocols/ReductionDataFormatter.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/TypeTraits.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class TimeDelta;
namespace PUP {
class er;
}  // namespace PUP
namespace Tags {
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
  std::string operator()(double time, size_t num_points, double slab_size,
                         double min_time_step, double max_time_step,
                         double effective_time_step, double min_wall_time,
                         double max_wall_time) const;
  // NOLINTNEXTLINE
  void pup(PUP::er& p);
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
template <typename System>
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

  struct ObservePerCore {
    using type = bool;
    static constexpr Options::String help = {
        "Also write the data per-core in a file per-node."};
  };

  /// \cond
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveTimeStep);  // NOLINT
  /// \endcond

  using options = tmpl::list<SubfileName, PrintTimeToTerminal, ObservePerCore>;
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

  ObserveTimeStep();
  ObserveTimeStep(const std::string& subfile_name, bool output_time,
                  bool observe_per_core);

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using compute_tags_for_observation_box = tmpl::list<>;

  using return_tags = tmpl::list<>;
  // We obtain the grid size from the variables, rather than the mesh,
  // so that this observer is not DG-specific.
  using argument_tags =
      tmpl::list<::Tags::TimeStep, typename System::variables_tag>;

  template <typename ArrayIndex, typename ParallelComponent,
            typename Metavariables>
  void operator()(const TimeDelta& time_step,
                  const typename System::variables_tag::type& variables,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& array_index,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& observation_value) const {
    auto [observation_id, legend, reduction_data, formatter] =
        assemble_data(time_step, variables, observation_value);

    auto& local_observer = *Parallel::local_branch(
        Parallel::get_parallel_component<
            tmpl::conditional_t<Parallel::is_nodegroup_v<ParallelComponent>,
                                observers::ObserverWriter<Metavariables>,
                                observers::Observer<Metavariables>>>(cache));

    Parallel::ArrayComponentId array_component_id{
        std::add_pointer_t<ParallelComponent>{nullptr},
        Parallel::ArrayIndex<ArrayIndex>(array_index)};

    if constexpr (Parallel::is_nodegroup_v<ParallelComponent>) {
      const std::optional<int> observe_with_core_id =
          observe_per_core_
              ? std::make_optional(Parallel::my_node<int>(local_observer))
              : std::nullopt;
      Parallel::threaded_action<
          observers::ThreadedActions::CollectReductionDataOnNode>(
          local_observer, std::move(observation_id),
          std::move(array_component_id), subfile_path_, std::move(legend),
          std::move(reduction_data), std::move(formatter),
          observe_with_core_id);
    } else {
      Parallel::simple_action<observers::Actions::ContributeReductionData>(
          local_observer, std::move(observation_id),
          std::move(array_component_id), subfile_path_, std::move(legend),
          std::move(reduction_data), std::move(formatter), observe_per_core_);
    }
  }

  using observation_registration_tags = tmpl::list<>;
  std::pair<observers::TypeOfObservation, observers::ObservationKey>
  get_observation_type_and_key_for_registration() const;

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  auto assemble_data(const TimeDelta& time_step,
                     const typename System::variables_tag::type& variables,
                     const ObservationValue& observation_value) const
      -> std::tuple<observers::ObservationId, std::vector<std::string>,
                    ReductionData,
                    std::optional<Events::detail::FormatTimeOutput>>;

  std::string subfile_path_;
  bool output_time_;
  bool observe_per_core_;
};
}  // namespace Events
