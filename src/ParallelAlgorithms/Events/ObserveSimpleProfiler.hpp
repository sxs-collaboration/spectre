// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <atomic>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/SimpleProfiler.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
class NodeLock;
}  // namespace Parallel
namespace Parallel::Tags {
struct LastSimpleProfilerWrite;
struct SimpleProfilerData;
}  // namespace Parallel::Tags
namespace observers {
template <class Metavariables>
struct ObserverWriter;
}  // namespace observers
namespace profiling {
template <typename Metavariables>
struct SimpleProfilerDataComponent;
}  // namespace profiling
/// \endcond

namespace Events {
/*!
 * \brief %Observe profiling data.
 *
 * For each `SimpleProfiler` file label, writes the observation value
 * and the count, exclusive, and inclusive times, in nanoseconds, for
 * each column label.
 *
 * \warning For implementation simplicity, this observer assumes
 * observation values are monotonically increasing.  In
 * backward-in-time evolutions only the first observation will occur.
 */
class ObserveSimpleProfiler : public Event {
  struct GetTimingData {
    using return_type =
        std::optional<std::vector<profiling::SimpleProfilerData::TimingData>>;
    template <typename ParallelComponent, typename DbTags>
    static std::optional<std::vector<profiling::SimpleProfilerData::TimingData>>
    apply(db::DataBox<DbTags>& box,
          const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
          const double observation_value) {
      std::atomic<double>& last_write =
          db::get_mutable_reference<Parallel::Tags::LastSimpleProfilerWrite>(
              make_not_null(&box));
      double old_last_write = last_write.load(std::memory_order_acquire);
      if (old_last_write >= observation_value) {
        return std::nullopt;
      }

      if (not last_write.compare_exchange_strong(
              old_last_write, observation_value, std::memory_order_release,
              std::memory_order_relaxed)) {
        // Someone else updated the value.  It can't have been to a
        // value earlier than observation_value, since if so we would
        // have done that observation before getting to
        // observation_value, so the other thread is handling this.
        return std::nullopt;
      }

      auto& all_data =
          db::get_mutable_reference<Parallel::Tags::SimpleProfilerData>(
              make_not_null(&box));
      std::vector<profiling::SimpleProfilerData::TimingData> result{};
      result.reserve(all_data.size());
      for (auto& proc_data : all_data) {
        result.emplace_back(proc_data.timing_data());
      }
      return {std::move(result)};
    }
  };

 public:
  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the group to write to inside the HDF5 file without a "
        "preceding '/'."};
  };

  /// \cond
  explicit ObserveSimpleProfiler(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveSimpleProfiler);  // NOLINT
  /// \endcond

  using options = tmpl::list<SubfileName>;
  static constexpr Options::String help =
      "Observe profiling data.\n"
      "\n"
      "For each `SimpleProfiler` file label, writes the observation value\n"
      "and the count, exclusive, and inclusive times, in nanoseconds, for\n"
      "each column label.\n"
      "\n"
      "WARNING: For implementation simplicity, this observer assumes\n"
      "observation values are monotonically increasing.  In\n"
      "backward-in-time evolutions only the first observation will occur.";

  ObserveSimpleProfiler();
  explicit ObserveSimpleProfiler(const std::string& subfile_name);

  using compute_tags_for_observation_box = tmpl::list<>;

  using return_tags = tmpl::list<>;
  using argument_tags = tmpl::list<>;

  template <typename ArrayIndex, typename ParallelComponent,
            typename Metavariables>
  void operator()(Parallel::GlobalCache<Metavariables>& cache,
                  const ArrayIndex& /*array_index*/,
                  const ParallelComponent* const /*meta*/,
                  const ObservationValue& observation_value) const {
    auto timing_data = Parallel::local_synchronous_action<GetTimingData>(
        Parallel::get_parallel_component<
            profiling::SimpleProfilerDataComponent<Metavariables>>(cache),
        observation_value.value);

    if (not timing_data.has_value()) {
      return;
    }

    auto formatted_data = format_timing_data(
        observation_value, std::move(*timing_data),
        static_cast<size_t>(cache.first_proc_on_node(cache.my_node())));

    auto& observer_writer_proxy = Parallel::get_parallel_component<
        observers::ObserverWriter<Metavariables>>(cache);
    Parallel::threaded_action<
        observers::ThreadedActions::WriteMultipleReductionDataRows>(
        observer_writer_proxy[0], std::move(formatted_data));
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override;

  void pup(PUP::er& p) override;

 private:
  std::vector<observers::DatRows> format_timing_data(
      const ObservationValue& observation_value,
      const std::vector<profiling::SimpleProfilerData::TimingData>& timing_data,
      size_t first_proc_on_node) const;

  std::string subfile_path_;
};
}  // namespace Events
