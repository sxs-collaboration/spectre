// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Events/ObserveSimpleProfiler.hpp"

#include <cstddef>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "IO/Observer/ReductionActions.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/SimpleProfiler.hpp"

namespace Events {
ObserveSimpleProfiler::ObserveSimpleProfiler(CkMigrateMessage* m) : Event(m) {}

ObserveSimpleProfiler::ObserveSimpleProfiler() = default;

ObserveSimpleProfiler::ObserveSimpleProfiler(const std::string& subfile_name)
    : subfile_path_("/" + subfile_name) {}

bool ObserveSimpleProfiler::needs_evolved_variables() const { return false; }

void ObserveSimpleProfiler::pup(PUP::er& p) {
  Event::pup(p);
  p | subfile_path_;
}

std::vector<observers::DatRows> ObserveSimpleProfiler::format_timing_data(
    const ObservationValue& observation_value,
    const std::vector<profiling::SimpleProfilerData::TimingData>& timing_data,
    const size_t first_proc_on_node) const {
  std::vector<observers::DatRows> output{};
  for (size_t local_proc = 0; local_proc < timing_data.size(); ++local_proc) {
    const size_t proc = first_proc_on_node + local_proc;
    const auto& proc_data = timing_data[local_proc];
    for (const auto& [subfile_key, subfile_data] : proc_data) {
      std::string subfile_name = MakeString{} << subfile_path_ << "/Proc"
                                              << proc << "/" << subfile_key;
      std::vector<std::string> legend{};
      legend.reserve(3 * subfile_data.size() + 1);
      std::vector<std::vector<double>> rows(1);
      auto& row = rows[0];
      row.reserve(3 * subfile_data.size() + 1);
      legend.emplace_back(observation_value.name);
      row.emplace_back(observation_value.value);
      for (const auto& [column, times] : subfile_data) {
        legend.emplace_back(column.empty() ? "count" : column + " count");
        legend.emplace_back(column.empty() ? "exclusive"
                                           : column + " exclusive");
        legend.emplace_back(column.empty() ? "inclusive"
                                           : column + " inclusive");
        row.emplace_back(times.call_count);
        row.emplace_back(times.exclusive_nsec);
        row.emplace_back(times.inclusive_nsec);
      }
      output.emplace_back(std::move(subfile_name), std::move(legend),
                          std::move(rows));
    }
  }
  return output;
}

PUP::able::PUP_ID ObserveSimpleProfiler::my_PUP_ID = 0;  // NOLINT
}  // namespace Events
