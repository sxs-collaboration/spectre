// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/SimpleProfiler.hpp"

#include <cstdint>
#include <ctime>
#include <mutex>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <utility>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace profiling {
namespace {
struct Unprofiled {
  using FileLabel = NoLabel;
  using ColumnLabel = NoLabel;

  static std::string make_file_name(const NoLabel& /*file_label*/) {
    return "Unprofiled";
  }
};
}  // namespace

void SimpleProfilerBase::shadow(const std::uint64_t nsec) {
  shadowed_nsec_ += nsec;
}

void SimpleProfilerData::Times::pup(PUP::er& p) {
  p | call_count;
  p | exclusive_nsec;
  p | inclusive_nsec;
}

SimpleProfilerData::SimpleProfilerData() {
  // Create the entry for unprofiled time.  The pup routine wants to
  // update this during packing, but if it is created there it causes
  // desynchronization with the sizing phase and leads to memory
  // corruption.
  times_[Unprofiled::make_file_name({})][""];
}

SimpleProfilerData::SimpleProfilerData(SimpleProfilerData&& other) noexcept {
  const std::scoped_lock lock(other.mutex_);
  ASSERT(other.active_profiler_ == nullptr,
         "Cannot move SimpleProfilerData while in use");
  times_ = std::move(other.times_);
  type_times_ = std::move(other.type_times_);
  updated_time_ = other.updated_time_;
}

SimpleProfilerData& SimpleProfilerData::operator=(
    SimpleProfilerData&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  const std::scoped_lock lock(mutex_, other.mutex_);
  ASSERT(active_profiler_ == nullptr,
         "Cannot overwrite SimpleProfilerData while in use");
  ASSERT(other.active_profiler_ == nullptr,
         "Cannot move SimpleProfilerData while in use");
  times_ = std::move(other.times_);
  type_times_ = std::move(other.type_times_);
  updated_time_ = other.updated_time_;
  return *this;
}

SimpleProfilerData::TimingData SimpleProfilerData::timing_data() {
  const std::scoped_lock lock(mutex_);
  // Update the unprofiled time if there is no active profiler.  We
  // will miss the time from any active profiler, but there's nothing
  // we can do about that.
  update_unprofiled_time(current_time());
  return times_;
}

void SimpleProfilerData::pup(PUP::er& p) {
  // Even with the lock, it is not safe to serialize this class while
  // it is being updated, because the data size could change between
  // the sizing and packing phases.  However, we still want to lock so
  // that the sizer can be used to estimate the memory usage.
  const std::scoped_lock lock(mutex_);
  if (p.isPacking()) {
    // Update the unprofiled time if there is no active profiler.  We
    // will miss the time from any active profiler, but there's
    // nothing we can do about that.
    update_unprofiled_time(current_time());
  }
  p | times_;
  if (p.isUnpacking()) {
    type_times_.clear();
    // We probably won't be unpacking inside a profiled region since
    // that should happen very early in program execution, but leave
    // active_profiler_ alone just in case as overwriting it would
    // trigger an ASSERT.

    // We leave updated_time_ alone, but setting it to the current
    // time would also be reasonable.
  }
}

SimpleProfilerData::ProfilerState SimpleProfilerData::start_profiling(
    const gsl::not_null<SimpleProfilerBase*> profiler) {
  const std::scoped_lock lock(mutex_);
  const timespec now = current_time();
  update_unprofiled_time(now);
  ProfilerState result{now, active_profiler_};
  active_profiler_ = profiler;
  return result;
}

void SimpleProfilerData::active_profiler_is_moving(
    const gsl::not_null<SimpleProfilerBase*> source,
    const gsl::not_null<SimpleProfilerBase*> dest) {
  const std::scoped_lock lock(mutex_);
  ASSERT(active_profiler_ == source, "Moving a shadowed profiler");
  active_profiler_ = dest;
}

timespec SimpleProfilerData::current_time() {
  timespec result{};
  clock_gettime(CLOCK_MONOTONIC, &result);
  return result;
}

void SimpleProfilerData::update_unprofiled_time(const timespec& now) {
  if (active_profiler_ == nullptr) {
    record_result<Unprofiled>(NoLabel{}, NoLabel{}, updated_time_, now, 0);
  }
}
}  // namespace profiling
