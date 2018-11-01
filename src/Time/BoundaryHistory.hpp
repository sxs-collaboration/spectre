// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <boost/iterator/transform_iterator.hpp>
#include <cstddef>
#include <deque>
#include <map>
#include <pup.h>
#include <pup_stl.h>  // IWYU pragma: keep
#include <tuple>
#include <utility>

#include "ErrorHandling/Assert.hpp"
#include "Parallel/PupStlCpp11.hpp"  // IWYU pragma: keep
#include "Time/Time.hpp"  // IWYU pragma: keep
#include "Time/TimeId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeSteppers {

/// \ingroup TimeSteppersGroup
/// History data used by a TimeStepper for boundary integration.
/// \tparam LocalVars local variables passed to the boundary coupling
/// \tparam RemoteVars remote variables passed to the boundary coupling
/// \tparam CouplingResult result of the coupling function
template <typename LocalVars, typename RemoteVars, typename CouplingResult>
class BoundaryHistory {
  template <typename Vars>
  using IteratorType = boost::transform_iterator<
    const Time& (*)(const std::tuple<Time, Vars>&),
    typename std::deque<std::tuple<Time, Vars>>::const_iterator>;
 public:
  using local_iterator = IteratorType<LocalVars>;
  using remote_iterator = IteratorType<RemoteVars>;

  // No copying because of the pointers in the cache.  Moving is fine
  // because we also move the container being pointed into and maps
  // guarantee that this doesn't invalidate pointers.
  BoundaryHistory() = default;
  BoundaryHistory(const BoundaryHistory&) = delete;
  BoundaryHistory(BoundaryHistory&&) = default;
  BoundaryHistory& operator=(const BoundaryHistory&) = delete;
  BoundaryHistory& operator=(BoundaryHistory&&) = default;
  ~BoundaryHistory() = default;

  /// Add a new value to the end of the history of the indicated side.
  //@{
  void local_insert(const TimeId& time_id, LocalVars vars) noexcept {
    local_data_.emplace_back(time_id.time(), std::move(vars));
  }
  void remote_insert(const TimeId& time_id, RemoteVars vars) noexcept {
    remote_data_.emplace_back(time_id.time(), std::move(vars));
  }
  //@}

  /// Add a new value to the front of the history of the indicated
  /// side.  This is often convenient for setting initial data.
  //@{
  void local_insert_initial(const TimeId& time_id, LocalVars vars) noexcept {
    local_data_.emplace_front(time_id.time(), std::move(vars));
  }
  void remote_insert_initial(const TimeId& time_id, RemoteVars vars) noexcept {
    remote_data_.emplace_front(time_id.time(), std::move(vars));
  }
  //@}

  /// Mark all data before the passed point in history on the
  /// indicated side as unneeded so it can be removed.  Calling this
  /// directly should not often be necessary, as it is handled
  /// internally by the time steppers.
  //@{
  void local_mark_unneeded(const local_iterator& first_needed) noexcept {
    mark_unneeded<0>(make_not_null(&local_data_), first_needed);
  }
  void remote_mark_unneeded(const remote_iterator& first_needed) noexcept {
    mark_unneeded<1>(make_not_null(&remote_data_), first_needed);
  }
  //@}

  /// Access to the sequence of times on the indicated side.
  //@{
  local_iterator local_begin() const noexcept {
    return local_iterator(local_data_.begin(), std::get<0>);
  }
  local_iterator local_end() const noexcept {
    return local_iterator(local_data_.end(), std::get<0>);
  }

  remote_iterator remote_begin() const noexcept {
    return remote_iterator(remote_data_.begin(), std::get<0>);
  }
  remote_iterator remote_end() const noexcept {
    return remote_iterator(remote_data_.end(), std::get<0>);
  }

  size_t local_size() const noexcept { return local_data_.size(); }
  size_t remote_size() const noexcept { return remote_data_.size(); }
  //@}

  /// Evaluate the coupling function at the given local and remote
  /// history entries.  The coupling function will be passed the local
  /// and remote FluxVars and should return a CouplingResult.  Values are
  /// cached internally, so callers should ensure that the coupling
  /// functions provided on separate calls produce the same result.
  template <typename Coupling>
  const CouplingResult& coupling(Coupling&& c, const local_iterator& local,
                                 const remote_iterator& remote) const noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  template <size_t Side, typename DataType, typename Iterator>
  void mark_unneeded(gsl::not_null<DataType*> data,
                     const Iterator& first_needed) noexcept;

  std::deque<std::tuple<Time, LocalVars>> local_data_;
  std::deque<std::tuple<Time, RemoteVars>> remote_data_;
  // We use pointers instead of iterators because deque invalidates
  // iterators when elements are inserted or removed at the ends, but
  // not pointers.
  mutable std::map<std::pair<const Time*, const Time*>, CouplingResult>
      coupling_cache_;
};

template <typename LocalVars, typename RemoteVars, typename CouplingResult>
template <size_t Side, typename DataType, typename Iterator>
void BoundaryHistory<LocalVars, RemoteVars, CouplingResult>::mark_unneeded(
    gsl::not_null<DataType*> data, const Iterator& first_needed) noexcept {
  for (auto it = data->begin(); it != first_needed.base(); ++it) {
    // Clean out cache entries referring to the entry we are removing.
    for (auto cache_entry = coupling_cache_.begin();
         cache_entry != coupling_cache_.end();) {
      if (std::get<Side>(cache_entry->first) == &std::get<0>(*it)) {
        cache_entry = coupling_cache_.erase(cache_entry);
      } else {
        ++cache_entry;
      }
    }
  }
  data->erase(data->begin(), first_needed.base());
}

template <typename LocalVars, typename RemoteVars, typename CouplingResult>
template <typename Coupling>
const CouplingResult&
BoundaryHistory<LocalVars, RemoteVars, CouplingResult>::coupling(
    Coupling&& c, const local_iterator& local,
    const remote_iterator& remote) const noexcept {
  const auto insert_result = coupling_cache_.insert(
      std::make_pair(std::make_pair(&*local, &*remote), CouplingResult{}));
  CouplingResult& inserted_value = insert_result.first->second;
  const bool is_new_value = insert_result.second;
  if (is_new_value) {
    inserted_value =
        std::forward<Coupling>(c)(cpp17::as_const(std::get<1>(*local.base())),
                                  cpp17::as_const(std::get<1>(*remote.base())));
  }
  return inserted_value;
}

template <typename LocalVars, typename RemoteVars, typename CouplingResult>
void BoundaryHistory<LocalVars, RemoteVars, CouplingResult>::pup(
    PUP::er& p) noexcept {
  p | local_data_;
  p | remote_data_;

  const size_t cache_size = PUP_stl_container_size(p, coupling_cache_);
  if (p.isUnpacking()) {
    for (size_t entry_num = 0; entry_num < cache_size; ++entry_num) {
      size_t local_index, remote_index;
      CouplingResult cache_value;
      p | local_index;
      p | remote_index;
      p | cache_value;
      const auto cache_key =
          std::make_pair(&std::get<0>(local_data_[local_index]),
                         &std::get<0>(remote_data_[remote_index]));
      coupling_cache_.insert(std::make_pair(cache_key, cache_value));
    }
  } else {
    for (auto& cache_entry : coupling_cache_) {
      // clang-tidy: modernize-use-auto - Ensuring the correct type is
      // important here to prevent undefined behavior in charm.  I
      // want to be explicit.
      size_t local_index = static_cast<size_t>(  // NOLINT
          std::find_if(
              local_data_.begin(), local_data_.end(),
              [goal = cache_entry.first.first](const auto& entry) noexcept {
                return &std::get<0>(entry) == goal;
              }) -
          local_data_.begin());
      ASSERT(local_index < local_data_.size(),
             "Failed to find local history entry for cache entry");

      // clang-tidy: modernize-use-auto - Ensuring the correct type is
      // important here to prevent undefined behavior in charm.  I
      // want to be explicit.
      size_t remote_index = static_cast<size_t>(  // NOLINT
          std::find_if(
              remote_data_.begin(), remote_data_.end(),
              [goal = cache_entry.first.second](const auto& entry) noexcept {
                return &std::get<0>(entry) == goal;
              }) -
          remote_data_.begin());
      ASSERT(remote_index < remote_data_.size(),
             "Failed to find remote history entry for cache entry");

      p | local_index;
      p | remote_index;
      p | cache_entry.second;
    }
  }
}
}  // namespace TimeSteppers
