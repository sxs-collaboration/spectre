// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <deque>
#include <map>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>  // IWYU pragma: keep
#include <type_traits>
#include <utility>

#include "DataStructures/MathWrapper.hpp"
#include "Time/Time.hpp"  // IWYU pragma: keep
#include "Time/TimeStepId.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace TimeSteppers {

/// \ingroup TimeSteppersGroup
/// Type erased base class for evaluating BoundaryHistory couplings.
template <typename T>
class BoundaryHistoryEvaluator {
 protected:
  BoundaryHistoryEvaluator() = default;
  BoundaryHistoryEvaluator(const BoundaryHistoryEvaluator&) = default;
  BoundaryHistoryEvaluator(BoundaryHistoryEvaluator&&) = default;
  BoundaryHistoryEvaluator& operator=(const BoundaryHistoryEvaluator&) =
      default;
  BoundaryHistoryEvaluator& operator=(BoundaryHistoryEvaluator&&) = default;
  ~BoundaryHistoryEvaluator() = default;

 public:
  using iterator = std::deque<Time>::const_iterator;

  /// The current order of integration.
  virtual size_t integration_order() const = 0;

  /// Access to the sequence of times on the indicated side.
  /// @{
  virtual iterator local_begin() const = 0;
  virtual iterator local_end() const = 0;

  virtual iterator remote_begin() const = 0;
  virtual iterator remote_end() const = 0;

  virtual size_t local_size() const = 0;
  virtual size_t remote_size() const = 0;
  /// @}

  /// Evaluate the coupling function at the given local and remote
  /// history entries.  The coupling function will be passed the local
  /// and remote vars and should return a CouplingResult.  Values are
  /// cached in the associated BoundaryHistory object.
  virtual MathWrapper<const T> operator()(const iterator& local,
                                          const iterator& remote) const = 0;
};

/// \ingroup TimeSteppersGroup
/// Type erased base class for removing BoundaryHistory entries.
class BoundaryHistoryCleaner {
 protected:
  BoundaryHistoryCleaner() = default;
  BoundaryHistoryCleaner(const BoundaryHistoryCleaner&) = default;
  BoundaryHistoryCleaner(BoundaryHistoryCleaner&&) = default;
  BoundaryHistoryCleaner& operator=(const BoundaryHistoryCleaner&) = default;
  BoundaryHistoryCleaner& operator=(BoundaryHistoryCleaner&&) = default;
  ~BoundaryHistoryCleaner() = default;

 public:
  using iterator = std::deque<Time>::const_iterator;

  /// The current order of integration.
  virtual size_t integration_order() const = 0;

  /// Access to the sequence of times on the indicated side.
  /// @{
  virtual iterator local_begin() const = 0;
  virtual iterator local_end() const = 0;

  virtual iterator remote_begin() const = 0;
  virtual iterator remote_end() const = 0;

  virtual size_t local_size() const = 0;
  virtual size_t remote_size() const = 0;
  /// @}

  /// Mark all data before the passed point in history on the
  /// indicated side as unneeded so it can be removed.  Calling this
  /// outside of time stepper implementations should not often be
  /// necessary.
  /// @{
  virtual void local_mark_unneeded(const iterator& first_needed) const = 0;
  virtual void remote_mark_unneeded(const iterator& first_needed) const = 0;
  /// @}
};

/// \ingroup TimeSteppersGroup
/// History data used by a TimeStepper for boundary integration.
/// \tparam LocalVars local variables passed to the boundary coupling
/// \tparam RemoteVars remote variables passed to the boundary coupling
/// \tparam CouplingResult result of the coupling function
template <typename LocalVars, typename RemoteVars, typename CouplingResult>
class BoundaryHistory {
 public:
  using iterator = std::deque<Time>::const_iterator;

  // No copying because of the pointers in the cache.  Moving is fine
  // because we also move the container being pointed into and maps
  // guarantee that this doesn't invalidate pointers.
  BoundaryHistory() = default;
  BoundaryHistory(const BoundaryHistory&) = delete;
  BoundaryHistory(BoundaryHistory&&) = default;
  BoundaryHistory& operator=(const BoundaryHistory&) = delete;
  BoundaryHistory& operator=(BoundaryHistory&&) = default;
  ~BoundaryHistory() = default;

  explicit BoundaryHistory(const size_t integration_order)
      : integration_order_(integration_order) {}

  /// The current order of integration.  This should match the value
  /// in the History.
  /// @{
  size_t integration_order() const { return integration_order_; }
  void integration_order(const size_t integration_order) {
    integration_order_ = integration_order;
  }
  /// @}

  /// Add a new value to the end of the history of the indicated side.
  /// @{
  void local_insert(const TimeStepId& time_id, LocalVars vars) {
    ASSERT(time_id.substep() == 0, "Substeps not supported in LTS");
    local_data_.first.emplace_back(time_id.step_time());
    local_data_.second.emplace_back(std::move(vars));
  }
  void remote_insert(const TimeStepId& time_id, RemoteVars vars) {
    ASSERT(time_id.substep() == 0, "Substeps not supported in LTS");
    remote_data_.first.emplace_back(time_id.step_time());
    remote_data_.second.emplace_back(std::move(vars));
  }
  /// @}

  /// Add a new value to the front of the history of the indicated
  /// side.  This is often convenient for setting initial data.
  /// @{
  void local_insert_initial(const TimeStepId& time_id, LocalVars vars) {
    ASSERT(time_id.substep() == 0, "Substeps not supported in LTS");
    local_data_.first.emplace_front(time_id.step_time());
    local_data_.second.emplace_front(std::move(vars));
  }
  void remote_insert_initial(const TimeStepId& time_id, RemoteVars vars) {
    ASSERT(time_id.substep() == 0, "Substeps not supported in LTS");
    remote_data_.first.emplace_front(time_id.step_time());
    remote_data_.second.emplace_front(std::move(vars));
  }
  /// @}

  /// Access to the sequence of times on the indicated side.
  /// @{
  iterator local_begin() const { return local_data_.first.begin(); }
  iterator local_end() const { return local_data_.first.end(); }

  iterator remote_begin() const { return remote_data_.first.begin(); }
  iterator remote_end() const { return remote_data_.first.end(); }

  size_t local_size() const { return local_data_.first.size(); }
  size_t remote_size() const { return remote_data_.first.size(); }
  /// @}

  /// Look up the stored local data at the `time_id`. It is an error to request
  /// data at a `time_id` that has not been inserted yet.
  const LocalVars& local_data(const TimeStepId& time_id) const;

 private:
  template <typename Coupling>
  class BoundaryHistoryEvaluatorImpl final
      : public BoundaryHistoryEvaluator<math_wrapper_type<CouplingResult>> {
   public:
    static_assert(
        std::is_same_v<CouplingResult,
                       std::invoke_result_t<Coupling, const LocalVars&,
                                            const RemoteVars&>>,
        "Provided coupling return type does not match type stored in the "
        "history.");

   private:
    friend class BoundaryHistory;
    BoundaryHistoryEvaluatorImpl(
        const gsl::not_null<const BoundaryHistory*> history, Coupling coupling)
        : history_(history), coupling_(std::forward<Coupling>(coupling)) {}

   public:
    size_t integration_order() const override {
      return history_->integration_order();
    }

    iterator local_begin() const override { return history_->local_begin(); }
    iterator local_end() const override { return history_->local_end(); }

    iterator remote_begin() const override { return history_->remote_begin(); }
    iterator remote_end() const override { return history_->remote_end(); }

    size_t local_size() const override { return history_->local_size(); }
    size_t remote_size() const override { return history_->remote_size(); }

    MathWrapper<const math_wrapper_type<CouplingResult>> operator()(
        const iterator& local, const iterator& remote) const override;

   private:
    gsl::not_null<const BoundaryHistory*> history_;
    Coupling coupling_;
  };

 public:
  /// Returns an unspecified type derived from
  /// `BoundaryHistoryEvaluator<math_wrapper_type<CouplingResult>>`
  /// usable for evaluating the passed coupling.  Every call to this
  /// function must pass an equivalent value for `coupling` (in the
  /// sense that the same arguments produce the same result).
  template <typename Coupling>
  auto evaluator(Coupling&& coupling) const {
    using Result = BoundaryHistoryEvaluatorImpl<Coupling>;
    // Check the guarantee in the docs
    static_assert(
        std::is_convertible_v<
            Result*,
            BoundaryHistoryEvaluator<math_wrapper_type<CouplingResult>>*>);
    return Result(this, std::forward<Coupling>(coupling));
  }

 private:
  class BoundaryHistoryCleanerImpl final : public BoundaryHistoryCleaner {
   public:
    size_t integration_order() const override {
      return history_->integration_order();
    }

    iterator local_begin() const override { return history_->local_begin(); }
    iterator local_end() const override { return history_->local_end(); }

    iterator remote_begin() const override { return history_->remote_begin(); }
    iterator remote_end() const override { return history_->remote_end(); }

    size_t local_size() const override { return history_->local_size(); }
    size_t remote_size() const override { return history_->remote_size(); }

    void local_mark_unneeded(const iterator& first_needed) const override {
      history_->template mark_unneeded<0>(first_needed);
    }
    void remote_mark_unneeded(const iterator& first_needed) const override {
      history_->template mark_unneeded<1>(first_needed);
    }

   private:
    friend class BoundaryHistory;
    explicit BoundaryHistoryCleanerImpl(BoundaryHistory* const history)
        : history_(history) {}

    gsl::not_null<BoundaryHistory*> history_;
  };

 public:
  /// Returns an unspecified class derived from
  /// `BoundaryHistoryCleaner` suitable for expiring history entries.
  auto cleaner() {
    // Check the guarantee in the docs
    static_assert(std::is_convertible_v<BoundaryHistoryCleanerImpl*,
                                        BoundaryHistoryCleaner*>);
    return BoundaryHistoryCleanerImpl(this);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  std::ostream& print(std::ostream& os) const;

 private:
  template <size_t Side>
  void mark_unneeded(const iterator& first_needed);

  size_t integration_order_{0};
  // The type erased classes need access to the list of times, so we
  // can't store the (time, data) pairs in the natural data structure
  // but have to invert the deque and pair entries.
  std::pair<std::deque<Time>, std::deque<LocalVars>> local_data_;
  std::pair<std::deque<Time>, std::deque<RemoteVars>> remote_data_;
  // We use pointers instead of iterators because deque invalidates
  // iterators when elements are inserted or removed at the ends, but
  // not pointers.
  // NOLINTNEXTLINE(spectre-mutable)
  mutable std::map<std::pair<const Time*, const Time*>, CouplingResult>
      coupling_cache_;
};

template <typename LocalVars, typename RemoteVars, typename CouplingResult>
template <size_t Side>
void BoundaryHistory<LocalVars, RemoteVars, CouplingResult>::mark_unneeded(
    const iterator& first_needed) {
  auto& data = [this]() -> auto& {
    if constexpr (Side == 0) {
      return local_data_;
    } else {
      return remote_data_;
    }
  }();
  for (auto it = data.first.begin(); it != first_needed; ++it) {
    // Clean out cache entries referring to the entry we are removing.
    for (auto cache_entry = coupling_cache_.begin();
         cache_entry != coupling_cache_.end();) {
      if (std::get<Side>(cache_entry->first) == &*it) {
        cache_entry = coupling_cache_.erase(cache_entry);
      } else {
        ++cache_entry;
      }
    }
  }
  data.second.erase(data.second.begin(),
                    data.second.begin() + (first_needed - data.first.begin()));
  data.first.erase(data.first.begin(), first_needed);
}

template <typename LocalVars, typename RemoteVars, typename CouplingResult>
const LocalVars&
BoundaryHistory<LocalVars, RemoteVars, CouplingResult>::local_data(
    const TimeStepId& time_id) const {
  const Time& time = time_id.step_time();
  // Look up the data for this time, starting at the end of the `std::deque`,
  // i.e. the most-recently inserted data.
  auto value_it = local_data_.second.rbegin();
  for (auto time_it = local_data_.first.rbegin();
       time_it != local_data_.first.rend();
       ++time_it, ++value_it) {
    if (*time_it == time) {
      return *value_it;
    }
  }
  ERROR("No local data was found at time " << time << ".");
}

template <typename LocalVars, typename RemoteVars, typename CouplingResult>
void BoundaryHistory<LocalVars, RemoteVars, CouplingResult>::pup(PUP::er& p) {
  p | integration_order_;
  p | local_data_;
  p | remote_data_;

  const size_t cache_size = PUP_stl_container_size(p, coupling_cache_);
  if (p.isUnpacking()) {
    for (size_t entry_num = 0; entry_num < cache_size; ++entry_num) {
      size_t local_index = 0;
      size_t remote_index = 0;
      CouplingResult cache_value;
      p | local_index;
      p | remote_index;
      p | cache_value;
      const auto cache_key = std::make_pair(&local_data_.first[local_index],
                                            &remote_data_.first[remote_index]);
      coupling_cache_.insert(std::make_pair(cache_key, cache_value));
    }
  } else {
    for (auto& cache_entry : coupling_cache_) {
      // clang-tidy: modernize-use-auto - Ensuring the correct type is
      // important here to prevent undefined behavior in charm.  I
      // want to be explicit.
      size_t local_index = static_cast<size_t>(  // NOLINT
          std::find_if(local_data_.first.begin(), local_data_.first.end(),
                       [goal = cache_entry.first.first](const auto& entry) {
                         return &entry == goal;
                       }) -
          local_data_.first.begin());
      ASSERT(local_index < local_data_.first.size(),
             "Failed to find local history entry for cache entry");

      // clang-tidy: modernize-use-auto - Ensuring the correct type is
      // important here to prevent undefined behavior in charm.  I
      // want to be explicit.
      size_t remote_index = static_cast<size_t>(  // NOLINT
          std::find_if(remote_data_.first.begin(), remote_data_.first.end(),
                       [goal = cache_entry.first.second](const auto& entry) {
                         return &entry == goal;
                       }) -
          remote_data_.first.begin());
      ASSERT(remote_index < remote_data_.first.size(),
             "Failed to find remote history entry for cache entry");

      p | local_index;
      p | remote_index;
      p | cache_entry.second;
    }
  }
}

template <typename LocalVars, typename RemoteVars, typename CouplingResult>
template <typename Coupling>
MathWrapper<const math_wrapper_type<CouplingResult>>
BoundaryHistory<LocalVars, RemoteVars, CouplingResult>::
    BoundaryHistoryEvaluatorImpl<Coupling>::operator()(
        const iterator& local, const iterator& remote) const {
  const auto insert_result = history_->coupling_cache_.insert(
      std::make_pair(std::make_pair(&*local, &*remote), CouplingResult{}));
  CouplingResult& inserted_value = insert_result.first->second;
  const bool is_new_value = insert_result.second;
  if (is_new_value) {
    inserted_value =
        coupling_(history_->local_data_.second[static_cast<size_t>(
                      local - history_->local_data_.first.begin())],
                  history_->remote_data_.second[static_cast<size_t>(
                      remote - history_->remote_data_.first.begin())]);
  }
  return make_math_wrapper(inserted_value);
}

template <typename LocalVars, typename RemoteVars, typename CouplingResult>
std::ostream& BoundaryHistory<LocalVars, RemoteVars, CouplingResult>::print(
    std::ostream& os) const {
  using ::operator<<;
  os << "Integration order: " << integration_order_ << "\n";
  os << "Local Data:\n";
  auto local_value_it = local_data_.second.begin();
  for (auto local_time_it = local_data_.first.begin();
       local_time_it != local_data_.first.end();
       ++local_time_it, ++local_value_it) {
    os << "Time: " << *local_time_it << "\n";
    os << "Data: " << *local_value_it << "\n";
  }
  os << "Remote Data:\n";
  auto remote_value_it = remote_data_.second.begin();
  for (auto remote_time_it = remote_data_.first.begin();
       remote_time_it != remote_data_.first.end();
       ++remote_time_it, ++remote_value_it) {
    os << "Time: " << *remote_time_it << "\n";
    os << "Data: " << *remote_value_it << "\n";
  }
  return os;
}

template <typename LocalVars, typename RemoteVars, typename CouplingResult>
std::ostream& operator<<(
    std::ostream& os,
    const BoundaryHistory<LocalVars, RemoteVars, CouplingResult>& history) {
  return history.print(os);
}
}  // namespace TimeSteppers
