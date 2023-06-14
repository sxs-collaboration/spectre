// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "DataStructures/CircularDeque.hpp"
#include "DataStructures/MathWrapper.hpp"
#include "DataStructures/StaticDeque.hpp"
#include "Time/History.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/StlBoilerplate.hpp"
#include "Utilities/TMPL.hpp"

namespace TimeSteppers {

/// \ingroup TimeSteppersGroup
/// Access to the list of `TimeStepId`s in a `BoundaryHistory`.
/// @{
class ConstBoundaryHistoryTimes
    : public stl_boilerplate::RandomAccessSequence<ConstBoundaryHistoryTimes,
                                                   const TimeStepId, false> {
 protected:
  ~ConstBoundaryHistoryTimes() = default;

 public:
  virtual size_t size() const = 0;
  virtual const TimeStepId& operator[](size_t n) const = 0;
  virtual size_t integration_order(size_t n) const = 0;
  virtual size_t integration_order(const TimeStepId& id) const = 0;
};

class MutableBoundaryHistoryTimes : public ConstBoundaryHistoryTimes {
 protected:
  ~MutableBoundaryHistoryTimes() = default;

 public:
  virtual void pop_front() const = 0;
  virtual void clear() const = 0;
};
/// @}

/// \ingroup TimeSteppersGroup
/// Type erased base class for evaluating BoundaryHistory couplings.
///
/// The results are cached in the `BoundaryHistory` class.
template <typename UntypedCouplingResult>
class BoundaryHistoryEvaluator {
 public:
  virtual MathWrapper<const UntypedCouplingResult> operator()(
      const TimeStepId& local_id, const TimeStepId& remote_id) const = 0;

 protected:
  ~BoundaryHistoryEvaluator() = default;
};

/// \ingroup TimeSteppersGroup
/// History data used by a TimeStepper for boundary integration.
///
/// \tparam LocalData local data passed to the boundary coupling
/// \tparam RemoteData remote data passed to the boundary coupling
/// \tparam CouplingResult type of cached boundary couplings
template <typename LocalData, typename RemoteData, typename CouplingResult>
class BoundaryHistory {
 public:
  BoundaryHistory() = default;
  BoundaryHistory(const BoundaryHistory& other) = default;
  BoundaryHistory(BoundaryHistory&&) = default;
  BoundaryHistory& operator=(const BoundaryHistory& other) = default;
  BoundaryHistory& operator=(BoundaryHistory&&) = default;
  ~BoundaryHistory() = default;

  /// The wrapped types presented by the type-erased history.  One of
  /// the types in \ref MATH_WRAPPER_TYPES.
  using UntypedCouplingResult = math_wrapper_type<CouplingResult>;

  // Factored out of ConstSideAccess so that the base classes of
  // MutableSideAccess can have protected destructors.
  template <bool Local, bool Mutable>
  class SideAccessCommon
      : public tmpl::conditional_t<Mutable, MutableBoundaryHistoryTimes,
                                   ConstBoundaryHistoryTimes> {
   public:
    using MutableData = tmpl::conditional_t<Local, LocalData, RemoteData>;
    using Data = tmpl::conditional_t<Mutable, MutableData, const MutableData>;

    size_t size() const override { return parent_data().size(); }
    static constexpr size_t max_size() {
      return decltype(std::declval<ConstSideAccess>()
                          .parent_data())::max_size();
    }

    const TimeStepId& operator[](const size_t n) const override {
      return parent_data()[n].time_step_id;
    }
    size_t integration_order(const size_t n) const override {
      return parent_data()[n].integration_order;
    }
    size_t integration_order(const TimeStepId& id) const override {
      return entry_from_id(id).integration_order;
    }

    /// Access the data stored on the side.  When performed through a
    /// `MutableSideAccess`, these allow modification of the data.
    /// Performing such modifications likely invalidates the coupling
    /// cache for the associated `BoundaryHistory` object, which
    /// should be cleared.
    /// @{
    Data& data(const size_t n) const { return parent_data()[n].data; }
    Data& data(const TimeStepId& id) const { return entry_from_id(id).data; }
    /// @}

   protected:
    ~SideAccessCommon() = default;

    auto& parent_data() const {
      if constexpr (Local) {
        return parent_->local_data_;
      } else {
        return parent_->remote_data_;
      }
    }

    auto& entry_from_id(const TimeStepId& id) const {
      const auto entry =
          std::lower_bound(parent_data().begin(), parent_data().end(), id);
      ASSERT(entry != parent_data().end() and entry->time_step_id == id,
             "Id " << id << " not present.");
      return *entry;
    }

    using StoredHistory =
        tmpl::conditional_t<Mutable, BoundaryHistory, const BoundaryHistory>;
    explicit SideAccessCommon(const gsl::not_null<StoredHistory*> parent)
        : parent_(parent) {}

    gsl::not_null<StoredHistory*> parent_;
  };

  /// \cond
  template <bool Local>
  class ConstSideAccess;
  /// \endcond

  template <bool Local>
  class MutableSideAccess final : public SideAccessCommon<Local, true> {
   public:
    using Data = tmpl::conditional_t<Local, LocalData, RemoteData>;

    void pop_front() const override;
    void clear() const override;

    void insert(const TimeStepId& id, size_t integration_order,
                Data data) const;

    void insert_initial(const TimeStepId& id, size_t integration_order,
                        Data data) const;

   private:
    friend class BoundaryHistory;
    friend class ConstSideAccess<Local>;
    explicit MutableSideAccess(const gsl::not_null<BoundaryHistory*> parent)
        : SideAccessCommon<Local, true>(parent) {}
  };

  template <bool Local>
  class ConstSideAccess final : public SideAccessCommon<Local, false> {
   private:
    friend class BoundaryHistory;
    explicit ConstSideAccess(const gsl::not_null<const BoundaryHistory*> parent)
        : SideAccessCommon<Local, false>(parent) {}
  };

  MutableSideAccess<true> local() { return MutableSideAccess<true>(this); }
  ConstSideAccess<true> local() const { return ConstSideAccess<true>(this); }

  MutableSideAccess<false> remote() { return MutableSideAccess<false>(this); }
  ConstSideAccess<false> remote() const { return ConstSideAccess<false>(this); }

 private:
  template <typename Coupling>
  class EvaluatorImpl final
      : public BoundaryHistoryEvaluator<UntypedCouplingResult> {
   public:
    MathWrapper<const UntypedCouplingResult> operator()(
        const TimeStepId& local_id, const TimeStepId& remote_id) const;

   private:
    friend class BoundaryHistory;

    EvaluatorImpl(const gsl::not_null<const BoundaryHistory*> parent,
                  Coupling coupling)
        : parent_(parent), coupling_(std::move(coupling)) {}

    gsl::not_null<const BoundaryHistory*> parent_;
    Coupling coupling_;
  };

 public:
  /// Obtain an object that can evaluate type-erased boundary
  /// couplings.
  ///
  /// The passed functor must take objects of types `LocalData` and
  /// `RemoteData` and return an object convertible to
  /// `CouplingResult`.  Results are cached, so different calls to
  /// this function should pass equivalent couplings.
  template <typename Coupling>
  auto evaluator(Coupling&& coupling) const {
    return EvaluatorImpl<Coupling>(this, std::forward<Coupling>(coupling));
  }

  /// Clear the cached values.
  ///
  /// This is required after existing history entries that have been
  /// used in coupling calculations are mutated.
  void clear_coupling_cache();

  void pup(PUP::er& p);

  template <bool IncludeData>
  std::ostream& print(std::ostream& os, size_t padding_size = 0) const;

 private:
  template <typename Data>
  struct StepData {
    size_t integration_order{};
    TimeStepId time_step_id;
    Data data;

    void pup(PUP::er& p) {
      p | integration_order;
      p | time_step_id;
      p | data;
    }

    friend bool operator<(const StepData& a, const StepData& b) {
      return a.time_step_id < b.time_step_id;
    }
    friend bool operator<(const TimeStepId& a, const StepData& b) {
      return a < b.time_step_id;
    }
    friend bool operator<(const StepData& a, const TimeStepId& b) {
      return a.time_step_id < b;
    }
  };

  void insert_local(const TimeStepId& id, size_t integration_order,
                    LocalData data);
  void insert_remote(const TimeStepId& id, size_t integration_order,
                     RemoteData data);

  void insert_initial_local(const TimeStepId& id, size_t integration_order,
                            LocalData data);
  void insert_initial_remote(const TimeStepId& id, size_t integration_order,
                             RemoteData data);

  void pop_local();
  void pop_remote();

  StaticDeque<StepData<LocalData>, history_max_past_steps + 2> local_data_{};
  CircularDeque<StepData<RemoteData>> remote_data_{};

  // Putting the CircularDeque outermost means that we are inserting
  // and removing containers that do not allocate, so we don't have to
  // worry about that.
  // NOLINTNEXTLINE(spectre-mutable)
  mutable CircularDeque<StaticDeque<std::optional<CouplingResult>,
                                    decltype(local_data_)::max_size()>>
      couplings_;
};

template <typename LocalData, typename RemoteData, typename CouplingResult>
template <bool Local>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::MutableSideAccess<
    Local>::pop_front() const {
  if constexpr (Local) {
    this->parent_->pop_local();
  } else {
    this->parent_->pop_remote();
  }
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
template <bool Local>
void BoundaryHistory<LocalData, RemoteData,
                     CouplingResult>::MutableSideAccess<Local>::clear() const {
  while (not this->empty()) {
    pop_front();
  }
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
template <bool Local>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::MutableSideAccess<
    Local>::insert(const TimeStepId& id, const size_t integration_order,
                   Data data) const {
  ASSERT(id.substep() == 0, "Substeps not yet implemented.");
  ASSERT(this->parent_data().empty() or
             id > this->parent_data().back().time_step_id,
         "New data not newer than current data.");
  if constexpr (Local) {
    this->parent_->insert_local(id, integration_order, std::move(data));
  } else {
    this->parent_->insert_remote(id, integration_order, std::move(data));
  }
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
template <bool Local>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::MutableSideAccess<
    Local>::insert_initial(const TimeStepId& id, const size_t integration_order,
                           Data data) const {
  ASSERT(id.substep() == 0, "Cannot insert_initial with substeps.");
  ASSERT(this->parent_data().empty() or
             id < this->parent_data().front().time_step_id,
         "New data not older than current data.");
  if constexpr (Local) {
    this->parent_->insert_initial_local(id, integration_order, std::move(data));
  } else {
    this->parent_->insert_initial_remote(id, integration_order,
                                         std::move(data));
  }
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
template <typename Coupling>
auto BoundaryHistory<LocalData, RemoteData, CouplingResult>::EvaluatorImpl<
    Coupling>::operator()(const TimeStepId& local_id,
                          const TimeStepId& remote_id) const
    -> MathWrapper<const UntypedCouplingResult> {
  const auto local_entry = std::lower_bound(
      parent_->local_data_.begin(), parent_->local_data_.end(), local_id);
  ASSERT(local_entry != parent_->local_data_.end() and
             local_entry->time_step_id == local_id,
         "local_id not present");
  const auto remote_entry = std::lower_bound(
      parent_->remote_data_.begin(), parent_->remote_data_.end(), remote_id);
  ASSERT(remote_entry != parent_->remote_data_.end() and
             remote_entry->time_step_id == remote_id,
         "remote_id not present");

  auto& coupling_entry =
      parent_->couplings_
          [static_cast<size_t>(remote_entry - parent_->remote_data_.begin())]
          [static_cast<size_t>(local_entry - parent_->local_data_.begin())];
  if (not coupling_entry.has_value()) {
    coupling_entry.emplace(coupling_(local_entry->data, remote_entry->data));
  }
  return make_math_wrapper(*coupling_entry);
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData,
                     CouplingResult>::clear_coupling_cache() {
  for (auto& slice : couplings_) {
    for (auto& entry : slice) {
      entry.reset();
    }
  }
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::pup(PUP::er& p) {
  p | local_data_;
  p | remote_data_;
  p | couplings_;
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
template <bool IncludeData>
std::ostream& BoundaryHistory<LocalData, RemoteData, CouplingResult>::print(
    std::ostream& os, const size_t padding_size) const {
  const std::string pad(padding_size, ' ');
  using ::operator<<;
  const auto do_print = [&os, &pad](const auto& times) {
    for (const auto& step_id : times) {
      os << pad << " Time: " << step_id << " (order "
         << times.integration_order(step_id) << ")\n";
      if constexpr (IncludeData) {
        os << pad << "  Data: ";
        // os << times.data(step_id) fails to compile on gcc-11
        print_stl(os, times.data(step_id));
        os << "\n";
      }
    }
  };
  os << pad << "Local Data:\n";
  do_print(local());
  os << pad << "Remote Data:\n";
  do_print(remote());
  return os;
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::insert_local(
    const TimeStepId& id, const size_t integration_order, LocalData data) {
  local_data_.push_back({integration_order, id, std::move(data)});
  alg::for_each(couplings_, [](auto& x) { x.emplace_back(); });
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::insert_remote(
    const TimeStepId& id, const size_t integration_order, RemoteData data) {
  remote_data_.push_back({integration_order, id, std::move(data)});
  couplings_.emplace_back(local_data_.size());
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::
    insert_initial_local(const TimeStepId& id, const size_t integration_order,
                         LocalData data) {
  local_data_.push_front({integration_order, id, std::move(data)});
  alg::for_each(couplings_, [](auto& x) { x.emplace_front(); });
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::
    insert_initial_remote(const TimeStepId& id, const size_t integration_order,
                          RemoteData data) {
  remote_data_.push_front({integration_order, id, std::move(data)});
  couplings_.emplace_front(local_data_.size());
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::pop_local() {
  local_data_.pop_front();
  alg::for_each(couplings_, [](auto& x) { x.pop_front(); });
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::pop_remote() {
  remote_data_.pop_front();
  couplings_.pop_front();
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
std::ostream& operator<<(
    std::ostream& os,
    const BoundaryHistory<LocalData, RemoteData, CouplingResult>& history) {
  return history.template print<true>(os);
}
}  // namespace TimeSteppers
