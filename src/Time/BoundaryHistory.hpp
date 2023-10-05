// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <boost/container/static_vector.hpp>
#include <cstddef>
#include <optional>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <type_traits>
#include <utility>

#include "DataStructures/CircularDeque.hpp"
#include "DataStructures/MathWrapper.hpp"
#include "DataStructures/StaticDeque.hpp"
#include "Time/History.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Serialization/PupBoost.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/StlBoilerplate.hpp"
#include "Utilities/TMPL.hpp"

namespace TimeSteppers {

/// \ingroup TimeSteppersGroup
/// Access to the list of `TimeStepId`s in a `BoundaryHistory`.
///
/// For simplicity of implementation, iterable-container access is not
/// provided for substeps within a step, but is instead provided
/// through additional methods on this class.
/// @{
class ConstBoundaryHistoryTimes
    : public stl_boilerplate::RandomAccessSequence<ConstBoundaryHistoryTimes,
                                                   const TimeStepId, false> {
 protected:
  ~ConstBoundaryHistoryTimes() = default;

 public:
  virtual size_t size() const = 0;
  virtual const TimeStepId& operator[](size_t n) const = 0;
  virtual const TimeStepId& operator[](
      const std::pair<size_t, size_t>& step_and_substep) const = 0;
  virtual size_t integration_order(size_t n) const = 0;
  virtual size_t integration_order(const TimeStepId& id) const = 0;
  virtual size_t number_of_substeps(size_t n) const = 0;
  /// This returns the same value for any substep of the same step.
  virtual size_t number_of_substeps(const TimeStepId& id) const = 0;
};

class MutableBoundaryHistoryTimes : public ConstBoundaryHistoryTimes {
 protected:
  ~MutableBoundaryHistoryTimes() = default;

 public:
  /// Remove the earliest step and its substeps.
  virtual void pop_front() const = 0;
  virtual void clear() const = 0;
  /// Remove all substeps for step \p n except for the step itself.
  virtual void clear_substeps(size_t n) const = 0;
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
      return (*this)[{n, 0}];
    }
    const TimeStepId& operator[](
        const std::pair<size_t, size_t>& step_and_substep) const override {
      return entry(step_and_substep).id;
    }

    size_t integration_order(const size_t n) const override {
      return parent_data()[n].integration_order;
    }
    size_t integration_order(const TimeStepId& id) const override {
      return step_data(id).integration_order;
    }

    size_t number_of_substeps(const size_t n) const override {
      return parent_data()[n].substeps.size();
    }
    size_t number_of_substeps(const TimeStepId& id) const override {
      return step_data(id).substeps.size();
    }

    /// Access the data stored on the side.  When performed through a
    /// `MutableSideAccess`, these allow modification of the data.
    /// Performing such modifications likely invalidates the coupling
    /// cache for the associated `BoundaryHistory` object, which
    /// should be cleared.
    /// @{
    Data& data(const size_t n) const {
      return parent_data()[n].substeps.front().data;
    }
    Data& data(const TimeStepId& id) const {
      return entry(id).data;
    }
    /// @}

    /// Apply \p func to each entry.
    ///
    /// The function \p func must accept two arguments, one of type
    /// `const TimeStepId&` and a second of either type `const Data&`
    /// or `gsl::not_null<Data*>`.  (Note that `Data` may be a
    /// const-qualified type.)  If entries are modified, the coupling
    /// cache must be cleared by calling `clear_coupling_cache()` on
    /// the parent `BoundaryHistory` object.
    template <typename Func>
    void for_each(Func&& func) const;

   protected:
    ~SideAccessCommon() = default;

    auto& parent_data() const {
      if constexpr (Local) {
        return parent_->local_data_;
      } else {
        return parent_->remote_data_;
      }
    }

    auto& step_data(const TimeStepId& id) const {
      auto entry =
          std::upper_bound(parent_data().begin(), parent_data().end(), id);
      ASSERT(entry != parent_data().begin(), "Id " << id << " not present.");
      --entry;
      ASSERT(id.substep() < entry->substeps.size() and
             entry->substeps[id.substep()].id == id,
             "Id " << id << " not present.");
      return *entry;
    }

    auto& entry(const TimeStepId& id) const {
      // Bounds and consistency are checked in step_data()
      return step_data(id).substeps[id.substep()];
    }

    auto& entry(const std::pair<size_t, size_t>& step_and_substep) const {
      ASSERT(step_and_substep.first < parent_data().size(),
             "Step out of range: " << step_and_substep.first);
      auto& substeps = parent_data()[step_and_substep.first].substeps;
      ASSERT(step_and_substep.second < substeps.size(),
             "Substep out of range: " << step_and_substep.second);
      return substeps[step_and_substep.second];
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
    void clear_substeps(size_t n) const override;

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
    struct Entry {
      TimeStepId id;
      Data data;

      void pup(PUP::er& p) {
        p | id;
        p | data;
      }
    };

    size_t integration_order;
    // Unlike in History, the full step is the first entry, so we need
    // one more element.
    boost::container::static_vector<Entry, history_max_substeps + 1> substeps;

    void pup(PUP::er& p) {
      p | integration_order;
      p | substeps;
    }

    friend bool operator<(const StepData& a, const StepData& b) {
      return a.substeps.front().id < b.substeps.front().id;
    }
    friend bool operator<(const TimeStepId& a, const StepData& b) {
      return a < b.substeps.front().id;
    }
    friend bool operator<(const StepData& a, const TimeStepId& b) {
      return a.substeps.front().id < b;
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

  void clear_substeps_local(size_t n);
  void clear_substeps_remote(size_t n);

  StaticDeque<StepData<LocalData>, history_max_past_steps + 2> local_data_{};
  CircularDeque<StepData<RemoteData>> remote_data_{};

  template <typename Data>
  using CouplingSubsteps =
      boost::container::static_vector<Data, history_max_substeps + 1>;

  // Putting the CircularDeque outermost means that we are inserting
  // and removing containers that do not allocate, so we don't have to
  // worry about that.
  // NOLINTNEXTLINE(spectre-mutable)
  mutable CircularDeque<CouplingSubsteps<
      StaticDeque<CouplingSubsteps<std::optional<CouplingResult>>,
                  decltype(local_data_)::max_size()>>>
      couplings_;
};

template <typename LocalData, typename RemoteData, typename CouplingResult>
template <bool Local, bool Mutable>
template <typename Func>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::SideAccessCommon<
    Local, Mutable>::for_each(Func&& func) const {
  for (auto& step : parent_data()) {
    for (auto& substep : step.substeps) {
      if constexpr (std::is_invocable_v<Func&, const TimeStepId&,
                                        const Data&>) {
        func(std::as_const(substep.id), std::as_const(substep.data));
      } else {
        func(std::as_const(substep.id), make_not_null(&substep.data));
      }
    }
  }
}

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
    Local>::clear_substeps(const size_t n) const {
  if constexpr (Local) {
    this->parent_->clear_substeps_local(n);
  } else {
    this->parent_->clear_substeps_remote(n);
  }
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
template <bool Local>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::MutableSideAccess<
    Local>::insert(const TimeStepId& id, const size_t integration_order,
                   Data data) const {
  ASSERT(this->parent_data().empty() or
             id > this->parent_data().back().substeps.back().id,
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
             id < this->parent_data().front().substeps.front().id,
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
  auto local_entry = std::upper_bound(
      parent_->local_data_.begin(), parent_->local_data_.end(), local_id);
  ASSERT(local_entry != parent_->local_data_.begin(), "local_id not present");
  --local_entry;
  ASSERT(local_id.substep() < local_entry->substeps.size() and
         local_entry->substeps[local_id.substep()].id == local_id,
         "local_id not present");
  const auto local_step_offset =
      static_cast<size_t>(local_entry - parent_->local_data_.begin());

  auto remote_entry = std::upper_bound(
      parent_->remote_data_.begin(), parent_->remote_data_.end(), remote_id);
  ASSERT(remote_entry != parent_->remote_data_.begin(),
         "remote_id not present");
  --remote_entry;
  ASSERT(remote_id.substep() < remote_entry->substeps.size() and
         remote_entry->substeps[remote_id.substep()].id == remote_id,
         "remote_id not present");
  const auto remote_step_offset =
      static_cast<size_t>(remote_entry - parent_->remote_data_.begin());

  auto& coupling_entry = parent_->couplings_[remote_step_offset]
                             [remote_id.substep()][local_step_offset]
                             [local_id.substep()];
  if (not coupling_entry.has_value()) {
    coupling_entry.emplace(coupling_(
        local_entry->substeps[local_id.substep()].data,
        remote_entry->substeps[remote_id.substep()].data));
  }
  return make_math_wrapper(*coupling_entry);
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData,
                     CouplingResult>::clear_coupling_cache() {
  for (auto& remote_step : couplings_) {
    for (auto& remote_substep : remote_step) {
      for (auto& local_step : remote_substep) {
        for (auto& local_substep : local_step) {
          local_substep.reset();
        }
      }
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
    for (size_t step = 0; step < times.size(); ++step) {
      const size_t number_of_substeps = times.number_of_substeps(step);
      for (size_t substep = 0; substep < number_of_substeps; ++substep) {
        const auto id = times[{step, substep}];
        os << pad << " Time: " << id;
        if (substep == 0) {
          os << " (order " << times.integration_order(step) << ")";
        }
        os << "\n";
        if constexpr (IncludeData) {
          os << pad << "  Data: ";
          // os << times.data(id) fails to compile on gcc-11
          print_stl(os, times.data(id));
          os << "\n";
        }
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
  if (id.substep() == 0) {
    local_data_.push_back({integration_order, {}});
  } else {
    ASSERT(integration_order == local_data_.back().integration_order,
           "Cannot change integration order during a step.");
  }
  local_data_.back().substeps.push_back({id, std::move(data)});
  for (auto& remote_step : couplings_) {
    for (auto& remote_substep : remote_step) {
      if (id.substep() == 0) {
        remote_substep.emplace_back(1_st);
      } else {
        remote_substep.back().emplace_back();
      }
    }
  }
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::insert_remote(
    const TimeStepId& id, const size_t integration_order, RemoteData data) {
  if (id.substep() == 0) {
    remote_data_.push_back({integration_order, {}});
  } else {
    ASSERT(integration_order == remote_data_.back().integration_order,
           "Cannot change integration order during a step.");
  }
  remote_data_.back().substeps.push_back({id, std::move(data)});
  if (id.substep() == 0) {
    couplings_.emplace_back(1_st);
  } else {
    couplings_.back().emplace_back();
  }
  for (const auto& local_step : local_data_) {
    couplings_.back().back().emplace_back(local_step.substeps.size());
  }
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::
    insert_initial_local(const TimeStepId& id, const size_t integration_order,
                         LocalData data) {
  local_data_.push_front({integration_order, {}});
  local_data_.front().substeps.push_back({id, std::move(data)});
  for (auto& remote_step : couplings_) {
    for (auto& remote_substep : remote_step) {
      remote_substep.emplace_front(1_st);
    }
  }
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::
    insert_initial_remote(const TimeStepId& id, const size_t integration_order,
                          RemoteData data) {
  remote_data_.push_front({integration_order, {}});
  remote_data_.front().substeps.push_back({id, std::move(data)});
  couplings_.emplace_front(1_st);
  for (const auto& local_step : local_data_) {
    couplings_.front().back().emplace_back(local_step.substeps.size());
  }
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::pop_local() {
  local_data_.pop_front();
  for (auto& remote_step : couplings_) {
    for (auto& remote_substep : remote_step) {
      remote_substep.pop_front();
    }
  }
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData, CouplingResult>::pop_remote() {
  remote_data_.pop_front();
  couplings_.pop_front();
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData,
                     CouplingResult>::clear_substeps_local(const size_t n) {
  local_data_[n].substeps.erase(local_data_[n].substeps.begin() + 1,
                                local_data_[n].substeps.end());
  for (auto& remote_step : couplings_) {
    for (auto& remote_substep : remote_step) {
      auto& local_step = remote_substep[n];
      local_step.erase(local_step.begin() + 1, local_step.end());
    }
  }
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
void BoundaryHistory<LocalData, RemoteData,
                     CouplingResult>::clear_substeps_remote(const size_t n) {
  remote_data_[n].substeps.erase(remote_data_[n].substeps.begin() + 1,
                                 remote_data_[n].substeps.end());
  auto& remote_step = couplings_[n];
  remote_step.erase(remote_step.begin() + 1, remote_step.end());
}

template <typename LocalData, typename RemoteData, typename CouplingResult>
std::ostream& operator<<(
    std::ostream& os,
    const BoundaryHistory<LocalData, RemoteData, CouplingResult>& history) {
  return history.template print<true>(os);
}
}  // namespace TimeSteppers
