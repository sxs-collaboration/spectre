// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/container/static_vector.hpp>
#include <cstddef>
#include <iosfwd>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/CircularDeque.hpp"
#include "DataStructures/MathWrapper.hpp"
#include "Time/History.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StlBoilerplate.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace TimeSteppers {
namespace BoundaryHistory_detail {
template <typename Data>
struct StepData {
  struct Entry {
    TimeStepId id;
    Data data;

    void pup(PUP::er& p);
  };

  size_t integration_order;
  // Unlike in History, the full step is the first entry, so we need
  // one more element.
  boost::container::static_vector<Entry, history_max_substeps + 1> substeps;

  void pup(PUP::er& p);
};
}  // namespace BoundaryHistory_detail

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
  virtual const UntypedCouplingResult& operator()(
      const TimeStepId& local_id, const TimeStepId& remote_id) const = 0;

 protected:
  ~BoundaryHistoryEvaluator() = default;
};

/// \ingroup TimeSteppersGroup
/// History data used by a TimeStepper for boundary integration.
///
/// \tparam LocalData local data passed to the boundary coupling
/// \tparam RemoteData remote data passed to the boundary coupling
/// \tparam UntypedCouplingResult math_wrapper_type of cached boundary couplings
template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
class BoundaryHistory {
  static_assert(tmpl::list_contains_v<tmpl::list<MATH_WRAPPER_TYPES>,
                                      UntypedCouplingResult>);

 public:
  BoundaryHistory() = default;
  BoundaryHistory(const BoundaryHistory& other) = default;
  BoundaryHistory(BoundaryHistory&&) = default;
  BoundaryHistory& operator=(const BoundaryHistory& other) = default;
  BoundaryHistory& operator=(BoundaryHistory&&) = default;
  ~BoundaryHistory() = default;

  // Factored out of ConstSideAccess so that the base classes of
  // MutableSideAccess can have protected destructors.
  template <bool Local, bool Mutable>
  class SideAccessCommon
      : public tmpl::conditional_t<Mutable, MutableBoundaryHistoryTimes,
                                   ConstBoundaryHistoryTimes> {
   public:
    using MutableData = tmpl::conditional_t<Local, LocalData, RemoteData>;
    using Data = tmpl::conditional_t<Mutable, MutableData, const MutableData>;

    size_t size() const override;
    static constexpr size_t max_size() {
      return std::remove_cvref_t<decltype(std::declval<ConstSideAccess<Local>>()
                                              .parent_data())>::max_size();
    }

    const TimeStepId& operator[](size_t n) const override;
    const TimeStepId& operator[](
        const std::pair<size_t, size_t>& step_and_substep) const override;

    size_t integration_order(size_t n) const override;
    size_t integration_order(const TimeStepId& id) const override;

    size_t number_of_substeps(size_t n) const override;
    size_t number_of_substeps(const TimeStepId& id) const override;

    /// Access the data stored on the side.  When performed through a
    /// `MutableSideAccess`, these allow modification of the data.
    /// Performing such modifications likely invalidates the coupling
    /// cache for the associated `BoundaryHistory` object, which
    /// should be cleared.
    /// @{
    Data& data(size_t n) const;
    Data& data(const TimeStepId& id) const;
    /// @}

    /// Apply \p func to each entry.
    ///
    /// The function \p func must accept two arguments, one of type
    /// `const TimeStepId&` and a second of either type `const Data&`
    /// or `gsl::not_null<Data*>`, with the `not_null` version only
    /// available if this is a `MutableSideAccess`.  If \p func takes
    /// a `not_null`, it must return a `bool` indicating if it
    /// modified the entry.  If any entries are modified, the coupling
    /// cache of parent `BoundaryHistory` will be cleared.
    template <typename Func>
    void for_each(Func&& func) const;

   protected:
    ~SideAccessCommon() = default;

    tmpl::conditional_t<
        Mutable, CircularDeque<BoundaryHistory_detail::StepData<MutableData>>&,
        const CircularDeque<BoundaryHistory_detail::StepData<MutableData>>&>
    parent_data() const;

    auto& step_data(const TimeStepId& id) const;

    auto& entry(const TimeStepId& id) const;

    auto& entry(const std::pair<size_t, size_t>& step_and_substep) const;

    using StoredHistory =
        tmpl::conditional_t<Mutable, BoundaryHistory, const BoundaryHistory>;
    explicit SideAccessCommon(gsl::not_null<StoredHistory*> parent);

    gsl::not_null<StoredHistory*> parent_;
  };

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
    explicit MutableSideAccess(gsl::not_null<BoundaryHistory*> parent);
  };

  template <bool Local>
  class ConstSideAccess final : public SideAccessCommon<Local, false> {
   private:
    friend class BoundaryHistory;
    explicit ConstSideAccess(gsl::not_null<const BoundaryHistory*> parent);
  };

  MutableSideAccess<true> local();
  ConstSideAccess<true> local() const;

  MutableSideAccess<false> remote();
  ConstSideAccess<false> remote() const;

 private:
  template <typename Coupling>
  class EvaluatorImpl final
      : public BoundaryHistoryEvaluator<UntypedCouplingResult> {
   public:
    const UntypedCouplingResult& operator()(
        const TimeStepId& local_id, const TimeStepId& remote_id) const override;

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
  /// `RemoteData` and return an object with math_wrapper_type
  /// `UntypedCouplingResult`.  Results are cached, so different calls
  /// to this function should pass equivalent couplings.
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

  std::tuple<std::optional<UntypedCouplingResult>&, const LocalData&,
             const RemoteData&>
  find_cache_entry(const TimeStepId& local_id,
                   const TimeStepId& remote_id) const;

  CircularDeque<BoundaryHistory_detail::StepData<LocalData>> local_data_{};
  CircularDeque<BoundaryHistory_detail::StepData<RemoteData>> remote_data_{};

  template <typename Data>
  using CouplingSubsteps =
      boost::container::static_vector<Data, history_max_substeps + 1>;

  // NOLINTNEXTLINE(spectre-mutable)
  mutable CircularDeque<CouplingSubsteps<
      CircularDeque<CouplingSubsteps<std::optional<UntypedCouplingResult>>>>>
      couplings_;
};

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local, bool Mutable>
template <typename Func>
void BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    SideAccessCommon<Local, Mutable>::for_each(Func&& func) const {
  bool entries_changed = false;
  for (auto& step : parent_data()) {
    for (auto& substep : step.substeps) {
      if constexpr (std::is_invocable_v<Func&, const TimeStepId&,
                                        const Data&>) {
        func(std::as_const(substep.id), std::as_const(substep.data));
      } else {
        static_assert(Mutable,
                      "Cannot perform mutating for_each on a ConstSideAccess");
        if (func(std::as_const(substep.id), make_not_null(&substep.data))) {
          entries_changed = true;
        }
      }
    }
  }
  if constexpr (Mutable) {
    if (entries_changed) {
      // A minor optimization would be to only clear the cache entries
      // that have actually been invalidated, but most things that
      // modify the history modify all the entries.
      parent_->clear_coupling_cache();
    }
  }
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <typename Coupling>
auto BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    EvaluatorImpl<Coupling>::operator()(const TimeStepId& local_id,
                                        const TimeStepId& remote_id) const
    -> const UntypedCouplingResult& {
  const auto [coupling_entry, local_data, remote_data] =
      parent_->find_cache_entry(local_id, remote_id);
  if (not coupling_entry.has_value()) {
    auto new_entry = coupling_(local_data, remote_data);
    static_assert(std::is_same_v<math_wrapper_type<decltype(new_entry)>,
                                 UntypedCouplingResult>);
    coupling_entry.emplace(into_math_wrapper_type(std::move(new_entry)));
  }
  return *coupling_entry;
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
std::ostream& operator<<(std::ostream& os,
                         const BoundaryHistory<LocalData, RemoteData,
                                               UntypedCouplingResult>& history);
}  // namespace TimeSteppers

// Documentation for macro defined in the tpp file
#ifdef SPECTRE_DOXYGEN_INVOKED
/// \ingroup TimeSteppersGroup
/// Explicitly instantiate BoundaryHistory and helpers.  Should be
/// called with the template arguments for `BoundaryHistory`.
#define INSTANTIATE_BOUNDARY_HISTORY(...) UNSPECIFIED
#endif  // SPECTRE_DOXYGEN_INVOKED
