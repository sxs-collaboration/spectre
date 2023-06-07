// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/container/static_vector.hpp>
#include <cstddef>
#include <optional>
#include <pup.h>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/MathWrapper.hpp"
#include "DataStructures/StaticDeque.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ContainsAllocations.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/StlBoilerplate.hpp"
#include "Utilities/TMPL.hpp"

namespace TimeSteppers {

/// Largest number of past steps supported by the time stepper
/// `History`.  Corresponds to the `number_of_past_steps()` method of
/// `TimeStepper`.  AdamsBashforth with order 8 has the largest
/// requirement.
constexpr size_t history_max_past_steps = 7;
/// Largest number of substeps supported by the time stepper
/// `History`.  Corresponds to the `number_of_substeps()` and
/// `number_of_substeps_for_error()` methods of `TimeStepper`.
/// DormandPrince5 with an error estimate and Rk5Owren have the
/// largest requirement.
constexpr size_t history_max_substeps = 6;

/// \ingroup TimeSteppersGroup
/// Entry in the time-stepper history, in type-erased form.
///
/// The history access classes do not provide mutable references to
/// these structs, so they cannot be used to modify history data.
///
/// This struct mirrors the typed `StepRecord` struct.  See that for
/// details.
///
/// \tparam T One of the types in \ref MATH_WRAPPER_TYPES
template <typename T>
struct UntypedStepRecord {
  TimeStepId time_step_id;
  std::optional<T> value;
  T derivative;

  static_assert(tmpl::list_contains_v<tmpl::list<MATH_WRAPPER_TYPES>, T>);
};

template <typename T>
bool operator==(const UntypedStepRecord<T>& a, const UntypedStepRecord<T>& b);

template <typename T>
bool operator!=(const UntypedStepRecord<T>& a, const UntypedStepRecord<T>& b);

#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif  // defined(__GNUC__) and not defined(__clang__)
/// \ingroup TimeSteppersGroup
/// Access to the history data used by a TimeStepper in type-erased
/// form.  Obtain an instance with `History::untyped()`.
///
/// The methods mirror similar ones in `History`.  See that class for
/// details.
///
/// \tparam T One of the types in \ref MATH_WRAPPER_TYPES
template <typename T>
class ConstUntypedHistory
    : public stl_boilerplate::RandomAccessSequence<
          ConstUntypedHistory<T>, const UntypedStepRecord<T>, false> {
  static_assert(tmpl::list_contains_v<tmpl::list<MATH_WRAPPER_TYPES>, T>);

 protected:
  ConstUntypedHistory() = default;
  ConstUntypedHistory(const ConstUntypedHistory&) = default;
  ConstUntypedHistory(ConstUntypedHistory&&) = default;
  ConstUntypedHistory& operator=(const ConstUntypedHistory&) = default;
  ConstUntypedHistory& operator=(ConstUntypedHistory&&) = default;
  ~ConstUntypedHistory() = default;

 public:
  using WrapperType = T;

  /// \cond
  class UntypedSubsteps
      : public stl_boilerplate::RandomAccessSequence<
            UntypedSubsteps, const UntypedStepRecord<T>, true> {
   public:
    UntypedSubsteps() = delete;
    UntypedSubsteps(const UntypedSubsteps&) = delete;
    UntypedSubsteps(UntypedSubsteps&&) = default;
    UntypedSubsteps& operator=(const UntypedSubsteps&) = delete;
    UntypedSubsteps& operator=(UntypedSubsteps&&) = default;
    ~UntypedSubsteps() = default;

    using WrapperType = T;

    size_t size() const;
    size_t max_size() const;

    const UntypedStepRecord<T>& operator[](const size_t index) const;

   private:
    friend ConstUntypedHistory;
    explicit UntypedSubsteps(const ConstUntypedHistory& history);

    gsl::not_null<const ConstUntypedHistory*> history_;
  };
  /// \endcond

  virtual size_t integration_order() const = 0;

  virtual size_t size() const = 0;
  virtual size_t max_size() const = 0;

  virtual const UntypedStepRecord<T>& operator[](size_t index) const = 0;

  virtual const UntypedStepRecord<T>& operator[](
      const TimeStepId& id) const = 0;

  virtual bool at_step_start() const = 0;

  UntypedSubsteps substeps() const;

 private:
  friend UntypedSubsteps;
  virtual const boost::container::static_vector<UntypedStepRecord<WrapperType>,
                                                history_max_substeps>&
  substep_values() const = 0;
};

/// \ingroup TimeSteppersGroup
/// Mutable access to the history data used by a TimeStepper in
/// type-erased form.  Obtain an instance with `History::untyped()`.
///
/// Data cannot be inserted or modified through the type-erased
/// interface.  The only mutability exposed is the ability to delete
/// data.
///
/// The methods mirror similar ones in `History`.  See that class for
/// details.
///
/// \tparam T One of the types in \ref MATH_WRAPPER_TYPES
template <typename T>
class MutableUntypedHistory : public ConstUntypedHistory<T> {
  static_assert(tmpl::list_contains_v<tmpl::list<MATH_WRAPPER_TYPES>, T>);

 protected:
  MutableUntypedHistory() = default;
  MutableUntypedHistory(const MutableUntypedHistory&) = default;
  MutableUntypedHistory(MutableUntypedHistory&&) = default;
  MutableUntypedHistory& operator=(const MutableUntypedHistory&) = default;
  MutableUntypedHistory& operator=(MutableUntypedHistory&&) = default;
  ~MutableUntypedHistory() = default;

 public:
  virtual void discard_value(const TimeStepId& id_to_discard) const = 0;

  virtual void pop_front() const = 0;

  virtual void clear_substeps() const = 0;
};
#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) and not defined(__clang__)

/// \ingroup TimeSteppersGroup
/// Data in an entry of the time-stepper history.
///
/// The `value` field may be empty if the value has been discarded to
/// save memory.  See `History::discard_value` and
/// `ConstUntypedHistory::discard_value`.
template <typename Vars>
struct StepRecord {
  using DerivVars = db::prefix_variables<::Tags::dt, Vars>;

  TimeStepId time_step_id;
  std::optional<Vars> value;
  DerivVars derivative;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
};

template <typename Vars>
void StepRecord<Vars>::pup(PUP::er& p) {
  p | time_step_id;
  p | value;
  p | derivative;
}

template <typename Vars>
bool operator==(const StepRecord<Vars>& a, const StepRecord<Vars>& b) {
  return a.time_step_id == b.time_step_id and a.value == b.value and
         a.derivative == b.derivative;
}

template <typename Vars>
bool operator!=(const StepRecord<Vars>& a, const StepRecord<Vars>& b) {
  return not(a == b);
}

/// \cond
template <typename Vars>
class History;
/// \endcond

namespace History_detail {
// Find a record from either the steps or substeps.  This takes the
// arguments it does so it can be used to find both const and
// non-const values from both typed and untyped histories.
template <typename History>
decltype(auto) find_record(History&& history, const TimeStepId& id) {
  const size_t substep = id.substep();
  if (substep == 0) {
    for (size_t i = 0; i < history.size(); ++i) {
      auto& record = history[i];
      if (record.time_step_id == id) {
        return record;
      }
    }
    ERROR(id << " not present");
  } else {
    ASSERT(substep - 1 < history.substeps().size(), id << " not present");
    auto& record = history.substeps()[substep - 1];
    ASSERT(record.time_step_id == id, id << " not present");
    return record;
  }
}

#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#endif  // defined(__GNUC__) and not defined(__clang__)
template <typename UntypedBase>
class UntypedAccessCommon : public UntypedBase {
 protected:
  UntypedAccessCommon() = delete;
  UntypedAccessCommon(const UntypedAccessCommon&) = delete;
  UntypedAccessCommon(UntypedAccessCommon&&) = default;
  UntypedAccessCommon& operator=(const UntypedAccessCommon&) = delete;
  // Can't move-assign non-owning DataVectors.
  UntypedAccessCommon& operator=(UntypedAccessCommon&&) = delete;
  ~UntypedAccessCommon() = default;

 public:
  using WrapperType = typename UntypedBase::WrapperType;

  size_t integration_order() const override;

  size_t size() const override;
  size_t max_size() const override;

  const UntypedStepRecord<WrapperType>& operator[](
      const size_t index) const override;
  const UntypedStepRecord<WrapperType>& operator[](
      const TimeStepId& id) const override;

  bool at_step_start() const override;

 protected:
  template <typename Vars>
  UntypedAccessCommon(const History<Vars>& history) {
    reinitialize(history);
  }

  template <typename Vars>
  void reinitialize(const History<Vars>& history) const {
    // This class is presenting a view.  Regenerating its internal
    // representation after a change to the parent structure is an
    // implementation detail.
    auto* const mutable_this = const_cast<UntypedAccessCommon*>(this);
    mutable_this->integration_order_ = history.integration_order();
    mutable_this->step_values_.clear();
    for (const StepRecord<Vars>& record : history) {
      mutable_this->step_values_.push_back(make_untyped(record));
    }
    mutable_this->substep_values_.clear();
    for (const StepRecord<Vars>& record : history.substeps()) {
      mutable_this->substep_values_.push_back(make_untyped(record));
    }
  }

 private:
  const boost::container::static_vector<UntypedStepRecord<WrapperType>,
                                        history_max_substeps>&
  substep_values() const override;

  template <typename Vars>
  static UntypedStepRecord<WrapperType> make_untyped(
      const StepRecord<Vars>& record) {
    // This class only exposes these records as const references, so
    // it is OK if we break non-allocating references to the original
    // data by moving out of the MathWrappers since you can't modify
    // through them.
    return {record.time_step_id,
            record.value.has_value() ? std::optional{const_cast<WrapperType&&>(
                                           *make_math_wrapper(*record.value))}
                                     : std::nullopt,
            const_cast<WrapperType&&>(*make_math_wrapper(record.derivative))};
  }

  size_t integration_order_{};
  // static_vector never reallocates, so storing non-owning
  // DataVectors is safe.
  boost::container::static_vector<UntypedStepRecord<WrapperType>,
                                  history_max_past_steps + 2>
      step_values_{};
  boost::container::static_vector<UntypedStepRecord<WrapperType>,
                                  history_max_substeps>
      substep_values_{};
};
#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) and not defined(__clang__)
}  // namespace History_detail

/// \ingroup TimeSteppersGroup
/// The past-time data used by TimeStepper classes to update the
/// evolved variables.
///
/// This class exposes an STL-like container interface for accessing
/// the step (not substep) `StepRecord` data.  The records can be
/// freely modified through this interface, although modifying the
/// `time_step_id` field is generally inadvisable.
///
/// This class is designed to minimize the number of memory
/// allocations performed during a step, and so caches discarded
/// entries for reuse if they contain dynamic allocations.  During
/// steady-state operation, this class will perform no heap
/// allocations.  If `Vars` and the associated `DerivVars` do not
/// allocate internally, then this class will perform no heap
/// allocations under any circumstances.
template <typename Vars>
class History
    : public stl_boilerplate::RandomAccessSequence<History<Vars>,
                                                   StepRecord<Vars>, false> {
 public:
  using DerivVars = db::prefix_variables<::Tags::dt, Vars>;

  History() = default;
  explicit History(const size_t integration_order)
      : integration_order_(integration_order) {}
  History(const History& other);
  History(History&&) = default;
  History& operator=(const History& other);
  History& operator=(History&&) = default;

  /// The wrapped type presented by the type-erased history.  One of
  /// the types in \ref MATH_WRAPPER_TYPES.
  using UntypedVars = math_wrapper_type<Vars>;

  /// \cond
  // This wrapper around UntypedAccessCommon exists because we want
  // the special members of that to be protected, and we want this to
  // be final.
  class ConstUntypedAccess final : public History_detail::UntypedAccessCommon<
                                       ConstUntypedHistory<UntypedVars>> {
   public:
    ConstUntypedAccess() = delete;
    ConstUntypedAccess(const ConstUntypedAccess&) = delete;
    ConstUntypedAccess(ConstUntypedAccess&&) = default;
    ConstUntypedAccess& operator=(const ConstUntypedAccess&) = delete;
    ConstUntypedAccess& operator=(ConstUntypedAccess&&) = default;
    ~ConstUntypedAccess() = default;

   private:
    friend History;

    explicit ConstUntypedAccess(const History& history)
        : History_detail::UntypedAccessCommon<ConstUntypedHistory<UntypedVars>>(
              history) {}
  };

  class MutableUntypedAccess final : public History_detail::UntypedAccessCommon<
                                         MutableUntypedHistory<UntypedVars>> {
   public:
    MutableUntypedAccess() = delete;
    MutableUntypedAccess(const MutableUntypedAccess&) = delete;
    MutableUntypedAccess(MutableUntypedAccess&&) = default;
    MutableUntypedAccess& operator=(const MutableUntypedAccess&) = delete;
    MutableUntypedAccess& operator=(MutableUntypedAccess&&) = default;
    ~MutableUntypedAccess() = default;

    void discard_value(const TimeStepId& id_to_discard) const override {
      history_->discard_value(id_to_discard);
      this->reinitialize(*history_);
    }

    void pop_front() const override {
      history_->pop_front();
      this->reinitialize(*history_);
    }

    void clear_substeps() const override {
      history_->clear_substeps();
      this->reinitialize(*history_);
    }

   private:
    friend History;

    explicit MutableUntypedAccess(const gsl::not_null<History*> history)
        : History_detail::UntypedAccessCommon<
              MutableUntypedHistory<UntypedVars>>(*history),
          history_(history) {}

   private:
    gsl::not_null<History*> history_;
  };

  class ConstSubsteps : public stl_boilerplate::RandomAccessSequence<
                            ConstSubsteps, const StepRecord<Vars>, true> {
   public:
    ConstSubsteps() = delete;
    ConstSubsteps(const ConstSubsteps&) = delete;
    ConstSubsteps(ConstSubsteps&&) = default;
    ConstSubsteps& operator=(const ConstSubsteps&) = delete;
    ConstSubsteps& operator=(ConstSubsteps&&) = default;
    ~ConstSubsteps() = default;

    size_t size() const { return history_->substep_values_.size(); }
    static constexpr size_t max_size() { return history_max_substeps; }

    const StepRecord<Vars>& operator[](const size_t index) const {
      ASSERT(index < size(),
             "Requested substep " << index << " but only have " << size());
      return history_->substep_values_[index];
    }

   private:
    friend History;
    explicit ConstSubsteps(const History& history) : history_(&history) {}

    gsl::not_null<const History*> history_;
  };

  class MutableSubsteps
      : public stl_boilerplate::RandomAccessSequence<MutableSubsteps,
                                                     StepRecord<Vars>, true> {
   public:
    MutableSubsteps() = delete;
    MutableSubsteps(const MutableSubsteps&) = delete;
    MutableSubsteps(MutableSubsteps&&) = default;
    MutableSubsteps& operator=(const MutableSubsteps&) = delete;
    MutableSubsteps& operator=(MutableSubsteps&&) = default;
    ~MutableSubsteps() = default;

    size_t size() const { return history_->substep_values_.size(); }
    static constexpr size_t max_size() { return history_max_substeps; }

    StepRecord<Vars>& operator[](const size_t index) const {
      ASSERT(index < size(),
             "Requested substep " << index << " but only have " << size());
      return history_->substep_values_[index];
    }

   private:
    friend History;
    explicit MutableSubsteps(const gsl::not_null<History*> history)
        : history_(history) {}

    gsl::not_null<History*> history_;
  };
  /// \endcond

  /// Immutable, type-erased access to the history.  This method
  /// returns a class derived from `ConstUntypedHistory<UntypedVars>`.
  /// Any modifications to the History class invalidate the object
  /// returned by this function.
  ConstUntypedAccess untyped() const { return ConstUntypedAccess(*this); }

  /// Mutable, type-erased access to the history.  This method returns
  /// a class derived from `MutableUntypedHistory<UntypedVars>`.  Any
  /// modifications to the History class invalidate the object
  /// returned by this function, except for modifications performed
  /// through the object itself.
  MutableUntypedAccess untyped() { return MutableUntypedAccess(this); }

  /// Get or set the order the time stepper is running at.  Many time
  /// steppers expect this to have a particular value.  This has no
  /// effect on the storage of past data.
  /// @{
  size_t integration_order() const { return integration_order_; }
  void integration_order(const size_t new_integration_order) {
    integration_order_ = new_integration_order;
  }
  /// @}

  /// Type and value used to indicate that a record is to be created
  /// without the `value` field set.
  /// @{
  struct NoValue {};
  static constexpr NoValue no_value{};
  /// @}

  /// Insert a new entry into the history.  It will be inserted as a
  /// step or substep as appropriate.
  ///
  /// The supplied `time_step_id` must be later than the current
  /// latest entry, and if the substep of `time_step_id` is nonzero,
  /// the id must be consistent with the existing data for the current
  /// step.
  ///
  /// If the constant `History::no_value` is passed, the created
  /// record will not have its `value` field set.  This is useful for
  /// histories other than the history of the primary evolution
  /// integration, such as the implicit portion of an IMEX evolution.
  /// These other uses adjust the result of the main history, and so
  /// do not require values (which are only used for the zeroth-order
  /// terms).
  ///
  /// The `value` and `derivative` data will be copied into cached
  /// allocations, if any are available.  That is, only a copy, not a
  /// memory allocation, will be performed when possible.
  /// @{
  void insert(const TimeStepId& time_step_id, const Vars& value,
              const DerivVars& derivative);
  void insert(const TimeStepId& time_step_id, NoValue /*unused*/,
              const DerivVars& derivative);
  /// @}

  /// Insert a new entry in the history by modifying the fields of the
  /// record directly, instead of passing a value to copy.
  ///
  /// The (optional) \p value_inserter must be a functor callable with
  /// single argument of type `gsl::not_null<Vars*>` and the \p
  /// derivative_inserter must be callable with a single argument of
  /// type `gsl::not_null<DerivVars*>`.  The passed objects will be
  /// created from a cached memory allocation if one is available, and
  /// will otherwise be default-constructed.
  ///
  /// All the restrictions on valid values \p time_step_id for the
  /// `insert` method apply here as well.
  /// @{
  template <typename ValueFunc, typename DerivativeFunc>
  void insert_in_place(const TimeStepId& time_step_id,
                       ValueFunc&& value_inserter,
                       DerivativeFunc&& derivative_inserter);
  template <typename DerivativeFunc>
  void insert_in_place(const TimeStepId& time_step_id, NoValue /*unused*/,
                       DerivativeFunc&& derivative_inserter);
  /// @}

  /// Insert data at the start of the history, similar to `push_front`
  /// on some STL containers.  This can be useful when initializing a
  /// multistep integrator.
  ///
  /// This should only be used for initialization, and cannot be used
  /// for substep data.  The supplied `time_step_id` must be earlier
  /// than the current first entry.
  /// @{
  void insert_initial(TimeStepId time_step_id, Vars value,
                      DerivVars derivative);
  void insert_initial(TimeStepId time_step_id, NoValue /*unused*/,
                      DerivVars derivative);
  /// @}

  /// The number of stored step (not substep) entries.
  size_t size() const { return step_values_.size(); }
  /// The maximum number of step (not substep) entries that can be
  /// stored in the history.  This number is not very large, but will
  /// be sufficient for running a time stepper requiring
  /// `history_max_past_steps` past step values.
  static constexpr size_t max_size() {
    // In addition to the past steps, we must store the start and end
    // of the current step.
    return history_max_past_steps + 2;
  }

  /// Access the `StepRecord` for a step (not substep).
  /// @{
  const StepRecord<Vars>& operator[](const size_t index) const {
    return step_values_[index];
  }
  StepRecord<Vars>& operator[](const size_t index) {
    return step_values_[index];
  }
  /// @}

  /// Access the `StepRecord` for a step or substep with a given
  /// TimeStepId.  It is an error if there is no entry with that id.
  /// @{
  const StepRecord<Vars>& operator[](const TimeStepId& id) const {
    return History_detail::find_record(*this, id);
  }
  StepRecord<Vars>& operator[](const TimeStepId& id) {
    return History_detail::find_record(*this, id);
  }
  /// @}

  /// Get the value at the latest step or substep in the history, even
  /// if the value in that record has been discarded.  This is not
  /// available if `undo_latest` has been called since the last
  /// insertion.
  ///
  /// This function exists to allow access to the previous value after
  /// a potentially bad step has been taken without restricting how
  /// the time steppers can manage their history.
  const Vars& latest_value() const;

  /// Check whether we are at the start of a step, i.e, the most
  /// recent entry in the history is not a substep.
  bool at_step_start() const;

  /// Container view of the `StepRecord`s for the history substeps.
  /// These methods return classes providing an STL-like container
  /// interface to the substep data.  These containers are
  /// zero-indexed, so `substeps()[0]` will have a substep value of 1.
  ///
  /// These containers do not have methods to modify their sizes.  Use
  /// the methods on the `History` class for those operations.
  /// @{
  ConstSubsteps substeps() const { return ConstSubsteps(*this); }
  MutableSubsteps substeps() { return MutableSubsteps(this); }
  /// @}

  /// Clear the `value` in the indicated record.  It is an error if
  /// there is no record with the passed TimeStepId.  Any memory
  /// allocations will be cached for future reuse.
  void discard_value(const TimeStepId& id_to_discard);

  /// Drop the oldest step (not substep) entry in the history.  Any
  /// memory allocations will be cached for future reuse.
  void pop_front();

  /// Drop the newest step or substep entry in the history.  Any
  /// memory allocations will be cached for future reuse.
  void undo_latest();

  /// Remove all substep entries from the history.  Any memory
  /// allocations will be cached for future reuse.
  void clear_substeps();

  /// Remove all (step and substep) entries from the history.  Any
  /// memory allocations will be cached for future reuse.
  void clear();

  /// Release any cached memory allocations.
  void shrink_to_fit();

  /// Apply \p func to `make_not_null(&e)` for `e` every `derivative`
  /// and valid `*value` in records held by the history.
  template <typename F>
  void map_entries(F&& func);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  void discard_value(gsl::not_null<std::optional<Vars>*> value);
  void cache_allocations(gsl::not_null<StepRecord<Vars>*> record);

  template <typename ValueFunc, typename DerivativeFunc>
  void insert_impl(const TimeStepId& time_step_id, ValueFunc&& value_inserter,
                   DerivativeFunc&& derivative_inserter);

  template <typename InsertedVars>
  void insert_initial_impl(TimeStepId time_step_id, InsertedVars value,
                           DerivVars derivative);

  size_t integration_order_{0};

  StaticDeque<StepRecord<Vars>, max_size()> step_values_{};
  boost::container::static_vector<StepRecord<Vars>, history_max_substeps>
      substep_values_{};

  // If the algorithm undoes a substep, it may want to reset the
  // variables to the previous value.  We don't want the time stepper
  // to have to keep track of this when discarding values, so we hang
  // onto the last value if it gets discarded.
  std::optional<Vars> latest_value_if_discarded_{};

  // Memory allocations available for reuse.
  boost::container::static_vector<Vars, max_size() + history_max_substeps>
      vars_allocation_cache_{};
  boost::container::static_vector<DerivVars, max_size() + history_max_substeps>
      deriv_vars_allocation_cache_{};
};

// Don't copy the allocation caches.
template <typename Vars>
History<Vars>::History(const History& other)
    : integration_order_(other.integration_order_),
      step_values_(other.step_values_),
      substep_values_(other.substep_values_),
      latest_value_if_discarded_(other.latest_value_if_discarded_) {}

// Don't copy the allocation caches.
template <typename Vars>
History<Vars>& History<Vars>::operator=(const History& other) {
  integration_order_ = other.integration_order_;
  step_values_ = other.step_values_;
  substep_values_ = other.substep_values_;
  latest_value_if_discarded_ = other.latest_value_if_discarded_;
  return *this;
}

template <typename Vars>
void History<Vars>::insert(const TimeStepId& time_step_id, const Vars& value,
                           const DerivVars& derivative) {
  insert_impl(
      time_step_id,
      [&](const gsl::not_null<Vars*> record_value) { *record_value = value; },
      [&](const gsl::not_null<DerivVars*> record_derivative) {
        *record_derivative = derivative;
      });
}

template <typename Vars>
void History<Vars>::insert(const TimeStepId& time_step_id, NoValue /*unused*/,
                           const DerivVars& derivative) {
  insert_impl(time_step_id, NoValue{},
              [&](const gsl::not_null<DerivVars*> record_derivative) {
                *record_derivative = derivative;
              });
}

template <typename Vars>
template <typename ValueFunc, typename DerivativeFunc>
void History<Vars>::insert_in_place(const TimeStepId& time_step_id,
                                    ValueFunc&& value_inserter,
                                    DerivativeFunc&& derivative_inserter) {
  insert_impl(time_step_id, std::forward<ValueFunc>(value_inserter),
              std::forward<DerivativeFunc>(derivative_inserter));
}

template <typename Vars>
template <typename DerivativeFunc>
void History<Vars>::insert_in_place(const TimeStepId& time_step_id,
                                    NoValue /*unused*/,
                                    DerivativeFunc&& derivative_inserter) {
  insert_impl(time_step_id, NoValue{},
              std::forward<DerivativeFunc>(derivative_inserter));
}

template <typename Vars>
void History<Vars>::insert_initial(TimeStepId time_step_id, Vars value,
                                   DerivVars derivative) {
  insert_initial_impl(std::move(time_step_id), std::move(value),
                      std::move(derivative));
}

template <typename Vars>
void History<Vars>::insert_initial(TimeStepId time_step_id, NoValue /*unused*/,
                                   DerivVars derivative) {
  insert_initial_impl(std::move(time_step_id), NoValue{},
                      std::move(derivative));
}

template <typename Vars>
const Vars& History<Vars>::latest_value() const {
  const auto& latest_record =
      at_step_start() ? this->back() : substeps().back();
  if (latest_record.value.has_value()) {
    return *latest_record.value;
  } else {
    ASSERT(latest_value_if_discarded_.has_value(),
           "Latest value unavailable.  The latest insertion was undone.");
    return *latest_value_if_discarded_;
  }
}

template <typename Vars>
bool History<Vars>::at_step_start() const {
  ASSERT(not this->empty(), "History is empty");
  return substep_values_.empty() or
         substep_values_.back().time_step_id < this->back().time_step_id;
}

template <typename Vars>
void History<Vars>::discard_value(const TimeStepId& id_to_discard) {
  auto& latest_record = at_step_start() ? this->back() : substeps().back();
  if (latest_record.time_step_id == id_to_discard) {
    discard_value(&latest_value_if_discarded_);
    latest_value_if_discarded_ = std::move(latest_record.value);
    latest_record.value.reset();
  } else {
    discard_value(&History_detail::find_record(*this, id_to_discard).value);
  }
}

template <typename Vars>
void History<Vars>::pop_front() {
  ASSERT(not this->empty(), "History is empty");
  ASSERT(substeps().empty() or substeps().front().time_step_id.step_time() !=
                                   this->front().time_step_id.step_time(),
         "Cannot remove a step with substeps.  Call clear_substeps() first.");
  cache_allocations(&step_values_.front());
  step_values_.pop_front();
}

template <typename Vars>
void History<Vars>::undo_latest() {
  if (at_step_start()) {
    cache_allocations(&step_values_.back());
    step_values_.pop_back();
  } else {
    cache_allocations(&substep_values_.back());
    substep_values_.pop_back();
  }
  discard_value(&latest_value_if_discarded_);
}

template <typename Vars>
void History<Vars>::clear_substeps() {
  for (auto& record : substep_values_) {
    cache_allocations(&record);
  }
  substep_values_.clear();
}

template <typename Vars>
void History<Vars>::clear() {
  clear_substeps();
  while (not this->empty()) {
    pop_front();
  }
  discard_value(&latest_value_if_discarded_);
}

template <typename Vars>
void History<Vars>::shrink_to_fit() {
  vars_allocation_cache_.clear();
  vars_allocation_cache_.shrink_to_fit();
  deriv_vars_allocation_cache_.clear();
  deriv_vars_allocation_cache_.shrink_to_fit();
}

template <typename Vars>
template <typename F>
void History<Vars>::map_entries(F&& func) {
  for (auto& record : *this) {
    func(make_not_null(&record.derivative));
    if (record.value.has_value()) {
      func(make_not_null(&*record.value));
    }
  }
  for (auto& record : this->substeps()) {
    func(make_not_null(&record.derivative));
    if (record.value.has_value()) {
      func(make_not_null(&*record.value));
    }
  }
}

template <typename Vars>
void History<Vars>::pup(PUP::er& p) {
  p | integration_order_;
  p | step_values_;

  size_t substep_size = substep_values_.size();
  p | substep_size;
  substep_values_.resize(substep_size);
  for (auto& record : substep_values_) {
    p | record;
  }

  // Don't serialize the allocation cache.
}

// Doxygen is confused by this function for some reason.
/// \cond
template <typename Vars>
void History<Vars>::discard_value(
    const gsl::not_null<std::optional<Vars>*> value) {
  if (not value->has_value()) {
    return;
  }
  // If caching doesn't save anything, don't allocate memory for the cache.
  if (contains_allocations(**value)) {
    vars_allocation_cache_.emplace_back(std::move(**value));
  }
  value->reset();
}
/// \endcond

template <typename Vars>
void History<Vars>::cache_allocations(
    const gsl::not_null<StepRecord<Vars>*> record) {
  discard_value(&record->value);
  // If caching doesn't save anything, don't allocate memory for the cache.
  if (contains_allocations(record->derivative)) {
    deriv_vars_allocation_cache_.emplace_back(std::move(record->derivative));
  }
}

template <typename Vars>
template <typename ValueFunc, typename DerivativeFunc>
void History<Vars>::insert_impl(const TimeStepId& time_step_id,
                                ValueFunc&& value_inserter,
                                DerivativeFunc&& derivative_inserter) {
  ASSERT(this->empty() or time_step_id > this->back().time_step_id,
         "New entry at " << time_step_id
         << " must be later than previous entry at "
         << this->back().time_step_id);
  discard_value(&latest_value_if_discarded_);
  StepRecord<Vars> record{};
  record.time_step_id = time_step_id;
  if constexpr (not std::is_same_v<ValueFunc, NoValue>) {
    if (vars_allocation_cache_.empty()) {
      record.value.emplace();
    } else {
      record.value.emplace(std::move(vars_allocation_cache_.back()));
      vars_allocation_cache_.pop_back();
    }
    std::forward<ValueFunc>(value_inserter)(make_not_null(&*record.value));
  }
  if (not deriv_vars_allocation_cache_.empty()) {
    record.derivative = std::move(deriv_vars_allocation_cache_.back());
    deriv_vars_allocation_cache_.pop_back();
  }
  std::forward<DerivativeFunc>(derivative_inserter)(
      make_not_null(&record.derivative));

  const size_t substep = time_step_id.substep();
  if (substep == 0) {
    step_values_.push_back(std::move(record));
  } else {
    ASSERT(not this->empty(), "Cannot insert substep into empty history.");
    ASSERT(time_step_id.step_time() == this->back().time_step_id.step_time(),
           "Cannot insert substep " << time_step_id << " of different step "
           << this->back().time_step_id);
    ASSERT(substep == substeps().size() + 1,
           "Cannot insert substep " << substep << " following "
           << substeps().size());
    ASSERT(substep_values_.size() < substep_values_.max_size(),
           "Cannot insert new substep because the History is full.");
    substep_values_.push_back(std::move(record));
  }
}

template <typename Vars>
template <typename InsertedVars>
void History<Vars>::insert_initial_impl(TimeStepId time_step_id,
                                        InsertedVars value,
                                        DerivVars derivative) {
  ASSERT(vars_allocation_cache_.empty(),
         "insert_initial should only be used for initialization");
  ASSERT(deriv_vars_allocation_cache_.empty(),
         "insert_initial should only be used for initialization");
  ASSERT(time_step_id.substep() == 0, "Cannot use insert_initial for substeps");
  ASSERT(this->empty() or time_step_id < this->front().time_step_id,
         "New initial entry at " << time_step_id
         << " must be earlier than previous entry at "
         << this->front().time_step_id);

  if constexpr (std::is_same_v<InsertedVars, Vars>) {
    step_values_.push_front(StepRecord<Vars>{
        std::move(time_step_id), std::move(value), std::move(derivative)});
  } else {
    static_assert(std::is_same_v<InsertedVars, NoValue>);
    (void)value;
    step_values_.push_front(StepRecord<Vars>{
        std::move(time_step_id), std::nullopt, std::move(derivative)});
  }
}

template <typename Vars>
bool operator==(const History<Vars>& a, const History<Vars>& b) {
  return a.integration_order() == b.integration_order() and
         a.size() == b.size() and std::equal(a.begin(), a.end(), b.begin()) and
         a.substeps() == b.substeps();
}

template <typename Vars>
bool operator!=(const History<Vars>& a, const History<Vars>& b) {
  return not(a == b);
}
}  // namespace TimeSteppers
