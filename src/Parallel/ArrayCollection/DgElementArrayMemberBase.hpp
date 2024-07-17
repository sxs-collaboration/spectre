// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <cstddef>
#include <limits>
#include <pup.h>
#include <string>
#include <unordered_map>

#include "Domain/Structure/ElementId.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

namespace Parallel {
/*!
 * \brief The base class of a member of an DG element array/map on a nodegroup.
 *
 * The nodegroup DgElementCollection stores all the elements on a node. Each
 * of those elements is a `DgElementArrayMember` and has this base class to
 * make access easier so it can be used in a type-erased context since
 * `DgElementArrayMember` depends on the metavariables,
 * phase-dependent-action-list, and the simple tags needed from options.
 *
 * This class essentially mimicks a lot of the functionality of
 * `Parallel::DistributedObject` but does not involve Charm++ beyond
 * serialization.
 */
template <size_t Dim>
class DgElementArrayMemberBase : PUP::able {
 public:
  DgElementArrayMemberBase() = default;

  DgElementArrayMemberBase(const DgElementArrayMemberBase& /*rhs*/) = default;
  DgElementArrayMemberBase& operator=(const DgElementArrayMemberBase& /*rhs*/) =
      default;
  DgElementArrayMemberBase(DgElementArrayMemberBase&& /*rhs*/) = default;
  DgElementArrayMemberBase& operator=(DgElementArrayMemberBase&& /*rhs*/) =
      default;
  ~DgElementArrayMemberBase() override = default;

  WRAPPED_PUPable_abstract(DgElementArrayMemberBase);  // NOLINT

  explicit DgElementArrayMemberBase(CkMigrateMessage* msg);

  /// Start execution of the phase-dependent action list in `next_phase`. If
  /// `next_phase` has already been visited, execution will resume at the point
  /// where the previous execution of the same phase left off.
  virtual void start_phase(Parallel::Phase next_phase) = 0;

  /// Get the current phase
  Parallel::Phase phase() const;

  /// Tell the Algorithm it should no longer execute the algorithm. This does
  /// not mean that the execution of the program is terminated, but only that
  /// the algorithm has terminated. An algorithm can be restarted by passing
  /// `true` as the second argument to the `receive_data` method or by calling
  /// perform_algorithm(true).
  void set_terminate(gsl::not_null<size_t*> number_of_elements_terminated,
                     gsl::not_null<Parallel::NodeLock*> nodegroup_lock,
                     bool terminate);

  /// Check if an algorithm should continue being evaluated
  bool get_terminate() const;

  /// The zero-indexed step in the algorithm.
  size_t algorithm_step() const;

  /// Start evaluating the algorithm until it is stopped by an action.
  virtual void perform_algorithm() = 0;

  /// Print the expanded type aliases
  virtual std::string print_types() const = 0;

  /// Print the current state of the algorithm
  std::string print_state() const;

  /// Print the current contents of the inboxes
  virtual std::string print_inbox() const = 0;

  /// Print the current contents of the DataBox
  virtual std::string print_databox() const = 0;

  /// \brief The `inbox_lock()` only locks the inbox, nothing else. The inbox is
  /// unsafe to access without this lock.
  ///
  /// Use `element_lock()` to lock the rest of the element.
  ///
  /// This should always be managed by `std::unique_lock` or `std::lock_guard`.
  Parallel::NodeLock& inbox_lock();

  /// \brief Locks the element, except for the inbox, which is guarded by the
  /// `inbox_lock()`.
  ///
  /// This should always be managed by `std::unique_lock` or `std::lock_guard`.
  Parallel::NodeLock& element_lock();

  /// \brief Set which core this element should pretend to be bound to.
  void set_core(size_t core);

  /// \brief Get which core this element should pretend to be bound to.
  size_t get_core() const;

  /// Returns the name of the last "next iterable action" to be run before a
  /// deadlock occurred.
  const std::string& deadlock_analysis_next_iterable_action() const {
    return deadlock_analysis_next_iterable_action_;
  }

  void pup(PUP::er& p) override;

 protected:
  DgElementArrayMemberBase(ElementId<Dim> element_id, size_t node_number);

  Parallel::NodeLock inbox_lock_{};
  Parallel::NodeLock element_lock_{};
  bool performing_action_ = false;
  Parallel::Phase phase_{Parallel::Phase::Initialization};
  std::unordered_map<Parallel::Phase, size_t> phase_bookmarks_{};
  std::size_t algorithm_step_ = 0;

  bool terminate_{true};
  bool halt_algorithm_until_next_phase_{false};

  // Records the name of the next action to be called so that during deadlock
  // analysis we can print this out.
  std::string deadlock_analysis_next_iterable_action_{};
  ElementId<Dim> element_id_;
  size_t my_node_{std::numeric_limits<size_t>::max()};
  // There is no associated core. However, we use this as a method of
  // interoperating with core-aware concepts like the interpolation
  // framework. Once that framework is core-agnostic we will remove my_core_.
  size_t my_core_{std::numeric_limits<size_t>::max()};
};
}  // namespace Parallel
