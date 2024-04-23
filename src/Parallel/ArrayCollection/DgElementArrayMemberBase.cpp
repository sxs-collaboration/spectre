// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/ArrayCollection/DgElementArrayMemberBase.hpp"

#include <cstddef>
#include <pup.h>
#include <sstream>
#include <string>

#include "Domain/Structure/ElementId.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

namespace Parallel {
template <size_t Dim>
DgElementArrayMemberBase<Dim>::DgElementArrayMemberBase(
    ElementId<Dim> element_id, size_t node_number)
    : element_id_(element_id), my_node_(node_number) {}

template <size_t Dim>
DgElementArrayMemberBase<Dim>::DgElementArrayMemberBase(CkMigrateMessage* msg)
    : PUP::able(msg) {}

template <size_t Dim>
Phase DgElementArrayMemberBase<Dim>::phase() const {
  return phase_;
}

template <size_t Dim>
void DgElementArrayMemberBase<Dim>::set_terminate(
    const gsl::not_null<size_t*> number_of_elements_terminated,
    const gsl::not_null<Parallel::NodeLock*> nodegroup_lock,
    const bool terminate) {
  ASSERT(nodegroup_lock->try_lock() == false,
         "The nodegroup lock must be locked in order to call set_terminate on "
         "the DgElementArrayMember");
  if (not terminate_ and terminate) {
    ++(*number_of_elements_terminated);
  } else if (terminate_ and not terminate) {
    if ((*number_of_elements_terminated) > 0) {
      --(*number_of_elements_terminated);
    }
  } else {
    ASSERT(terminate_ == terminate,
           "The DG element with id "
               << element_id_ << " currently has termination status "
               << terminate_ << " and is being set to " << terminate
               << ". This is an internal inconsistency problem.");
  }
  terminate_ = terminate;
}

template <size_t Dim>
bool DgElementArrayMemberBase<Dim>::get_terminate() const {
  return terminate_;
}

template <size_t Dim>
size_t DgElementArrayMemberBase<Dim>::algorithm_step() const {
  return algorithm_step_;
}

template <size_t Dim>
std::string DgElementArrayMemberBase<Dim>::print_state() const {
  using ::operator<<;
  std::ostringstream os;
  os << "State:\n";
  os << "performing_action_ = " << std::boolalpha << performing_action_
     << ";\n";
  os << "phase_ = " << phase_ << ";\n";
  os << "phase_bookmarks_ = " << phase_bookmarks_ << ";\n";
  os << "algorithm_step_ = " << algorithm_step_ << ";\n";
  os << "terminate_ = " << terminate_ << ";\n";
  os << "halt_algorithm_until_next_phase_ = "
     << halt_algorithm_until_next_phase_ << ";\n";
  os << "array_index_ = " << element_id_ << ";\n";
  return os.str();
}

template <size_t Dim>
Parallel::NodeLock& DgElementArrayMemberBase<Dim>::inbox_lock() {
  return inbox_lock_;
}

template <size_t Dim>
Parallel::NodeLock& DgElementArrayMemberBase<Dim>::element_lock() {
  return element_lock_;
}

template <size_t Dim>
void DgElementArrayMemberBase<Dim>::set_core(size_t core) {
  my_core_ = core;
}

template <size_t Dim>
size_t DgElementArrayMemberBase<Dim>::get_core() const {
  return my_core_;
}

template <size_t Dim>
void DgElementArrayMemberBase<Dim>::pup(PUP::er& p) {
  PUP::able::pup(p);
  p | performing_action_;
  p | phase_;
  p | phase_bookmarks_;
  p | algorithm_step_;
  p | terminate_;
  p | halt_algorithm_until_next_phase_;
  p | deadlock_analysis_next_iterable_action_;
  p | element_id_;
  if (p.isUnpacking()) {
    // Node locks are default-constructed, which is fine
    //
    // Note: my_node_ is set by derived class pup
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class DgElementArrayMemberBase<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace Parallel
