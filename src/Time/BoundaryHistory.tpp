// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Time/BoundaryHistory.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <tuple>
#include <utility>

#include "DataStructures/CircularDeque.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Serialization/PupBoost.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace TimeSteppers {
namespace BoundaryHistory_detail {
template <typename Data>
void StepData<Data>::Entry::pup(PUP::er& p) {
  p | id;
  p | data;
}

template <typename Data>
void StepData<Data>::pup(PUP::er& p) {
  p | integration_order;
  p | substeps;
}

template <typename Data>
bool operator<(const StepData<Data>& a, const StepData<Data>& b) {
  return a.substeps.front().id < b.substeps.front().id;
}

template <typename Data>
bool operator<(const TimeStepId& a, const StepData<Data>& b) {
  return a < b.substeps.front().id;
}

template <typename Data>
bool operator<(const StepData<Data>& a, const TimeStepId& b) {
  return a.substeps.front().id < b;
}
}  // namespace BoundaryHistory_detail

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local, bool Mutable>
size_t BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    SideAccessCommon<Local, Mutable>::size() const {
  return parent_data().size();
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local, bool Mutable>
const TimeStepId&
BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::SideAccessCommon<
    Local, Mutable>::operator[](const size_t n) const {
  return (*this)[{n, 0}];
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local, bool Mutable>
const TimeStepId&
BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::SideAccessCommon<
    Local, Mutable>::operator[](const std::pair<size_t, size_t>&
                                    step_and_substep) const {
  return entry(step_and_substep).id;
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local, bool Mutable>
size_t BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    SideAccessCommon<Local, Mutable>::integration_order(const size_t n) const {
  return parent_data()[n].integration_order;
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local, bool Mutable>
size_t
BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::SideAccessCommon<
    Local, Mutable>::integration_order(const TimeStepId& id) const {
  return step_data(id).integration_order;
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local, bool Mutable>
size_t BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    SideAccessCommon<Local, Mutable>::number_of_substeps(const size_t n) const {
  return parent_data()[n].substeps.size();
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local, bool Mutable>
size_t
BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::SideAccessCommon<
    Local, Mutable>::number_of_substeps(const TimeStepId& id) const {
  return step_data(id).substeps.size();
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local, bool Mutable>
auto BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    SideAccessCommon<Local, Mutable>::data(const size_t n) const -> Data& {
  return parent_data()[n].substeps.front().data;
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local, bool Mutable>
auto BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    SideAccessCommon<Local, Mutable>::data(const TimeStepId& id) const
    -> Data& {
  return entry(id).data;
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local, bool Mutable>
auto BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    SideAccessCommon<Local, Mutable>::parent_data() const
    -> tmpl::conditional_t<
        Mutable, CircularDeque<BoundaryHistory_detail::StepData<MutableData>>&,
        const CircularDeque<BoundaryHistory_detail::StepData<MutableData>>&> {
  if constexpr (Local) {
    return parent_->local_data_;
  } else {
    return parent_->remote_data_;
  }
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local, bool Mutable>
auto& BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    SideAccessCommon<Local, Mutable>::step_data(const TimeStepId& id) const {
  auto entry = std::upper_bound(parent_data().begin(), parent_data().end(), id);
  ASSERT(entry != parent_data().begin(), "Id " << id << " not present.");
  --entry;
  ASSERT(id.substep() < entry->substeps.size() and
             entry->substeps[id.substep()].id == id,
         "Id " << id << " not present.");
  return *entry;
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local, bool Mutable>
auto& BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    SideAccessCommon<Local, Mutable>::entry(const TimeStepId& id) const {
  // Bounds and consistency are checked in step_data()
  return step_data(id).substeps[id.substep()];
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local, bool Mutable>
auto& BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    SideAccessCommon<Local, Mutable>::entry(
        const std::pair<size_t, size_t>& step_and_substep) const {
  ASSERT(step_and_substep.first < parent_data().size(),
         "Step out of range: " << step_and_substep.first);
  auto& substeps = parent_data()[step_and_substep.first].substeps;
  ASSERT(step_and_substep.second < substeps.size(),
         "Substep out of range: " << step_and_substep.second);
  return substeps[step_and_substep.second];
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local, bool Mutable>
BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::SideAccessCommon<
    Local, Mutable>::SideAccessCommon(const gsl::not_null<StoredHistory*>
                                          parent)
    : parent_(parent) {}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local>
BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    MutableSideAccess<Local>::MutableSideAccess(
        const gsl::not_null<BoundaryHistory*> parent)
    : SideAccessCommon<Local, true>(parent) {}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local>
BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::ConstSideAccess<
    Local>::ConstSideAccess(const gsl::not_null<const BoundaryHistory*> parent)
    : SideAccessCommon<Local, false>(parent) {}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
auto BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::local()
    -> MutableSideAccess<true> {
  return MutableSideAccess<true>(this);
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
auto BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::local()
    const -> ConstSideAccess<true> {
  return ConstSideAccess<true>(this);
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
auto BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::remote()
    -> MutableSideAccess<false> {
  return MutableSideAccess<false>(this);
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
auto BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::remote()
    const -> ConstSideAccess<false> {
  return ConstSideAccess<false>(this);
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local>
void BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    MutableSideAccess<Local>::pop_front() const {
  if constexpr (Local) {
    this->parent_->pop_local();
  } else {
    this->parent_->pop_remote();
  }
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local>
void BoundaryHistory<LocalData, RemoteData,
                     UntypedCouplingResult>::MutableSideAccess<Local>::clear()
    const {
  while (not this->empty()) {
    pop_front();
  }
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local>
void BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    MutableSideAccess<Local>::clear_substeps(const size_t n) const {
  if constexpr (Local) {
    this->parent_->clear_substeps_local(n);
  } else {
    this->parent_->clear_substeps_remote(n);
  }
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local>
void BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    MutableSideAccess<Local>::insert(const TimeStepId& id,
                                     const size_t integration_order,
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

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool Local>
void BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    MutableSideAccess<Local>::insert_initial(const TimeStepId& id,
                                             const size_t integration_order,
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

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
void BoundaryHistory<LocalData, RemoteData,
                     UntypedCouplingResult>::clear_coupling_cache() {
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

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
void BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::pup(
    PUP::er& p) {
  p | local_data_;
  p | remote_data_;
  p | couplings_;
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
template <bool IncludeData>
std::ostream&
BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::print(
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

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
void BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    insert_local(const TimeStepId& id, const size_t integration_order,
                 LocalData data) {
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

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
void BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    insert_remote(const TimeStepId& id, const size_t integration_order,
                  RemoteData data) {
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

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
void BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
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

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
void BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    insert_initial_remote(const TimeStepId& id, const size_t integration_order,
                          RemoteData data) {
  remote_data_.push_front({integration_order, {}});
  remote_data_.front().substeps.push_back({id, std::move(data)});
  couplings_.emplace_front(1_st);
  for (const auto& local_step : local_data_) {
    couplings_.front().back().emplace_back(local_step.substeps.size());
  }
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
void BoundaryHistory<LocalData, RemoteData,
                     UntypedCouplingResult>::pop_local() {
  local_data_.pop_front();
  for (auto& remote_step : couplings_) {
    for (auto& remote_substep : remote_step) {
      remote_substep.pop_front();
    }
  }
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
void BoundaryHistory<LocalData, RemoteData,
                     UntypedCouplingResult>::pop_remote() {
  remote_data_.pop_front();
  couplings_.pop_front();
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
void BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    clear_substeps_local(const size_t n) {
  local_data_[n].substeps.erase(local_data_[n].substeps.begin() + 1,
                                local_data_[n].substeps.end());
  for (auto& remote_step : couplings_) {
    for (auto& remote_substep : remote_step) {
      auto& local_step = remote_substep[n];
      local_step.erase(local_step.begin() + 1, local_step.end());
    }
  }
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
void BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::
    clear_substeps_remote(const size_t n) {
  remote_data_[n].substeps.erase(remote_data_[n].substeps.begin() + 1,
                                 remote_data_[n].substeps.end());
  auto& remote_step = couplings_[n];
  remote_step.erase(remote_step.begin() + 1, remote_step.end());
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
std::tuple<std::optional<UntypedCouplingResult>&, const LocalData&,
           const RemoteData&>
BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>::find_cache_entry(
    const TimeStepId& local_id, const TimeStepId& remote_id) const {
  auto local_entry =
      std::upper_bound(local_data_.begin(), local_data_.end(), local_id);
  ASSERT(local_entry != local_data_.begin(), "local_id not present");
  --local_entry;
  ASSERT(local_id.substep() < local_entry->substeps.size() and
             local_entry->substeps[local_id.substep()].id == local_id,
         "local_id not present");
  const auto local_step_offset =
      static_cast<size_t>(local_entry - local_data_.begin());

  auto remote_entry =
      std::upper_bound(remote_data_.begin(), remote_data_.end(), remote_id);
  ASSERT(remote_entry != remote_data_.begin(), "remote_id not present");
  --remote_entry;
  ASSERT(remote_id.substep() < remote_entry->substeps.size() and
             remote_entry->substeps[remote_id.substep()].id == remote_id,
         "remote_id not present");
  const auto remote_step_offset =
      static_cast<size_t>(remote_entry - remote_data_.begin());

  return {couplings_[remote_step_offset][remote_id.substep()][local_step_offset]
                    [local_id.substep()],
          local_entry->substeps[local_id.substep()].data,
          remote_entry->substeps[remote_id.substep()].data};
}

template <typename LocalData, typename RemoteData,
          typename UntypedCouplingResult>
std::ostream& operator<<(
    std::ostream& os,
    const BoundaryHistory<LocalData, RemoteData, UntypedCouplingResult>&
        history) {
  return history.template print<true>(os);
}
}  // namespace TimeSteppers

#define INSTANTIATE_BOUNDARY_HISTORY(...)                                      \
  template class TimeSteppers::BoundaryHistory<__VA_ARGS__>;                   \
  template class TimeSteppers::BoundaryHistory<__VA_ARGS__>::ConstSideAccess<  \
      false>;                                                                  \
  template class TimeSteppers::BoundaryHistory<__VA_ARGS__>::ConstSideAccess<  \
      true>;                                                                   \
  template class TimeSteppers::BoundaryHistory<                                \
      __VA_ARGS__>::MutableSideAccess<false>;                                  \
  template class TimeSteppers::BoundaryHistory<                                \
      __VA_ARGS__>::MutableSideAccess<true>;                                   \
  template class TimeSteppers::BoundaryHistory<__VA_ARGS__>::SideAccessCommon< \
      false, false>;                                                           \
  template class TimeSteppers::BoundaryHistory<__VA_ARGS__>::SideAccessCommon< \
      false, true>;                                                            \
  template class TimeSteppers::BoundaryHistory<__VA_ARGS__>::SideAccessCommon< \
      true, false>;                                                            \
  template class TimeSteppers::BoundaryHistory<__VA_ARGS__>::SideAccessCommon< \
      true, true>;                                                             \
  template std::ostream&                                                       \
  TimeSteppers::BoundaryHistory<__VA_ARGS__>::print<false>(                    \
      std::ostream & os, const size_t padding_size) const;                     \
  template std::ostream&                                                       \
  TimeSteppers::BoundaryHistory<__VA_ARGS__>::print<true>(                     \
      std::ostream & os, const size_t padding_size) const;
