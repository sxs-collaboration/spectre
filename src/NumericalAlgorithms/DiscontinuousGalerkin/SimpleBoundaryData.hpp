// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/none.hpp>
#include <boost/optional.hpp>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep
#include <utility>

#include "ErrorHandling/Assert.hpp"
#include "Time/Time.hpp"
#include "Utilities/BoostHelpers.hpp"  // IWYU pragma: keep

namespace dg {

/// \ingroup DiscontinuousGalerkinGroup
/// \brief Storage of boundary data on two sides of a mortar
///
/// Typically, values are inserted into this container by the flux
/// communication actions.
template <typename LocalVars, typename RemoteVars>
class SimpleBoundaryData {
 public:
  /// Add a value.  This function must be called once between calls to
  /// extract.
  //@{
  void local_insert(Time time, LocalVars vars) noexcept;
  void remote_insert(Time time, RemoteVars vars) noexcept;
  //@}

  /// Return the inserted data and reset the state to empty.
  std::pair<LocalVars, RemoteVars> extract() noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  Time time_;
  boost::optional<LocalVars> local_data_;
  boost::optional<RemoteVars> remote_data_;
};

template <typename LocalVars, typename RemoteVars>
void SimpleBoundaryData<LocalVars, RemoteVars>::local_insert(
    Time time, LocalVars vars) noexcept {
  ASSERT(not local_data_, "Already received local data.");
  ASSERT(not remote_data_ or time == time_,
         "Received local data at time " << time
         << " but already have remote data at time " << time_);
  time_ = time;
  local_data_ = std::move(vars);
}

template <typename LocalVars, typename RemoteVars>
void SimpleBoundaryData<LocalVars, RemoteVars>::remote_insert(
    Time time, RemoteVars vars) noexcept {
  ASSERT(not remote_data_, "Already received remote data.");
  ASSERT(not local_data_ or time == time_,
         "Received remote data at time " << time
         << " but already have local data at time " << time_);
  time_ = time;
  remote_data_ = std::move(vars);
}

template <typename LocalVars, typename RemoteVars>
std::pair<LocalVars, RemoteVars>
SimpleBoundaryData<LocalVars, RemoteVars>::extract() noexcept {
  ASSERT(local_data_ and remote_data_,
         "Tried to extract boundary data, but do not have "
         << (local_data_ ? "remote" : remote_data_ ? "local" : "any")
         << " data.");
  const auto result =
      std::make_pair(std::move(*local_data_), std::move(*remote_data_));
  local_data_ = boost::none;
  remote_data_ = boost::none;
  return result;
}

template <typename LocalVars, typename RemoteVars>
void SimpleBoundaryData<LocalVars, RemoteVars>::pup(PUP::er& p) noexcept {
  p | time_;
  p | local_data_;
  p | remote_data_;
}

}  // namespace dg
