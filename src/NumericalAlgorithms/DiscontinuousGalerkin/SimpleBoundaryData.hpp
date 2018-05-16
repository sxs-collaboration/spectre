// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/none.hpp>
#include <boost/optional.hpp>
#include <ostream>
#include <pup.h>  // IWYU pragma: keep
#include <utility>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/BoostHelpers.hpp"  // IWYU pragma: keep

namespace dg {

/// \ingroup DiscontinuousGalerkinGroup
/// \brief Storage of boundary data on two sides of a mortar
///
/// Typically, values are inserted into this container by the flux
/// communication actions.
template <typename TemporalId, typename LocalVars, typename RemoteVars>
class SimpleBoundaryData {
 public:
  /// Add a value.  This function must be called once between calls to
  /// extract.
  //@{
  void local_insert(TemporalId temporal_id, LocalVars vars) noexcept;
  void remote_insert(TemporalId temporal_id, RemoteVars vars) noexcept;
  //@}

  /// Return the inserted data and reset the state to empty.
  std::pair<LocalVars, RemoteVars> extract() noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  TemporalId temporal_id_{};
  boost::optional<LocalVars> local_data_{};
  boost::optional<RemoteVars> remote_data_{};
};

template <typename TemporalId, typename LocalVars, typename RemoteVars>
void SimpleBoundaryData<TemporalId, LocalVars, RemoteVars>::local_insert(
    TemporalId temporal_id, LocalVars vars) noexcept {
  ASSERT(not local_data_, "Already received local data.");
  ASSERT(not remote_data_ or temporal_id == temporal_id_,
         "Received local data at " << temporal_id
         << ", but already have remote data at " << temporal_id_);
  temporal_id_ = std::move(temporal_id);
  local_data_ = std::move(vars);
}

template <typename TemporalId, typename LocalVars, typename RemoteVars>
void SimpleBoundaryData<TemporalId, LocalVars, RemoteVars>::remote_insert(
    TemporalId temporal_id, RemoteVars vars) noexcept {
  ASSERT(not remote_data_, "Already received remote data.");
  ASSERT(not local_data_ or temporal_id == temporal_id_,
         "Received remote data at " << temporal_id
         << ", but already have local data at " << temporal_id_);
  temporal_id_ = std::move(temporal_id);
  remote_data_ = std::move(vars);
}

template <typename TemporalId, typename LocalVars, typename RemoteVars>
std::pair<LocalVars, RemoteVars>
SimpleBoundaryData<TemporalId, LocalVars, RemoteVars>::extract() noexcept {
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

template <typename TemporalId, typename LocalVars, typename RemoteVars>
void SimpleBoundaryData<TemporalId, LocalVars, RemoteVars>::pup(
    PUP::er& p) noexcept {
  p | temporal_id_;
  p | local_data_;
  p | remote_data_;
}

}  // namespace dg
