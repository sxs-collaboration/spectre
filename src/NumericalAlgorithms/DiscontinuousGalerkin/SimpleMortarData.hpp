// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>
#include <optional>
#include <pup.h>  // IWYU pragma: keep
#include <utility>

#include "Parallel/PupStlCpp17.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"

namespace dg {

/// \ingroup DiscontinuousGalerkinGroup
/// \brief Storage of boundary data on two sides of a mortar
///
/// Typically, values are inserted into this container by the flux
/// communication actions.
template <typename TemporalId, typename LocalVars, typename RemoteVars>
class SimpleMortarData {
 public:
  SimpleMortarData() = default;
  SimpleMortarData(const SimpleMortarData&) = default;
  SimpleMortarData(SimpleMortarData&&) = default;
  SimpleMortarData& operator=(const SimpleMortarData&) = default;
  SimpleMortarData& operator=(SimpleMortarData&&) = default;
  ~SimpleMortarData() = default;

  /// The argument is ignored.  It exists for compatibility with
  /// BoundaryHistory.
  explicit SimpleMortarData(const size_t /*integration_order*/) noexcept {}

  /// These functions do nothing.  They exist for compatibility with
  /// BoundaryHistory.
  /// @{
  size_t integration_order() const noexcept { return 0; }
  void integration_order(const size_t /*integration_order*/) noexcept {}
  /// @}

  /// Add a value.  This function must be called once between calls to
  /// extract.
  /// @{
  void local_insert(TemporalId temporal_id, LocalVars vars) noexcept;
  void remote_insert(TemporalId temporal_id, RemoteVars vars) noexcept;
  /// @}

  /// Return the inserted data and reset the state to empty.
  std::pair<LocalVars, RemoteVars> extract() noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

  /// Retrieve the local data at `temporal_id`
  const LocalVars& local_data(const TemporalId& temporal_id) const noexcept {
    ASSERT(local_data_, "Local data not available.");
    ASSERT(temporal_id == temporal_id_,
           "Only have local data at temporal_id "
               << temporal_id_ << ", but requesting at " << temporal_id);
    return *local_data_;
  };

  /// Retrieve the remote data at `temporal_id`
  const RemoteVars& remote_data(const TemporalId& temporal_id) const noexcept {
    ASSERT(remote_data_, "Remote data not available.");
    ASSERT(temporal_id == temporal_id_,
           "Only have remote data at temporal_id "
               << temporal_id_ << ", but requesting at " << temporal_id);
    return *remote_data_;
  };

 private:
  TemporalId temporal_id_{};
  std::optional<LocalVars> local_data_{};
  std::optional<RemoteVars> remote_data_{};
};

template <typename TemporalId, typename LocalVars, typename RemoteVars>
void SimpleMortarData<TemporalId, LocalVars, RemoteVars>::local_insert(
    TemporalId temporal_id, LocalVars vars) noexcept {
  ASSERT(not local_data_.has_value(), "Already received local data.");
  ASSERT(not remote_data_.has_value() or temporal_id == temporal_id_,
         "Received local data at " << temporal_id
                                   << ", but already have remote data at "
                                   << temporal_id_);
  temporal_id_ = std::move(temporal_id);
  local_data_ = std::move(vars);
}

template <typename TemporalId, typename LocalVars, typename RemoteVars>
void SimpleMortarData<TemporalId, LocalVars, RemoteVars>::remote_insert(
    TemporalId temporal_id, RemoteVars vars) noexcept {
  ASSERT(not remote_data_.has_value(), "Already received remote data.");
  ASSERT(not local_data_.has_value() or temporal_id == temporal_id_,
         "Received remote data at " << temporal_id
                                    << ", but already have local data at "
                                    << temporal_id_);
  temporal_id_ = std::move(temporal_id);
  remote_data_ = std::move(vars);
}

template <typename TemporalId, typename LocalVars, typename RemoteVars>
std::pair<LocalVars, RemoteVars>
SimpleMortarData<TemporalId, LocalVars, RemoteVars>::extract() noexcept {
  ASSERT(local_data_.has_value() and remote_data_.has_value(),
         "Tried to extract boundary data, but do not have "
             << (local_data_ ? "remote" : remote_data_ ? "local" : "any")
             << " data.");
  auto result =
      std::make_pair(std::move(*local_data_), std::move(*remote_data_));
  local_data_ = std::nullopt;
  remote_data_ = std::nullopt;
  return result;
}

template <typename TemporalId, typename LocalVars, typename RemoteVars>
void SimpleMortarData<TemporalId, LocalVars, RemoteVars>::pup(
    PUP::er& p) noexcept {
  p | temporal_id_;
  p | local_data_;
  p | remote_data_;
}

}  // namespace dg
