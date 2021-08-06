// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <tuple>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/TimeStepId.hpp"

namespace Cce::InterfaceManagers {

/*!
 * \brief Simple implementation of a `GhInterfaceManager` that only
 * provides boundary data on matching `TimeStepId`s
 *
 * \details This version of the interface manager assumes that the CCE system
 * and the generalized harmonic system that it communicates with evolve with an
 * identical time stepper and on identical time step intervals (they evolve in
 * 'lock step'). As a result, and to streamline communications, new data is
 * always immediately 'ready' and requests are ignored to produce the behavior
 * of just immediately gauge-transforming and sending the data to the CCE
 * component as soon as it becomes available from the GH system.
 *
 * \warning Using this interface manager when the GH components and the CCE
 * evolution are not identically stepped is considered undefined behavior. For
 * current implementations involving dense output and local time-stepping, this
 * interface manager can only be used during self-start.
 */
class GhLockstep : public GhInterfaceManager {
 public:
  using GhInterfaceManager::gh_variables;

  static constexpr Options::String help{
      "Pass data between GH and CCE systems on matching timesteps only."};

  using options = tmpl::list<>;

  GhLockstep() = default;

  explicit GhLockstep(CkMigrateMessage* /*unused*/) noexcept {}

  WRAPPED_PUPable_decl_template(GhLockstep);  // NOLINT

  std::unique_ptr<GhInterfaceManager> get_clone() const noexcept override;

  /// \brief Store a provided data set in a `std::deque`.
  ///
  /// \details The lock-step constraint ensures that only the generalized
  /// harmonic variables `spacetime_metric`, `phi`, and `pi` are used. The
  /// remaining variables are accepted to comply with the more general abstract
  /// interface.
  void insert_gh_data(TimeStepId time_id,
                      const tnsr::aa<DataVector, 3>& spacetime_metric,
                      const tnsr::iaa<DataVector, 3>& phi,
                      const tnsr::aa<DataVector, 3>& pi) noexcept;

  void request_gh_data(const TimeStepId& time_id) noexcept override;

  /// \brief Return a `std::optional<std::tuple>` of the least recently
  /// submitted generalized harmonic boundary data if any exists and removes it
  /// from the internal `std::deque`, otherwise returns `std::nullopt`.
  auto retrieve_and_remove_first_ready_gh_data() noexcept
      -> std::optional<std::tuple<TimeStepId, gh_variables>> override;

  /// \brief This class ignores requests to ensure a one-way communication
  /// pattern, so the number of requests is always 0.
  size_t number_of_pending_requests() const noexcept override { return 0; }

  /// \brief The number of times at which data from a GH evolution have been
  /// stored and not yet retrieved
  size_t number_of_gh_times() const noexcept override {
    return provided_data_.size();
  }

  /// Serialization for Charm++.
  void pup(PUP::er& p) noexcept override;

 private:
  std::map<TimeStepId, gh_variables> provided_data_;
  std::set<TimeStepId> requests_;
};

}  // namespace Cce::InterfaceManagers
