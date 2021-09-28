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
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterpolationStrategies.hpp"
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
 * evolution are not identically stepped is considered undefined behavior. The
 * outcome will likely be that CCE will fail to evolve and the boundary data
 * will be continually inserted into a rapidly expanding inbox.
 */
class GhLockstep : public GhInterfaceManager {
 public:
  using GhInterfaceManager::gh_variables;

  static constexpr Options::String help{
      "Pass data between GH and CCE systems on matching timesteps only."};

  using options = tmpl::list<>;

  GhLockstep() = default;

  explicit GhLockstep(CkMigrateMessage* /*unused*/) {}

  WRAPPED_PUPable_decl_template(GhLockstep);  // NOLINT

  std::unique_ptr<GhInterfaceManager> get_clone() const override;

  /// \brief Store a provided data set in a `std::deque`.
  ///
  /// \details The lock-step constraint ensures that only the generalized
  /// harmonic variables `spacetime_metric`, `phi`, and `pi` are used. The
  /// remaining variables are accepted to comply with the more general abstract
  /// interface.
  void insert_gh_data(TimeStepId time_id,
                      const tnsr::aa<DataVector, 3>& spacetime_metric,
                      const tnsr::iaa<DataVector, 3>& phi,
                      const tnsr::aa<DataVector, 3>& pi,
                      const tnsr::aa<DataVector, 3>& dt_spacetime_metric = {},
                      const tnsr::iaa<DataVector, 3>& dt_phi = {},
                      const tnsr::aa<DataVector, 3>& dt_pi = {}) override;

  /// \brief next time information is ignored by this implementation, so this is
  /// a no-op.
  void insert_next_gh_time(TimeStepId /*time_id*/,
                           TimeStepId /*next_time_id*/) override {}

  /// \brief Requests are ignored by this implementation, so this is a no-op.
  void request_gh_data(const TimeStepId& /*time_id*/) override {}

  /// \brief Return a `std::optional<std::tuple>` of the least recently
  /// submitted generalized harmonic boundary data if any exists and removes it
  /// from the internal `std::deque`, otherwise returns `std::nullopt`.
  auto retrieve_and_remove_first_ready_gh_data()
      -> std::optional<std::tuple<TimeStepId, gh_variables>> override;

  /// \brief This class ignores requests to ensure a one-way communication
  /// pattern, so the number of requests is always 0.
  size_t number_of_pending_requests() const override { return 0; }

  /// \brief The number of times at which data from a GH evolution have been
  /// stored and not yet retrieved
  size_t number_of_gh_times() const override { return provided_data_.size(); }

  /// Serialization for Charm++.
  void pup(PUP::er& p) override;

  InterpolationStrategy get_interpolation_strategy() const override {
    return InterpolationStrategy::EverySubstep;
  };

 private:
  std::deque<std::tuple<TimeStepId, gh_variables>> provided_data_;
};

}  // namespace Cce::InterfaceManagers
