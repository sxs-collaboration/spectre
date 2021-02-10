// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <tuple>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterpolationStrategies.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/History.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::InterfaceManagers {

/*!
 * \brief Implementation of a `GhInterfaceManager` that provides data according
 * to local time-stepping dense output.
 *
 * \details This class receives data from the Generalized Harmonic system
 * sufficient to perform the time-stepping dense output to arbitrary times
 * required by CCE. From the Generalized Harmonic system, it receives the
 * spacetime metric \f$g_{a b}\f$ and Generalized Harmonic \f$\Phi_{i a b}\f$
 * and \f$\Pi_{ab}\f$, as well as each of their time derivatives, the current
 * `TimeStepId`, and the next `TimeStepId` via
 * `GhLocalTimeStepping::insert_gh_data()`. The CCE system supplies requests for
 * time steps via `GhLocalTimeStepping::request_gh_data()` and receives dense
 * output boundary data via
 * `GhLocalTimeStepping::retrieve_and_remove_first_ready_gh_data()`.
 */
class GhLocalTimeStepping : public GhInterfaceManager {
 public:
  using dt_gh_variables = Variables<tmpl::list<
      ::Tags::dt<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>,
      ::Tags::dt<GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>,
      ::Tags::dt<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>>>;

  struct AdamsBashforthOrder {
    using type = size_t;
    static constexpr Options::String help = {
        "Convergence order for the internal Adams-Bashforth stepper"};
    static type lower_bound() noexcept { return 1; }
    static type upper_bound() noexcept {
      return TimeSteppers::AdamsBashforthN::maximum_order;
    }
  };

  static constexpr Options::String help{
      "Pass data between GH and CCE systems via Adams-Bashforth local "
      "time-stepping"};

  using options = tmpl::list<AdamsBashforthOrder>;

  GhLocalTimeStepping() = default;

  explicit GhLocalTimeStepping(const size_t order)
      : order_{order}, boundary_history_{order}, time_stepper_{order} {}

  explicit GhLocalTimeStepping(CkMigrateMessage* /*unused*/) noexcept {}

  WRAPPED_PUPable_decl_template(GhLocalTimeStepping);  // NOLINT

  std::unique_ptr<GhInterfaceManager> get_clone() const noexcept override;

  /// \brief Store the provided data set to prepare for time-stepping dense
  /// output.
  ///
  /// \details The `next_time_id` is required to infer the span of time values
  /// that should be permitted for the dense output, and at what point the CCE
  /// system should wait for additional data from the GH system.
  void insert_gh_data(TimeStepId time_id,
                      const tnsr::aa<DataVector, 3>& spacetime_metric,
                      const tnsr::iaa<DataVector, 3>& phi,
                      const tnsr::aa<DataVector, 3>& pi,
                      const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
                      const tnsr::iaa<DataVector, 3>& dt_phi,
                      const tnsr::aa<DataVector, 3>& dt_pi) noexcept override;

  void insert_next_gh_time(TimeStepId time_id,
                           TimeStepId next_time_id) noexcept override;

  /// \brief Store the next time step that will be required by the CCE system to
  /// proceed with the evolution.
  ///
  /// \details The values of these time steps will be used to generate the dense
  /// output from the provided GH data.
  void request_gh_data(const TimeStepId& time_id) noexcept override;

  /// \brief Return a `std::optional` of either the dense-output data at the
  /// least recently requested time, or `std::nullopt` if not enough GH data has
  /// been supplied yet.
  auto retrieve_and_remove_first_ready_gh_data() noexcept
      -> std::optional<std::tuple<TimeStepId, gh_variables>> override;

  /// The number of requests that have been submitted and not yet retrieved.
  size_t number_of_pending_requests() const noexcept override {
    return requests_.size();
  }

  /// \brief  The number of times for which data from the GH system is stored.
  ///
  /// \details  This will be roughly the order of the time stepper plus the
  /// number of times that the GH system is ahead of the CCE system.
  size_t number_of_gh_times() const noexcept override {
    return pre_history_.size() + boundary_history_.size();
  }

  /// Serialization for Charm++.
  void pup(PUP::er& p) noexcept override;

  InterpolationStrategy get_interpolation_strategy() const noexcept override {
    return InterpolationStrategy::EveryStep;
  };

 private:
  // performs the needed logic to move entries from pre_history_ into the
  // boundary_history_ as appropriate for the current requests_
  void update_history() noexcept;

  size_t order_ = 3;

  std::deque<
      std::tuple<TimeStepId, std::optional<gh_variables>,
                 std::optional<TimeStepId>, std::optional<dt_gh_variables>>>
      pre_history_;
  std::deque<TimeStepId> requests_;

  TimeSteppers::History<gh_variables, dt_gh_variables> boundary_history_;
  TimeStepId latest_next_;
  TimeSteppers::AdamsBashforthN time_stepper_;
};

}  // namespace Cce::InterfaceManagers
