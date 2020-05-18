// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/InterfaceManagers/GhLocalTimeStepping.hpp"

#include <cstddef>
#include <deque>
#include <memory>
#include <tuple>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/History.hpp"
#include "Time/TimeStepId.hpp"

namespace Cce::InterfaceManagers {

std::unique_ptr<GhInterfaceManager> GhLocalTimeStepping::get_clone()
    const noexcept {
  return std::make_unique<GhLocalTimeStepping>(*this);
}

void GhLocalTimeStepping::insert_gh_data(
    TimeStepId time_id, const tnsr::aa<DataVector, 3>& spacetime_metric,
    const tnsr::iaa<DataVector, 3>& phi, const tnsr::aa<DataVector, 3>& pi,
    TimeStepId next_time_id, const tnsr::aa<DataVector, 3>& dt_spacetime_metric,
    const tnsr::iaa<DataVector, 3>& dt_phi,
    const tnsr::aa<DataVector, 3>& dt_pi) noexcept {
  gh_variables input_gh_variables{get<0, 0>(spacetime_metric).size()};
  dt_gh_variables input_dt_gh_variables{get<0, 0>(spacetime_metric).size()};
  get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>(
      input_gh_variables) = spacetime_metric;
  get<::Tags::dt<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>>(
      input_dt_gh_variables) = dt_spacetime_metric;
  get<GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>(input_gh_variables) =
      pi;
  get<::Tags::dt<GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>>(
      input_dt_gh_variables) = dt_pi;
  get<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(
      input_gh_variables) = phi;
  get<::Tags::dt<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>>(
      input_dt_gh_variables) = dt_phi;

  // If no pending requests, we don't know what time will be needed next, so
  // just stash the data in the deque.
  if (requests_.empty() or
      requests_.front().substep_time().value() <=
          time_id.substep_time().value() or
      not pre_history_.empty()) {
    // NOLINTNEXTLINE(performance-move-const-arg)
    pre_history_.emplace_back(std::move(time_id), std::move(input_gh_variables),
                              // NOLINTNEXTLINE(performance-move-const-arg)
                              std::move(next_time_id),
                              std::move(input_dt_gh_variables));
    return;
  }

  // if we have pending requests, we know how much history to insert, so we take
  // from the deque and then add in the current data if suitable
  // NOLINTNEXTLINE(performance-move-const-arg)
  boundary_history_.insert(std::move(time_id), input_gh_variables,
                           input_dt_gh_variables);
  // NOLINTNEXTLINE(performance-move-const-arg)
  latest_next_ = std::move(next_time_id);

  if (boundary_history_.size() > order_) {
    boundary_history_.mark_unneeded(
        boundary_history_.begin() +
        static_cast<ptrdiff_t>(boundary_history_.size() - order_));
  }
}

void GhLocalTimeStepping::request_gh_data(const TimeStepId& time_id) noexcept {
  requests_.push_back(time_id);
  if (requests_.size() == 1) {
    update_history();
  }
}

void GhLocalTimeStepping::update_history() noexcept {
  if (requests_.empty()) {
    return;
  }
  while (not pre_history_.empty() and
         requests_.front().substep_time().value() >
             get<0>(pre_history_.front()).substep_time().value()) {
    boundary_history_.insert(get<0>(pre_history_.front()),
                             get<1>(pre_history_.front()),
                             get<3>(pre_history_.front()));
    latest_next_ = get<2>(pre_history_.front());
    pre_history_.pop_front();
  }

  if (boundary_history_.size() > order_) {
    boundary_history_.mark_unneeded(
        boundary_history_.begin() +
        static_cast<ptrdiff_t>(boundary_history_.size() - order_));
  }
}

auto GhLocalTimeStepping::retrieve_and_remove_first_ready_gh_data() noexcept
    -> boost::optional<std::tuple<TimeStepId, gh_variables>> {
  if (requests_.empty()) {
    return boost::none;
  }
  const double first_request = requests_.front().substep_time().value();
  if ((boundary_history_.end() - 1)->value() < first_request and
      latest_next_.substep_time().value() >= first_request) {
    gh_variables latest_values = (boundary_history_.end() - 1).value();
    time_stepper_.dense_update_u(make_not_null(&latest_values),
                                 boundary_history_, first_request);
    // NOLINTNEXTLINE(performance-move-const-arg)
    std::tuple requested_data{std::move(requests_.front()),
                              std::move(latest_values)};
    requests_.pop_front();
    update_history();
    return requested_data;
  }
  return boost::none;
}

void GhLocalTimeStepping::pup(PUP::er& p) noexcept {
  p | order_;
  p | pre_history_;
  p | requests_;
  p | boundary_history_;
  p | latest_next_;
  p | time_stepper_;
}

/// \cond
PUP::able::PUP_ID GhLocalTimeStepping::my_PUP_ID = 0;
/// \endcond
}  // namespace Cce::InterfaceManagers
