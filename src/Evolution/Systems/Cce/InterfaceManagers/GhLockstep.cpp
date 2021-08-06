// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"

#include <deque>
#include <memory>
#include <tuple>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/TimeStepId.hpp"

namespace Cce::InterfaceManagers {

std::unique_ptr<GhInterfaceManager> GhLockstep::get_clone() const noexcept {
  return std::make_unique<GhLockstep>(*this);
}

void GhLockstep::insert_gh_data(TimeStepId time_id,
                                const tnsr::aa<DataVector, 3>& spacetime_metric,
                                const tnsr::iaa<DataVector, 3>& phi,
                                const tnsr::aa<DataVector, 3>& pi) noexcept {
  gh_variables input_gh_variables{get<0, 0>(spacetime_metric).size()};
  get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>(
      input_gh_variables) = spacetime_metric;
  get<GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>(input_gh_variables) =
      pi;
  get<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(
      input_gh_variables) = phi;
  // NOLINTNEXTLINE(performance-move-const-arg)
  provided_data_.insert({std::move(time_id), std::move(input_gh_variables)});
}

void GhLockstep::request_gh_data(const TimeStepId& time_id) noexcept {
  requests_.push_back(time_id);
}

auto GhLockstep::retrieve_and_remove_first_ready_gh_data() noexcept
    -> std::optional<std::tuple<TimeStepId, gh_variables>> {
  if (provided_data_.empty() or requests_.empty()) {
    return std::nullopt;
  }
  if (provided_data_.count(requests_.front()) == 0_st) {
    return std::nullopt;
  }
  const std::tuple<TimeStepId, gh_variables> return_data{
      requests_.front(), std::move(provided_data_[requests_.front()])};
  provided_data_.erase(requests_.front());
  requests_.pop_front();
  return return_data;
}

void GhLockstep::pup(PUP::er& p) noexcept {
  p | provided_data_;
  p | requests_;
}

PUP::able::PUP_ID GhLockstep::my_PUP_ID = 0;
}  // namespace Cce::InterfaceManagers
