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

void GhLockstep::insert_gh_data(
    TimeStepId time_id, tnsr::aa<DataVector, 3> spacetime_metric,
    tnsr::iaa<DataVector, 3> phi, tnsr::aa<DataVector, 3> pi,
    TimeStepId /*next_time_id*/,
    const tnsr::aa<DataVector, 3> /*dt_spacetime_metric*/,
    const tnsr::iaa<DataVector, 3> /*dt_phi*/,
    const tnsr::aa<DataVector, 3> /*dt_pi*/) noexcept {
  // NOLINTNEXTLINE(performance-move-const-arg)
  gh_variables input_gh_variables{get<0, 0>(spacetime_metric).size()};
  get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>(
      input_gh_variables) = spacetime_metric;
  get<GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>(input_gh_variables) =
      pi;
  get<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(
      input_gh_variables) = phi;
  provided_data_.emplace_back(std::move(time_id),
                              std::move(input_gh_variables));
}

auto GhLockstep::retrieve_and_remove_first_ready_gh_data() noexcept
    -> boost::optional<std::tuple<TimeStepId, gh_variables>> {
  if (provided_data_.empty()) {
    return boost::none;
  }
  const auto return_data = std::move(provided_data_.front());
  provided_data_.pop_front();
  return return_data;
}

void GhLockstep::pup(PUP::er& p) noexcept { p | provided_data_; }

/// \cond
PUP::able::PUP_ID GhLockstep::my_PUP_ID = 0;
/// \endcond
}  // namespace Cce::InterfaceManagers
