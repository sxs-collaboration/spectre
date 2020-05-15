// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"

#include <deque>
#include <memory>
#include <tuple>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/TimeStepId.hpp"

namespace Cce::InterfaceManagers {

std::unique_ptr<GhInterfaceManager> GhLockstep::get_clone() const noexcept {
  return std::make_unique<GhLockstep>(*this);
}

void GhLockstep::insert_gh_data(
    TimeStepId time_id, tnsr::aa<DataVector, 3> spacetime_metric,
    tnsr::iaa<DataVector, 3> phi, tnsr::aa<DataVector, 3> pi,
    const tnsr::aa<DataVector, 3> /*dt_spacetime_metric*/,
    const tnsr::iaa<DataVector, 3> /*dt_phi*/,
    const tnsr::aa<DataVector, 3> /*dt_pi*/) noexcept {
  // NOLINTNEXTLINE(performance-move-const-arg)
  provided_data_.emplace_back(std::move(time_id), std::move(spacetime_metric),
                              std::move(phi), std::move(pi));
}

boost::optional<std::tuple<TimeStepId, tnsr::aa<DataVector, 3>,
                           tnsr::iaa<DataVector, 3>, tnsr::aa<DataVector, 3>>>
GhLockstep::retrieve_and_remove_first_ready_gh_data() noexcept {
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
