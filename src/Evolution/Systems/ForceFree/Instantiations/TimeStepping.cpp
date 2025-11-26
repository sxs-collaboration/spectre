// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/System.hpp"
#include "Time/ChangeTimeStepperOrder.tpp"
#include "Time/CleanHistory.tpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"

template class ChangeTimeStepperOrder<ForceFree::System>;
template class CleanHistory<ForceFree::System>;
template class RecordTimeStepperData<ForceFree::System>;
template class UpdateU<ForceFree::System, false>;
template class UpdateU<ForceFree::System, true>;
