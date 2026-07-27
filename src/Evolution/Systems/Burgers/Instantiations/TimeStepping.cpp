// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/System.hpp"
#include "Time/ChangeTimeStepperOrder.tpp"
#include "Time/CleanHistory.tpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"

template class ChangeTimeStepperOrder<Burgers::System>;
template class CleanHistory<Burgers::System>;
template class RecordTimeStepperData<Burgers::System>;
template class UpdateU<Burgers::System>;
