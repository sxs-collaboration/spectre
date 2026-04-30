// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Time/ChangeTimeStepperOrder.tpp"
#include "Time/CleanHistory.tpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"

template class ChangeTimeStepperOrder<grmhd::ValenciaDivClean::System>;
template class CleanHistory<grmhd::ValenciaDivClean::System>;
template class RecordTimeStepperData<grmhd::ValenciaDivClean::System>;
template class UpdateU<grmhd::ValenciaDivClean::System>;
