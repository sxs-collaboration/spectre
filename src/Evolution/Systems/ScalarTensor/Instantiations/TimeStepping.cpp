// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarTensor/System.hpp"
#include "Time/CleanHistory.tpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"

template class CleanHistory<ScalarTensor::System>;
template class RecordTimeStepperData<ScalarTensor::System>;
template class UpdateU<ScalarTensor::System, false>;
template class UpdateU<ScalarTensor::System, true>;
