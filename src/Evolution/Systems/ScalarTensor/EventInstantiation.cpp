// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarTensor/System.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.tpp"
#include "ParallelAlgorithms/Events/ObserveTimeStepVolume.tpp"

template class Events::ObserveTimeStep<ScalarTensor::System>;
template class dg::Events::ObserveTimeStepVolume<ScalarTensor::System>;
