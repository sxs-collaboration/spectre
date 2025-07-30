// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarTensor/System.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStepVolume.tpp"

template class dg::Events::ObserveTimeStepVolume<ScalarTensor::System>;
