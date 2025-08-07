// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/System.hpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"

template class RecordTimeStepperData<Burgers::System>;
template class UpdateU<Burgers::System, false>;
template class UpdateU<Burgers::System, true>;
