// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/System.hpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"

template class RecordTimeStepperData<ForceFree::System>;
template class UpdateU<ForceFree::System, false>;
template class UpdateU<ForceFree::System, true>;
