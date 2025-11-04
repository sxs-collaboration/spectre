// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"

template class RecordTimeStepperData<grmhd::ValenciaDivClean::System>;
template class UpdateU<grmhd::ValenciaDivClean::System, false>;
template class UpdateU<grmhd::ValenciaDivClean::System, true>;
