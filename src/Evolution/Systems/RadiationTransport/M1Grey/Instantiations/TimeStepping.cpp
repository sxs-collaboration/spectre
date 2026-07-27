// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RadiationTransport/M1Grey/System.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "Time/ChangeTimeStepperOrder.tpp"
#include "Time/CleanHistory.tpp"
#include "Time/RecordTimeStepperData.tpp"
#include "Time/UpdateU.tpp"
#include "Utilities/TMPL.hpp"

template class ChangeTimeStepperOrder<RadiationTransport::M1Grey::System<
    tmpl::list<neutrinos::ElectronNeutrinos<1>>>>;
template class CleanHistory<RadiationTransport::M1Grey::System<
    tmpl::list<neutrinos::ElectronNeutrinos<1>>>>;
template class RecordTimeStepperData<RadiationTransport::M1Grey::System<
    tmpl::list<neutrinos::ElectronNeutrinos<1>>>>;
template class UpdateU<RadiationTransport::M1Grey::System<
    tmpl::list<neutrinos::ElectronNeutrinos<1>>>>;
