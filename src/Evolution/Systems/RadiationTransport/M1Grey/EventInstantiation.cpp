// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RadiationTransport/M1Grey/System.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "ParallelAlgorithms/Events/ObserveTimeStep.tpp"
#include "Utilities/TMPL.hpp"

template class Events::ObserveTimeStep<RadiationTransport::M1Grey::System<
    tmpl::list<neutrinos::ElectronNeutrinos<1>>>>;
