// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/CleanMortarHistory.tpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/System.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "Utilities/TMPL.hpp"

template class evolution::dg::CleanMortarHistory<
    RadiationTransport::M1Grey::System<
        tmpl::list<neutrinos::ElectronNeutrinos<1>>>>;
