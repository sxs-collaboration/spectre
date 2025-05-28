// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Particles/MonteCarlo/System.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/System.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"

using neutrino_species_elec = tmpl::list<neutrinos::ElectronNeutrinos<1>>;
using neutrino_species_elec_anti =
    tmpl::list<neutrinos::ElectronNeutrinos<1>,
               neutrinos::ElectronAntiNeutrinos<1>>;

#define GHMHD_NEUTRINOS                                                    \
  (RadiationTransport::NoNeutrinos::System, Particles::MonteCarlo::System, \
   RadiationTransport::M1Grey::System<neutrino_species_elec>,              \
   RadiationTransport::M1Grey::System<neutrino_species_elec_anti>)
