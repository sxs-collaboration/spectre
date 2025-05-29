// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Particles/MonteCarlo/System.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"

#define GHMHD_NEUTRINOS \
  (RadiationTransport::NoNeutrinos::System, Particles::MonteCarlo::System)
