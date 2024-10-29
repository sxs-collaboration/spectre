// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <random>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Particles/MonteCarlo/MonteCarloOptions.hpp"
#include "Evolution/Particles/MonteCarlo/NeutrinoInteractionTable.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"

/// Items related to the evolution of particles
/// Items related to Monte-Carlo radiation transport
/// Tags for MC
namespace Particles::MonteCarlo::Tags {

/// Simple tag containing the vector of Monte-Carlo
/// packets belonging to an element.
struct PacketsOnElement : db::SimpleTag {
  using type = std::vector<Particles::MonteCarlo::Packet>;
};

/// Simple tag containing an approximation of the light
/// crossing time for each cell (the shortest time among
/// all coordinate axis directions).
template <typename DataType>
struct CellLightCrossingTime : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// Simple tag storing the coupling term between
/// MC and tilde_Tau (i.e. the energy variable)
template <typename DataType>
struct CouplingTildeTau : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// Simple tag storing the coupling term between
/// MC and tilde_RhoYe (i.e. the composition variable)
template <typename DataType>
struct CouplingTildeRhoYe : db::SimpleTag {
  using type = Scalar<DataType>;
};

/// Simple tag storing the coupling term between
/// MC and tilde_S (i.e. the momentum variable)
template <typename DataType, size_t Dim>
struct CouplingTildeS : db::SimpleTag {
  using type = tnsr::i<DataType, Dim, Frame::Inertial>;
};

/// Simple tag storing the random number generator
/// used by Monte-Carlo
struct RandomNumberGenerator : db::SimpleTag {
  using type = std::mt19937;
};

/// Simple tag containing the minimum energy of
/// packets at the current time. This can depend
/// on neutrino species and time, but not location.
template <size_t NeutrinoSpecies>
struct MinimumPacketEnergyAtEmission : db::SimpleTag {
  using type = std::array<double, NeutrinoSpecies>;
};

/// Simple tag for the table of neutrino-matter interaction
/// rates (emission, absorption and scattering for each
/// energy bin and neutrino species).
template <size_t EnergyBins, size_t NeutrinoSpecies>
struct InteractionRatesTable : db::SimpleTag {
  using type =
      std::unique_ptr<NeutrinoInteractionTable<EnergyBins, NeutrinoSpecies>>;
  static constexpr bool pass_metavariables = false;
  using option_tags =
      typename NeutrinoInteractionTable<EnergyBins, NeutrinoSpecies>::options;
  static type create_from_options(const std::string filename) {
    std::unique_ptr<Particles::MonteCarlo::NeutrinoInteractionTable<
        EnergyBins, NeutrinoSpecies>>
        interaction_table_ptr =
            std::make_unique<Particles::MonteCarlo::NeutrinoInteractionTable<
                EnergyBins, NeutrinoSpecies>>(filename);
    return interaction_table_ptr;
    ;
  }
};

template <size_t NeutrinoSpecies>
struct MonteCarloOptions : db::SimpleTag {
  using type = std::unique_ptr<
      Particles::MonteCarlo::MonteCarloOptions<NeutrinoSpecies>>;
  static constexpr bool pass_metavariables = false;
  using option_tags = typename Particles::MonteCarlo::MonteCarloOptions<
      NeutrinoSpecies>::options;
  static type create_from_options(
      const std::array<double, NeutrinoSpecies> initial_packet_energy,
      const size_t desired_packets_per_species) {
    std::unique_ptr<Particles::MonteCarlo::MonteCarloOptions<NeutrinoSpecies>>
        mc_options_ptr = std::make_unique<
            Particles::MonteCarlo::MonteCarloOptions<NeutrinoSpecies>>(
            initial_packet_energy, desired_packets_per_species);
    return mc_options_ptr;
    ;
  }
};

}  // namespace Particles::MonteCarlo::Tags
