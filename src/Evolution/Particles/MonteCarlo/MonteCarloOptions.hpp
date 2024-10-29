// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <pup.h>

#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Particles::MonteCarlo {

template <size_t NeutrinoSpecies>
class MonteCarloOptions : public PUP::able {
 public:
  explicit MonteCarloOptions(
      const std::array<double, NeutrinoSpecies> initial_packet_energy,
      const size_t desired_packets_per_species)
      : initial_packet_energy_(initial_packet_energy),
        desired_packets_per_species_(desired_packets_per_species) {}

  static constexpr Options::String help = {
      "Global options for Monte-Carlo evolution.\n"
      "InitialPacketEnergy: [double, double, double]    \n"
      "DesiredPacketsPerSpecies: size_t                 \n"};

  struct InitialPacketEnergy {
    using type = std::array<double, NeutrinoSpecies>;
    static constexpr Options::String help{
        "Initial energy used to create packets"};
  };

  struct DesiredPacketsPerSpecies {
    using type = size_t;
    static constexpr Options::String help{
        "Target number of MC packets per species"};
  };

  using options = tmpl::list<InitialPacketEnergy, DesiredPacketsPerSpecies>;

  explicit MonteCarloOptions(CkMigrateMessage* msg) : PUP::able(msg) {}

  using PUP::able::register_constructor;
  void pup(PUP::er& p) override;
  WRAPPED_PUPable_decl_template(MonteCarloOptions);

  const std::array<double, NeutrinoSpecies>& get_initial_packet_energy() const {
    return initial_packet_energy_;
  }

  const size_t& get_desired_packets_per_species() const {
    return desired_packets_per_species_;
  }

 private:
  std::array<double, NeutrinoSpecies> initial_packet_energy_;
  size_t desired_packets_per_species_;
};

}  // namespace Particles::MonteCarlo
