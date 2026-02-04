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
      const std::array<double, NeutrinoSpecies> initial_packet_energy)
      : initial_packet_energy_(initial_packet_energy) {}

  static constexpr Options::String help = {
      "Global options for Monte-Carlo evolution.\n"
      "InitialPacketEnergy: [double, double, double]    \n"};

  struct InitialPacketEnergy {
    using type = std::array<double, NeutrinoSpecies>;
    static constexpr Options::String help{
        "Initial energy used to create packets"};
  };

  using options = tmpl::list<InitialPacketEnergy>;

  using PUP::able::register_constructor;
  void pup(PUP::er& p) override;
  WRAPPED_PUPable_decl_template(MonteCarloOptions);

  const std::array<double, NeutrinoSpecies>& get_initial_packet_energy() const {
    return initial_packet_energy_;
  }

 private:
  std::array<double, NeutrinoSpecies> initial_packet_energy_;
};

}  // namespace Particles::MonteCarlo
