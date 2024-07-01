// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <map>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/Particles/MonteCarlo/GhostZoneCommunicationStep.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
/// \endcond

/// Items related to the evolution of particles
/// Items related to Monte-Carlo radiation transport
namespace Particles::MonteCarlo {

/*!
 * \brief The container for ghost zone data to be communicated
 *        in the Monte-Carlo algorithm
 *
 * The stored data consists of
 * (1) The fluid variables in those ghost cells, in the form
 *     of a DataVector. The data should contain the
 *     scalars (baryon density, temperature, electron fraction,
 *     and cell light-crossing time) in pre-step communication
 *     and (coupling_tilde_tau, coupling_tilde_s, coupling_rho_ye)
 *     in post-step communication
 * (2) The packets moved from a neighbor to this element. This
 *     can be std::null_ptr in pre-step communication
 */
template <size_t Dim>
struct McGhostZoneData {
  void pup(PUP::er& p) {
    p | ghost_zone_hydro_variables;
    p | packets_entering_this_element;
  }

  DataVector ghost_zone_hydro_variables{};
  std::optional<std::vector<Particles::MonteCarlo::Packet>>
      packets_entering_this_element{};
};

/// Inbox tag to be used before MC step
template <size_t Dim, CommunicationStep CommStep>
struct McGhostZoneDataInboxTag {
  using stored_type = McGhostZoneData<Dim>;

  using temporal_id = TimeStepId;
  using type = std::map<TimeStepId, DirectionalIdMap<Dim, stored_type>>;
  using value_type = type;

  CommunicationStep comm_step = CommStep;

  template <typename ReceiveDataType>
  static size_t insert_into_inbox(const gsl::not_null<type*> inbox,
                                  const temporal_id& time_step_id,
                                  ReceiveDataType&& data) {
    auto& current_inbox = (*inbox)[time_step_id];
    if (not current_inbox.insert(std::forward<ReceiveDataType>(data))
                  .second) {
     ERROR("Failed to insert data to receive at instance '"
           << time_step_id
           << " in McGhostZonePreStepDataInboxTag.\n");
    }
    return current_inbox.size();
  }

  void pup(PUP::er& /*p*/) {}
};

// Tags to put in DataBox for MC communication.
namespace Tags {

/// Simple tag for the structure containing ghost zone data
/// for Monte Carlo radiation transport.
template <size_t Dim>
struct McGhostZoneDataTag : db::SimpleTag {
  using type = DirectionalIdMap<Dim, McGhostZoneData<Dim>>;
};

}  // namespace Tags

} // namespace Particles::MonteCarlo
