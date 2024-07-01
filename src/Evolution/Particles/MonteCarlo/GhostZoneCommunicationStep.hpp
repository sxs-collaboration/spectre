// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// Items related to the evolution of particles
/// Items related to Monte-Carlo radiation transport
namespace Particles::MonteCarlo {

/// This class is used to template communication actions
/// in the Monte-Carlo code, to know which of the two
/// communication steps we need to take care of
/// (just before or just after the MC step).
enum class CommunicationStep {
  /// PreStep should occur just before the MC step.
  /// It should send to the ghost zones of each
  /// element the fluid and metric data.
  PreStep,
  /// PostStep should occur just after a MC step.
  /// It sends packets that have left their current
  /// element to the element they now belong to.
  /// NOT CODE YET: This is also when information
  /// about energy/momentum/lepton number exchance
  /// in the ghost zones should be communicated back
  /// to the corresponding live point for coupling to
  /// the fluid evolution.
  PostStep
};

}  // namespace Particles::MonteCarlo
