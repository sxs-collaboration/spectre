// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>

#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

namespace Triggers {

/*!
 * \brief This trigger is true when the worldtube is entirely within a
 * coordinate sphere of radius 1.99 M centered on the origin in the inertial
 * frame. This assumes a black hole mass of 1M.
 */
class InsideHorizon : public SPECTRE_CHARM_DERIVED(InsideHorizon, Trigger) {
 public:
  /// \cond
  InsideHorizon() = default;
  WRAPPED_PUPable_decl_template(InsideHorizon);  // NOLINT
  /// \endcond

  static constexpr Options::String help =
      "Triggers if the worldtube is entirely inside a coordinate sphere with "
      "radius 1.99 M centered on the origin in the inertial frame.";
  static constexpr size_t Dim = 3;
  using options = tmpl::list<>;
  using argument_tags = tmpl::list<
      CurvedScalarWave::Worldtube::Tags::ParticlePositionVelocity<Dim>,
      CurvedScalarWave::Worldtube::Tags::WorldtubeRadiusParameters>;

  bool operator()(const std::array<tnsr::I<double, Dim, Frame::Inertial>, 2>&
                      position_and_velocity,
                  const std::array<double, 4>& worldtube_radius_params) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) override {}
};
}  // namespace Triggers
