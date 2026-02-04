// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <pup.h>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Time.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

namespace Triggers {
/*!
 * \brief This trigger returns true when the scalar charge is about to cross one
 * of the specified areal radii.
 *
 * \details As the domain is adjusted to track the position of the scalar
 * charge, the time step needs to be dynamically adjusted accordingly. This
 * trigger can be used to set the time step according to the radial position of
 * the scalar charge which gives a good approximation of when the time step
 * should be adjusted.
 * The trigger only approximates whether the particle might cross during the
 * next time step and may therefore fire twice.
 */
class OrbitRadius : public Trigger {
 public:
  /// \cond
  OrbitRadius() = default;
  WRAPPED_PUPable_decl_template(OrbitRadius);  // NOLINT
  /// \endcond

  struct Radii {
    using type = std::vector<double>;
    static constexpr Options::String help =
        "The orbital radii which should trigger when crossed by the "
        "scalar charge.";
  };

  static constexpr Options::String help =
      "Trigger that fires when the scalar charge crosses specified radii.";
  using options = tmpl::list<Radii>;

  explicit OrbitRadius(const std::vector<double>& radii);

  using argument_tags =
      tmpl::list<CurvedScalarWave::Worldtube::Tags::ParticlePositionVelocity<3>,
                 Tags::TimeStep>;

  bool operator()(
      const std::array<tnsr::I<double, 3>, 2>& position_and_velocity,
      const TimeDelta& time_step) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  std::vector<double> radii_{};
};
}  // namespace Triggers
