// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "Domain/Creators/Tags/ObjectCenter.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::Tags {
struct FunctionsOfTime;
}  // namespace domain::Tags
/// \endcond

namespace control_system::ControlErrors {
/*!
 * \brief Tracks the grid-frame object centers.
 *
 * Tracks the 3d grid-frame centers of the two objects. Tracking the
 * grid-frame centers is better than tracking the inertial-frame centers
 * because the timescale at which the grid frame locations change is much
 * larger than the inertial ones. This is because in the grid frame only the
 * radial inspiral really changes because the rotation control system handles
 * the orbital movement. The orbital time scale is much larger than the
 * Cartesian coordinate location timescale, so splitting the system into
 * angular and radial allows for both to be slowly varying.
 *
 * The intended use case of the output of this control system is for binary
 * neutron star simulations where the rotation control needs to be disabled at
 * merger. It is also necessary for changing from harmonic to damped harmonic
 * gauge near when the neutron stars merge. Finally, it can also be used for
 * triggering a grid change from the inspiral grid to the remnant grid in a
 * BNS merger simulation.
 *
 * Requirements:
 * - This control error requires that there be either one or two objects in the
 *   simulation (typically neutron stars).
 * - Currently this control error can only be used with the \link
 *   control_system::Systems::GridCenters GridCenters \endlink control system
 * - There must exist a `PiecewisePolynomial<2>` function of time named
 *   "GridCenters". It is assumed that components `[0,2]` are the grid
 *   coordinates of object A and the components `[3,5]`  are the grid
 *   coordinates of object B.
 */
struct GridCenters : tt::ConformsTo<protocols::ControlError> {
  using object_centers =
      domain::object_list<domain::ObjectLabel::A, domain::ObjectLabel::B>;

  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Computes the control error for the grid centers of two objects. "
      "This should not take any options."};

  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  std::optional<double> get_suggested_timescale() const { return std::nullopt; }

  void reset() {}

  explicit GridCenters(const Options::Context& context = {});

  void pup(PUP::er& p);

  template <typename Metavariables, typename... TupleTags>
  DataVector operator()(const ::TimescaleTuner<true>& /*unused*/,
                        const Parallel::GlobalCache<Metavariables>& cache,
                        const double time,
                        const std::string& function_of_time_name,
                        const tuples::TaggedTuple<TupleTags...>& measurements) {
    using grid_center_A =
        control_system::QueueTags::Center<::domain::ObjectLabel::A,
                                          Frame::Grid>;
    using grid_center_B =
        control_system::QueueTags::Center<::domain::ObjectLabel::B,
                                          Frame::Grid>;

    const auto& measured_grid_position_of_A = get<grid_center_A>(measurements);
    const auto& measured_grid_position_of_B = get<grid_center_B>(measurements);

    return impl(get<domain::Tags::FunctionsOfTime>(cache)
                    .at(function_of_time_name)
                    ->func(time)[0],
                measured_grid_position_of_A, measured_grid_position_of_B);
  }

 private:
  static DataVector impl(const DataVector& fot_positions_dv,
                         const DataVector& measured_grid_position_of_A,
                         const DataVector& measured_grid_position_of_B);
};
}  // namespace control_system::ControlErrors
