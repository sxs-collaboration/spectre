// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "ApparentHorizons/ObjectLabel.hpp"
#include "ControlSystem/DataVectorHelpers.hpp"
#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Tags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace domain::Tags {
template <size_t Dim>
struct Domain;
struct FunctionsOfTime;
}  // namespace domain::Tags
/// \endcond

namespace control_system {
namespace ControlErrors {
/*!
 * \brief Control error in the 3D
 * \link domain::CoordinateMaps::TimeDependent::Rotation Rotation \endlink
 * coordinate map
 *
 * \details Computes the error in the angle rotated by the system using a
 * slightly modified version of Eq. (41) from \cite Ossokine2013zga. The
 * equation is
 *
 * \f[ \delta\vec{q} = \frac{\vec{C}\times\vec{X}}{\vec{C}\cdot\vec{X}} \f]
 *
 * where \f$\vec{X} = \vec{x}_B - \vec{x}_A\f$ and \f$\vec{C} = \vec{c}_B -
 * \vec{c}_A\f$. Here, object B is located on the positive x-axis and object A
 * is located on the negative x-axis, \f$\vec{X}\f$ is the difference in
 * positions of the centers of the mapped objects, and
 * \f$\vec{C}\f$ is the difference of the centers of the excision spheres, all
 * in the grid frame.
 *
 * Requirements:
 * - This control error requires that there be exactly two objects in the
 *   simulation
 * - Currently both these objects must be black holes
 * - Currently this control system can only be used with the \link
 *   control_system::Systems::Rotation Rotation \endlink control system
 */
struct Rotation : tt::ConformsTo<protocols::ControlError> {
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Computes the control error for rotation control. This should not "
      "take any options."};

  void pup(PUP::er& /*p*/) {}

  template <typename Metavariables, typename... TupleTags>
  DataVector operator()(const Parallel::GlobalCache<Metavariables>& cache,
                        const double /*time*/,
                        const std::string& /*function_of_time_name*/,
                        const tuples::TaggedTuple<TupleTags...>& measurements) {
    const auto& domain = get<domain::Tags::Domain<3>>(cache);

    using center_A = control_system::QueueTags::Center<::ah::ObjectLabel::A>;
    using center_B = control_system::QueueTags::Center<::ah::ObjectLabel::B>;

    const DataVector grid_position_of_A = array_to_datavector(
        domain.excision_spheres().at("ObjectAExcisionSphere").center());
    const DataVector grid_position_of_B = array_to_datavector(
        domain.excision_spheres().at("ObjectBExcisionSphere").center());
    const DataVector& current_position_of_A = get<center_A>(measurements);
    const DataVector& current_position_of_B = get<center_B>(measurements);

    // A is to the left of B in grid frame. To get positive differences,
    // take B - A
    const DataVector grid_diff = grid_position_of_B - grid_position_of_A;
    const DataVector current_diff =
        current_position_of_B - current_position_of_A;

    const double grid_dot_current = dot(grid_diff, current_diff);
    const DataVector grid_cross_current = cross(grid_diff, current_diff);

    return grid_cross_current / grid_dot_current;
  }
};
}  // namespace ControlErrors
}  // namespace control_system
