// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "ControlSystem/DataVectorHelpers.hpp"
#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Tags/ObjectCenter.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
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
 * where \f$\vec{X} = \vec{x}_A - \vec{x}_B\f$ and \f$\vec{C} = \vec{c}_A -
 * \vec{c}_B\f$. Here, object A is located on the positive x-axis and object B
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
  static constexpr size_t expected_number_of_excisions = 2;

  using object_centers =
      domain::object_list<domain::ObjectLabel::A, domain::ObjectLabel::B>;

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
    using center_A =
        control_system::QueueTags::Center<::domain::ObjectLabel::A>;
    using center_B =
        control_system::QueueTags::Center<::domain::ObjectLabel::B>;

    const tnsr::I<double, 3, Frame::Grid>& grid_position_of_A =
        Parallel::get<domain::Tags::ObjectCenter<domain::ObjectLabel::A>>(
            cache);
    const tnsr::I<double, 3, Frame::Grid>& grid_position_of_B =
        Parallel::get<domain::Tags::ObjectCenter<domain::ObjectLabel::B>>(
            cache);
    const DataVector& current_position_of_A = get<center_A>(measurements);
    const DataVector& current_position_of_B = get<center_B>(measurements);

    // A is to the right of B in grid frame. To get positive differences,
    // take A - B
    const auto grid_diff_tnsr = tenex::evaluate<ti::I>(
        grid_position_of_A(ti::I) - grid_position_of_B(ti::I));
    const DataVector grid_diff{
        {grid_diff_tnsr[0], grid_diff_tnsr[1], grid_diff_tnsr[2]}};
    const DataVector current_diff =
        current_position_of_A - current_position_of_B;

    const double grid_dot_current = dot(grid_diff, current_diff);
    const DataVector grid_cross_current = cross(grid_diff, current_diff);

    return grid_cross_current / grid_dot_current;
  }
};
}  // namespace ControlErrors
}  // namespace control_system
