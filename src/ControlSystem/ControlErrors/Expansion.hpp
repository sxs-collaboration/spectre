// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "ControlSystem/Measurements/BothHorizons.hpp"
#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
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
 * \link domain::CoordinateMaps::TimeDependent::CubicScale CubicScale \endlink
 * coordinate map
 *
 * \details Computes the error in the map parameter \f$a(t)\f$ using Eq. (40)
 * from \cite Ossokine2013zga (see \link
 * domain::CoordinateMaps::TimeDependent::CubicScale CubicScale \endlink for a
 * definition of \f$a(t)\f$). The equation is
 *
 * \f{align}{
 *   \delta a &= a\left( \frac{\vec{X}\cdot\vec{C}}{||\vec{C}||^2} - 1 \right)
 * \\ \delta a &= a\left( \frac{X_0}{C_0} - 1 \right) \f}
 *
 * where \f$\vec{X} = \vec{x}_A - \vec{x}_B\f$ and \f$\vec{C} = \vec{c}_A -
 * \vec{c}_B\f$. Here, object A is located on the positive x-axis and object B
 * is located on the negative x-axis, \f$\vec{X}\f$ is the difference in
 * positions of the centers of the mapped objects, and
 * \f$\vec{C}\f$ is the difference of the centers of the excision spheres, all
 * in the grid frame. It is assumed that the positions of the excision spheres
 * are exactly along the x-axis, which is why we were able to make the
 * simplification in the second line above.
 *
 * Requirements:
 * - This control error requires that there be exactly two objects in the
 *   simulation
 * - Currently both these objects must be black holes
 * - Currently this control system can only be used with the \link
 *   control_system::Systems::Expansion Expansion \endlink control system
 */
struct Expansion : tt::ConformsTo<protocols::ControlError> {
  static constexpr size_t expected_number_of_excisions = 2;

  using object_centers =
      domain::object_list<domain::ObjectLabel::A, domain::ObjectLabel::B>;

  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Computes the control error for expansion control. This should not "
      "take any options."};

  void pup(PUP::er& /*p*/) {}

  template <typename Metavariables, typename... TupleTags>
  DataVector operator()(const Parallel::GlobalCache<Metavariables>& cache,
                        const double time,
                        const std::string& function_of_time_name,
                        const tuples::TaggedTuple<TupleTags...>& measurements) {
    const auto& functions_of_time = get<domain::Tags::FunctionsOfTime>(cache);

    const double current_expansion_factor =
        functions_of_time.at(function_of_time_name)->func(time)[0][0];

    using center_A =
        control_system::QueueTags::Center<::domain::ObjectLabel::A>;
    using center_B =
        control_system::QueueTags::Center<::domain::ObjectLabel::B>;

    const double grid_position_of_A =
        Parallel::get<domain::Tags::ObjectCenter<domain::ObjectLabel::A>>(
            cache)[0];
    const double grid_position_of_B =
        Parallel::get<domain::Tags::ObjectCenter<domain::ObjectLabel::B>>(
            cache)[0];
    const double current_position_of_A = get<center_A>(measurements)[0];
    const double current_position_of_B = get<center_B>(measurements)[0];

    // A is to the right of B in grid frame. To get positive differences,
    // take A - B
    const double expected_expansion_factor =
        current_expansion_factor *
        (current_position_of_A - current_position_of_B) /
        (grid_position_of_A - grid_position_of_B);

    return {expected_expansion_factor - current_expansion_factor};
  }
};
}  // namespace ControlErrors
}  // namespace control_system
