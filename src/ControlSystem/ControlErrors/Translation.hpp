// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/math/quaternion.hpp>
#include <cstddef>
#include <pup.h>

#include "ControlSystem/ControlErrors/Expansion.hpp"
#include "ControlSystem/ControlErrors/Rotation.hpp"
#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/CoordinateMaps/TimeDependent/RotationMatrixHelpers.hpp"
#include "Domain/Creators/Tags/ObjectCenter.hpp"
#include "Domain/FunctionsOfTime/QuaternionHelpers.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
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
 * \brief Control error in the 3D \link
 * domain::CoordinateMaps::TimeDependent::Translation Translation \endlink
 * coordinate map
 *
 * \details Computes the error in how much the system has translated. When there
 * are two excisions, it does this by using a modified version of Eq. (42) from
 * \cite Ossokine2013zga. The equation is
 *
 * \begin{equation}
 * \left(0, \delta\vec{T}\right) = a\mathbf{q}\left(\frac{1}{2}(\mathbf{x}_A
 * + \mathbf{x}_B - \frac{1}{2}(\mathbf{c}_A + \mathbf{c}_B)) - \mathbf{\delta
 * q}\wedge\frac{1}{2}(\mathbf{c}_A + \mathbf{c}_B) - \frac{\delta
 * a}{a}\frac{1}{2}(\mathbf{c}_A + \mathbf{c}_B) \right)\mathbf{q}^*
 * \end{equation}
 *
 * where object A is located on the positive x-axis in the grid frame, bold face
 * letters are quaternions, vectors are promoted to quaternions as \f$
 * \mathbf{v} = (0, \vec{v}) \f$, \f$ \mathbf{q} \f$ is the quaternion from the
 * \link domain::CoordinateMaps::TimeDependent::Rotation Rotation \endlink map,
 * \f$ a \f$ is the function \f$ a(t) \f$ from the \link
 * domain::CoordinateMaps::TimeDependent::CubicScale CubicScale \endlink map,
 * \f$ \mathbf{\delta q}\wedge\mathbf{c}_A \equiv (0, \delta\vec{q} \times
 * \vec{c}_A) \f$, \f$ \delta\vec{q} \f$ is the \link
 * control_system::ControlErrors::Rotation Rotation \endlink control error, and
 * \f$ \delta a\f$ is the \link control_system::ControlErrors::Expansion
 * Expansion \endlink control error.
 *
 * When there is only a single excision, the control error assumes that the
 * center of the excision is at the origin. Thus, the control error is just
 * taken to be the current center of the horizon mapped through the expansion
 * and rotation maps if there are any.
 *
 * Requirements:
 * - This control error requires that there be either one or two objects in the
 *   simulation
 * - Currently this control error can only be used with the \link
 *   control_system::Systems::Translation Translation \endlink control system
 * - There must exist an expansion map and a quaternion rotation map in the
 *   coordinate map with names "Expansion" and "Rotation", respectively if there
 *   are two object. If there is a single object, then the "Expansion" and
 *   "Rotation" maps may exist, but don't need to.
 */
template <size_t NumberOfObjects>
struct Translation : tt::ConformsTo<protocols::ControlError> {
  static_assert(NumberOfObjects == 1 or NumberOfObjects == 2,
                "Translation control can only work with 1 or 2 objects.");

  using object_centers = tmpl::conditional_t<
      NumberOfObjects == 1, domain::object_list<domain::ObjectLabel::None>,
      domain::object_list<domain::ObjectLabel::A, domain::ObjectLabel::B>>;

  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Computes the control error for translation control. This should not "
      "take any options."};

  void pup(PUP::er& /*p*/) {}

  template <typename Metavariables, typename... TupleTags>
  DataVector operator()(const ::TimescaleTuner<true>& /*unused*/,
                        const Parallel::GlobalCache<Metavariables>& cache,
                        const double time,
                        const std::string& /*function_of_time_name*/,
                        const tuples::TaggedTuple<TupleTags...>& measurements) {
    const auto& functions_of_time = get<domain::Tags::FunctionsOfTime>(cache);

    if constexpr (NumberOfObjects == 2) {
      using quat = boost::math::quaternion<double>;

      const quat quaternion = datavector_to_quaternion(
          functions_of_time.at("Rotation")->func(time)[0]);
      const double expansion_factor =
          functions_of_time.at("Expansion")->func(time)[0][0];

      using center_A =
          control_system::QueueTags::Center<::domain::ObjectLabel::A>;
      using center_B =
          control_system::QueueTags::Center<::domain::ObjectLabel::B>;

      const tnsr::I<double, 3, Frame::Grid>& grid_position_of_A_tnsr =
          Parallel::get<domain::Tags::ObjectCenter<domain::ObjectLabel::A>>(
              cache);
      const DataVector grid_position_of_A{{grid_position_of_A_tnsr[0],
                                           grid_position_of_A_tnsr[1],
                                           grid_position_of_A_tnsr[2]}};
      const DataVector& current_position_of_A = get<center_A>(measurements);

      const tnsr::I<double, 3, Frame::Grid>& grid_position_of_B_tnsr =
          Parallel::get<domain::Tags::ObjectCenter<domain::ObjectLabel::B>>(
              cache);
      const DataVector grid_position_of_B{{grid_position_of_B_tnsr[0],
                                           grid_position_of_B_tnsr[1],
                                           grid_position_of_B_tnsr[2]}};
      const DataVector& current_position_of_B = get<center_B>(measurements);

      const DataVector grid_position_average =
          0.5 * (grid_position_of_A + grid_position_of_B);
      const DataVector current_position_average =
          0.5 * (current_position_of_A + current_position_of_B);

      const DataVector grid_separation =
          grid_position_of_A - grid_position_of_B;
      const DataVector current_separation =
          current_position_of_A - current_position_of_B;

      // These quantities come from the translation control implementation in
      // SpEC
      const double current_separation_dot_grid_separation =
          dot(current_separation, grid_separation);
      const double current_separation_dot_grid_average =
          dot(current_separation, grid_position_average);
      const double grid_separation_dot_grid_average =
          dot(grid_separation, grid_position_average);
      const double grid_separation_dot_grid_separation =
          dot(grid_separation, grid_separation);

      // From eq. 42 in 1304.3067 where the grid and current position are
      // swapped from only object A to the average grid and current positions of
      // both objects.
      const DataVector translation_control =
          expansion_factor *
          (grid_separation_dot_grid_separation * current_position_average -
           current_separation_dot_grid_separation * grid_position_average -
           grid_separation_dot_grid_average * current_separation +
           current_separation_dot_grid_average * grid_separation) /
          grid_separation_dot_grid_separation;
      const quat middle_expression =
          datavector_to_quaternion(translation_control);

      // Because we are converting from a quaternion to a DataVector, there will
      // be four components in the DataVector. However, translation control only
      // requires three (the latter three to be exact, because the first
      // component should be 0. We ASSERT this also.)
      const DataVector result_with_four_components = quaternion_to_datavector(
          quaternion * middle_expression * conj(quaternion));
      ASSERT(equal_within_roundoff(result_with_four_components[0], 0.0),
             "Error in computing translation control error. First component of "
             "resulting quaternion should be 0.0, but is " +
                 get_output(result_with_four_components[0]) + " instead.");

      return {result_with_four_components[1], result_with_four_components[2],
              result_with_four_components[3]};
    } else {
      const DataVector& current_position =
          get<control_system::QueueTags::Center<::domain::ObjectLabel::None>>(
              measurements);
      DataVector result{3, 0.0};

      double expansion_factor = 1.0;
      if (functions_of_time.count("Expansion") == 1) {
        expansion_factor = functions_of_time.at("Expansion")->func(time)[0][0];
      }

      if (functions_of_time.count("Rotation") == 1) {
        const Matrix rot_matrix =
            rotation_matrix<3>(time, *functions_of_time.at("Rotation").get());

        for (size_t i = 0; i < 3; i++) {
          for (size_t j = 0; j < 3; j++) {
            result[i] += rot_matrix(i, j) * current_position[j];
          }
        }
      } else {
        result = current_position;
      }

      result *= expansion_factor;

      return result;
    }
  }
};
}  // namespace ControlErrors
}  // namespace control_system
