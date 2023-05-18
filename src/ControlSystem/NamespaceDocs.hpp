// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \ingroup ControlSystemGroup
/// Control systems and related functionality
namespace control_system {
/// \ingroup ControlSystemGroup
/// All Actions related to the control system
namespace Actions {}

/*!
 * \ingroup ControlSystemGroup
 * \brief All control errors that will be used in control systems.
 *
 * \details A control error is a struct that conforms to the
 * control_system::protocols::ControlError protocol. Control errors compute the
 * error between current map parameters and what they are expected to be. See
 * an example of a control error here:
 * \snippet Helpers/ControlSystem/Examples.hpp ControlError
 */
namespace ControlErrors {}

/*!
 * \ingroup ControlSystemGroup
 * \brief All control systems.
 *
 * \details A control system is a struct that conforms to the
 * control_system::protocols::ControlSystem protocol. They are used to control
 * the time dependent coordinate maps in an evolution. See an example of a
 * control system here:
 * \snippet Helpers/ControlSystem/Examples.hpp ControlSystem
 */
namespace Systems {}

/// \ingroup ControlSystemGroup
/// Classes and functions used in implementation of size control
namespace size {}

}  // namespace control_system
