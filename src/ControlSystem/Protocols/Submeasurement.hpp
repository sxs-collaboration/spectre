// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TMPL.hpp"

namespace control_system::protocols {
/*!
 * \brief Definition of a portion of a measurement for the control
 * systems
 *
 * These structs are referenced from structs conforming to the
 * Measurement protocol.  They define independent parts of a control
 * system measurement, such as individual horizon-finds in a
 * two-horizon measurement.
 *
 * A conforming struct must provide
 *
 * - An `interpolation_target_tag` type alias templated on the \ref
 *   ControlSystem "control systems" using this submeasurement.  (This template
 *   parameter must be used in the call to `RunCallbacks` discussed below.) This
 *   alias may be `void` if the submeasurement does not use an interpolation
 *   target tag.  This is only used to collect the tags that must be registered
 *   in the metavariables.
 * - An `event` type alias also templated on the
 *   \ref ControlSystem "control systems" using this submeasurement which is an
 *   `::Event`. It is templated on the control systems because the event usually
 *   takes the `interpolation_target_tag` as a template parameter. Currently,
 *   this event must be fully functional when it is default constructed. It will
 *   not be constructed with any arguments.
 *
 * The `event` will be run on every element, and they must collectively
 * result in a single call on one chare (which need not be one of the element
 * chares) to `control_system::RunCallbacks<ConformingStructBeingDefinedHere,
 * ControlSystems>::apply`. This will almost always require performing a
 * reduction. The `ControlSystems` template parameter passed to `RunCallbacks`
 * here must be the same type that was passed to the `interpolation_target_tag`
 * and `event` type aliases. The `ControlSystems` template parameter will be
 * a list of all control systems that use the same Submeasurement.
 *
 * Here's an example for a class conforming to this protocol:
 *
 * \snippet Helpers/ControlSystem/Examples.hpp Submeasurement
 */
struct Submeasurement {
  template <typename ConformingType>
  struct test {
    struct DummyControlSystem;

    using interpolation_target_tag =
        typename ConformingType::template interpolation_target_tag<
            tmpl::list<DummyControlSystem>>;

    using event =
        typename ConformingType::template event<tmpl::list<DummyControlSystem>>;
  };
};
}  // namespace control_system::protocols
