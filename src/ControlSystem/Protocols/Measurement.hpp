// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "ControlSystem/Protocols/Submeasurement.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system::protocols {
/*!
 * \brief Definition of a measurement for the control systems
 *
 * A class conforming to this protocol is referenced from each control
 * system to define measurements to be made.  Multiple control systems
 * can share the same measurement.
 *
 * A conforming class must provide a `submeasurements` type alias to a
 * `tmpl::list` of structs conforming to the Submeasurement protocol.
 *
 * Here's an example for a class conforming to this protocol:
 *
 * \snippet Helpers/ControlSystem/Examples.hpp Measurement
 */
struct Measurement {
  template <typename ConformingType>
  struct test {
    using submeasurements = typename ConformingType::submeasurements;

    template <typename T>
    using assert_conforms_to_t =
        std::bool_constant<tt::assert_conforms_to<T, Submeasurement>>;
    static_assert(tmpl::all<submeasurements,
                            tmpl::bind<assert_conforms_to_t, tmpl::_1>>::value);
  };
};
}  // namespace control_system::protocols
