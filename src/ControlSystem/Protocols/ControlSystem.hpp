// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <type_traits>

#include "ControlSystem/Protocols/Measurement.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system::protocols {
/// \brief Definition of a control system
///
/// Defines a control system for controlling a FunctionOfTime.
///
/// A conforming class must provide the following functionality:
///
/// - a static function `name` returning a `std::string`.  This
///   corresponds to the name of the FunctionOfTime controlled by this
///   system.
///
/// - a type alias `measurement` to a struct implementing the
///   Measurement protocol.
///
/// - a member struct (or type alias) `process_measurement`, defining
///   the following:
///
///   - a templated type alias `argument_tags`, which must produce a
///     `tmpl::list` of tags when instantiated for any submeasurement
///     of the control system's `measurement`.
///
///   - a static function `apply` that accepts as arguments:
///
///     - any submeasurement of the control system's `measurement`.
///       For measurements with multiple submeasurements, this can be
///       accomplished either by a template parameter or by
///       overloading the function.
///
///     - the values corresponding to `argument_tags` instantiated with the
///       type of the first argument.
///
///     - the global cache `Parallel::GlobalCache<Metavariables>&`
///
///     - a `const LinkedMessageId<double>&` identifying the
///       measurement, with the `id` field being the measurement time.
///
///   The `apply` function will be called once for each submeasurement
///   of the control system's `measurement` using data from the
///   `DataBox` passed to `RunCallbacks`.  It should communicate any
///   necessary data to the control system singleton.
///
/// Here's an example for a class conforming to this protocol:
///
/// \snippet Helpers/ControlSystem/Examples.hpp ControlSystem
struct ControlSystem {
  template <typename ConformingType>
  struct test {
    static_assert(std::is_same_v<std::decay_t<decltype(ConformingType::name())>,
                                 std::string>);

    using measurement = typename ConformingType::measurement;
    static_assert(tt::assert_conforms_to<measurement, Measurement>);

    template <typename Submeasurement>
    struct check_process_measurement_argument_tags {
      using type =
          typename ConformingType::process_measurement::template argument_tags<
              Submeasurement>;
    };

    using process_measurement_argument_tags =
        tmpl::transform<typename measurement::submeasurements,
                        check_process_measurement_argument_tags<tmpl::_1>>;

    // We can't check the apply operator, because the tags may be
    // incomplete types, so we don't know what the argument types are.
  };
};
}  // namespace control_system::protocols
