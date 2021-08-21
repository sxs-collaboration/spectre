// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TMPL.hpp"

namespace control_system::protocols {
/// \brief Definition of a portion of a measurement for the control
/// systems
///
/// These structs are referenced from structs conforming to the
/// Measurement protocol.  They define independent parts of a control
/// system measurement, such as individual horizon-finds in a
/// two-horizon measurement.
///
/// A conforming struct must provide an `interpolation_target_tag`
/// type alias templated on the \ref ControlSystem "control systems"
/// using this submeasurement.  (This template parameter must be used
/// in the call to `RunCallbacks` discussed below.)  This alias may be
/// `void` if the submeasurement does not use an interpolation target
/// tag.  This is only used to collect the tags that must be
/// registered in the metavariables.
///
/// A struct must also define an `argument_tags` type list and an
/// `apply` function taking the corresponding arguments followed by
///
/// - `const LinkedMessageId<double>& measurement_id`
/// - `Parallel::GlobalCache<Metavariables>& cache`
/// - `const /*component-defined*/& array_index`
/// - `const ParallelComponent* /*meta*/`
/// - `ControlSystems /*meta*/`
///
/// where the last two are generally template parameters to the
/// function.  This function will be called on every element, and
/// these calls must collectively result in a single call on one chare
/// (which need not be one of the element chares) to
/// `control_system::RunCallbacks<ConformingStructBeingDefinedHere,
/// ControlSystems>::apply`.  This will almost always require
/// performing a reduction.
///
/// The time the measurement occurs as is available as
/// `measurement_id.id`.
///
/// Here's an example for a class conforming to this protocol:
///
/// \snippet Helpers/ControlSystem/Examples.hpp Submeasurement
struct Submeasurement {
  template <typename ConformingType>
  struct test {
    struct DummyControlSystem;

    using interpolation_target_tag =
        typename ConformingType::template interpolation_target_tag<
            tmpl::list<DummyControlSystem>>;

    using argument_tags = typename ConformingType::argument_tags;

    // We can't check the apply operator, because the tags may be
    // incomplete types, so we don't know what the argument types are.
  };
};
}  // namespace control_system::protocols
