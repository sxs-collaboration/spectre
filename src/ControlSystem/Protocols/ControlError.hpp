// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace control_system::protocols {
/// \brief Definition of a control error
///
/// A control error is used within a control system to compute how far off the
/// the value you are controlling is from its expected value.
///
/// A conforming type must specify:
///
/// - a call operator that returns a DataVector with a signature the same as in
///   the example shown here:
/// - a `static constexpr size_t expected_number_of_excisions` which specifies
///   the number of excisions necessary in order to compute the control error.
/// - a type alias `object_centers` to a `domain::object_list` of
///   `domain::ObjectLabel`s. These are the objects that will require the
///   `domain::Tags::ObjectCenter`s tags to be in the GlobalCache for this
///   control system to work.
///
///
///   \snippet Helpers/ControlSystem/Examples.hpp ControlError
struct ControlError {
  template <typename ConformingType>
  struct test {
    struct DummyMetavariables;
    struct DummyTupleTags;

    static constexpr size_t expected_number_of_excisions =
        ConformingType::expected_number_of_excisions;

    using object_centers = typename ConformingType::object_centers;

    static_assert(
        std::is_same_v<
            DataVector,
            decltype(ConformingType{}(
                std::declval<
                    const Parallel::GlobalCache<DummyMetavariables>&>(),
                std::declval<const double>(),
                std::declval<const std::string&>(),
                std::declval<const tuples::TaggedTuple<DummyTupleTags>&>()))>);
  };
};
}  // namespace control_system::protocols
