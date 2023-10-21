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

template <bool AllowDecrease>
struct TimescaleTuner;

namespace control_system::protocols {
namespace detail {

struct DummyMetavariables {
  using component_list = tmpl::list<>;
};
struct DummyTupleTags {
  using type = int;
};

template <typename T, bool AllowDecrease>
struct has_signature
    : std::is_invocable_r<DataVector, T, const ::TimescaleTuner<AllowDecrease>&,
                          const Parallel::GlobalCache<DummyMetavariables>&,
                          const double, const std::string&,
                          const tuples::TaggedTuple<DummyTupleTags>&> {};
}  // namespace detail
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
/// \note The TimescaleTuner can have it's template parameter be either `true`
/// or `false`.
///
///   \snippet Helpers/ControlSystem/Examples.hpp ControlError
struct ControlError {
  template <typename ConformingType>
  struct test {
    static constexpr size_t expected_number_of_excisions =
        ConformingType::expected_number_of_excisions;

    using object_centers = typename ConformingType::object_centers;

    static_assert(detail::has_signature<ConformingType, true>::value or
                  detail::has_signature<ConformingType, false>::value);
  };
};
}  // namespace control_system::protocols
