// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

#include "Utilities/TypeTraits.hpp"

namespace observers {
/*!
 * \ingroup ObserversGroup
 * \brief Specifies the type of observation.
 *
 * Below `sender` is the component passing the data, reduction or volume, to the
 * observer component.
 */
enum class TypeOfObservation {
  /// The sender will only perform reduction observations
  Reduction,
  /// The sender will only perform volume observations
  Volume,
  /// The sender will perform both reduction and volume observations
  ReductionAndVolume
};

std::ostream& operator<<(std::ostream& os, const TypeOfObservation& t) noexcept;

// @{
/// Inherits off of `std::true_type` if `T` has a member variable
/// `RegisterWithObserver`
template <class T, class = cpp17::void_t<>>
struct has_register_with_observer : std::false_type {};

/// \cond
template <class T>
struct has_register_with_observer<
    T, cpp17::void_t<decltype(T::RegisterWithObserver)>> : std::true_type {};
/// \endcond

template <class T>
constexpr bool has_register_with_observer_v =
    has_register_with_observer<T>::value;
// @}
}  // namespace observers
