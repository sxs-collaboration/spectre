// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

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
}  // namespace observers
