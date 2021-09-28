// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <functional>
#include <iosfwd>
#include <limits>
#include <string>
#include <type_traits>

#include "Utilities/PrettyType.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace observers {
/// \cond
class ObservationId;
/// \endcond

/// Used as a key in maps to keep track of how many elements have registered.
class ObservationKey {
 public:
  ObservationKey() = default;

  explicit ObservationKey(std::string tag);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  size_t key() const { return key_; }

  const std::string& tag() const;

 private:
  size_t key_{0};
  std::string tag_{};
};

bool operator==(const ObservationKey& lhs, const ObservationKey& rhs);
bool operator!=(const ObservationKey& lhs, const ObservationKey& rhs);

std::ostream& operator<<(std::ostream& os, const ObservationKey& t);

/*!
 * \ingroup ObserversGroup
 * \brief A unique identifier for an observation representing the type of
 * observation and the instance (e.g. time) at which the observation occurs.
 *
 * The ObservationId is used to uniquely identify an observation inside our H5
 * subfiles that can be agreed upon across the system. That is, it does not
 * depend on hardware rounding of floating points because it is an integral. One
 * example would be a `Time` for output observed every `N` steps in a global
 * time stepping evolution. Another example could be having a counter class for
 * dense output observations that increments the counter for each observation
 * but has a value equal to the physical time.
 *
 * The constructor takes a `double` representing the current "time" at which we
 * are observing. For an evolution this could be the physical time, while for an
 * elliptic solve this could be a combination of nonlinear and linear iteration.
 *
 * A specialization of `std::hash` is provided to allow using `ObservationId`
 * as a key in associative containers.
 */
class ObservationId {
 public:
  ObservationId() = default;
  /*!
   * \brief Construct from a value and a string tagging the observation
   *
   * The tag is a unique string used to identify the particular observation.
   */
  ObservationId(double t, std::string tag);

  ObservationId(double t, ObservationKey key);

  /// Hash used to distinguish between ObservationIds of different
  /// types and subtypes. This hash does not contain any information about the
  /// `Id` or its value.
  const ObservationKey& observation_key() const { return observation_key_; }

  /// Hash distinguishing different ObservationIds, including
  /// the value of the `Id`.
  size_t hash() const { return combined_hash_; }

  /// The simulation "time".
  double value() const { return value_; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  ObservationKey observation_key_{};
  size_t combined_hash_{0};
  double value_{std::numeric_limits<double>::signaling_NaN()};
};

bool operator==(const ObservationId& lhs, const ObservationId& rhs);
bool operator!=(const ObservationId& lhs, const ObservationId& rhs);

std::ostream& operator<<(std::ostream& os, const ObservationId& t);
}  // namespace observers

namespace std {
template <>
struct hash<observers::ObservationId> {
  size_t operator()(const observers::ObservationId& t) const {
    return t.hash();
  }
};

template <>
struct hash<observers::ObservationKey> {
  size_t operator()(const observers::ObservationKey& t) const {
    return t.key();
  }
};
}  // namespace std
