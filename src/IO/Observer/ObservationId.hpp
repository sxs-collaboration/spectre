// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <functional>
#include <iosfwd>
#include <string>

#include "Utilities/PrettyType.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace observers {
/*!
 * \ingroup ObserversGroup
 * \brief A type-erased identifier that combines the identifier's type
 * and hash used to uniquely identify an observation inside of a
 * `h5::File::SUB_FILE`.
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
   * \brief Construct from a value and an `ObservationType`.
   *
   * `ObservationType` is used to distinguish different classes that
   * do observing, so that messages to ObserverWriter do not collide.
   * Any class that uses the Observer or ObserverWriter infrastructure
   * should pass itself (or some other type that uniquely identifies
   * who is doing the observing) as the ObservationType when creating
   * `ObservationId`s.  If a single class is responsible for different
   * kinds of observation that can be distinguished only at runtime,
   * these different kinds of observation should be denoted by passing
   * the optional `observation_subtype` string.
   */
  template <typename ObservationType>
  explicit ObservationId(double t, const ObservationType& meta,
                         const std::string& observation_subtype = "") noexcept;

  /// Hash used to distinguish between ObservationIds of different
  /// types and subtypes. This hash does not contain any information about the
  /// `Id` or its value.
  size_t observation_type_hash() const noexcept {
    return observation_type_hash_;
  }

  /// Hash distinguishing different ObservationIds, including
  /// the value of the `Id`.
  size_t hash() const noexcept { return combined_hash_; }

  double value() const noexcept { return value_; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

 private:
  size_t observation_type_hash_;
  size_t combined_hash_;
  double value_;
};

template <typename ObservationType>
ObservationId::ObservationId(const double t, const ObservationType& /*meta*/,
                             const std::string& observation_subtype) noexcept
    : observation_type_hash_(std::hash<std::string>{}(
          pretty_type::get_name<ObservationType>() + observation_subtype)),
      combined_hash_([&t](size_t type_hash) {
        size_t combined = type_hash;
        boost::hash_combine(combined, t);
        return combined;
      }(observation_type_hash_)),
      value_(t) {}

bool operator==(const ObservationId& lhs, const ObservationId& rhs) noexcept;
bool operator!=(const ObservationId& lhs, const ObservationId& rhs) noexcept;

std::ostream& operator<<(std::ostream& os, const ObservationId& t) noexcept;
}  // namespace observers

namespace std {
template <>
struct hash<observers::ObservationId> {
  size_t operator()(const observers::ObservationId& t) const noexcept {
    return t.hash();
  }
};
}  // namespace std
