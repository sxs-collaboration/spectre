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
 * The identifier must have a `value()` method that returns a double
 * representing the current "time" at which we are observing. For an evolution
 * this could be the physical time, while for an elliptic solve this could be a
 * combination of nonlinear and linear iteration.
 *
 * A specialization of `std::hash` is provided to allow using `ObservationId`
 * as a key in associative containers.
 */
class ObservationId {
 public:
  ObservationId() = default;

  /*!
   * \brief Construct from an ID of type `Id`
   */
  template <typename Id>
  explicit ObservationId(const Id& t) noexcept;

  size_t hash() const noexcept { return combined_hash_; }

  double value() const noexcept { return value_; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

 private:
  size_t combined_hash_;
  double value_;
};

template <typename Id>
ObservationId::ObservationId(const Id& t) noexcept
    : combined_hash_([&t]() {
        size_t combined = std::hash<std::string>{}(pretty_type::get_name<Id>());
        boost::hash_combine(combined, t);
        return combined;
      }()),
      value_(t.value()) {}

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
