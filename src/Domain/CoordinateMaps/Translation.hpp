// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
class FunctionOfTime;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain {
namespace CoordMapsTimeDependent {
/*!
 * \ingroup CoordMapsTimeDependentGroup
 * \brief Translation map defined by \f$x = \xi+T(t)\f$.
 *
 * The map adds a translation, \f$T(t)\f$, to the coordinates \f$\xi\f$,
 * where \f$T(t)\f$ is a FunctionOfTime.
 */
class Translation {
 public:
  static constexpr size_t dim = 1;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 1> operator()(
      const std::array<T, 1>& source_coords, double time,
      const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
      noexcept;

  boost::optional<std::array<double, 1>> inverse(
      const std::array<double, 1>& target_coords, double time,
      const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
      noexcept;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 1> frame_velocity(
      const std::array<T, 1>& source_coords, double time,
      const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
      noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> inv_jacobian(
      const std::array<T, 1>& source_coords) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> jacobian(
      const std::array<T, 1>& source_coords) const noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

  bool is_identity() const noexcept { return false; }

 private:
  friend bool operator==(const Translation& lhs,
                         const Translation& rhs) noexcept;

  std::string f_of_t_name_ = "trans";
};

inline bool operator!=(
    const CoordMapsTimeDependent::Translation& lhs,
    const CoordMapsTimeDependent::Translation& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace CoordMapsTimeDependent
}  // namespace domain
