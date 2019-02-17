// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <limits>
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
 * \brief CubicScale map defined by \f$x = a(t)*\xi+(b(t)-a(t))\xi^3/X^2\f$,
 *  where \f$X\f$ is the outer boundary.
 *
 * The map scales the coordinates \f$\xi\f$ near the center by a factor
 * \f$a(t)\f$, while the coordinates near the outer boundary \f$X\f$, are scaled
 * by a factor \f$b(t)\f$. Here \f$a(t)\f$ and \f$b(t)\f$ are FunctionsOfTime.
 *
 * Currently, only a 1-D implementation.
 */
class CubicScale {
 public:
  static constexpr size_t dim = 1;

  explicit CubicScale(double outer_boundary) noexcept;
  CubicScale() = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 1> operator()(
      const std::array<T, 1>& source_coords, double time,
      const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
      noexcept;

  /// Returns boost::none if the point is outside the range of the map.
  template <typename T>
  boost::optional<std::array<tt::remove_cvref_wrap_t<T>, 1>> inverse(
      const std::array<T, 1>& target_coords, double time,
      const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
      noexcept;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 1> frame_velocity(
      const std::array<T, 1>& source_coords, double time,
      const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
      noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> inv_jacobian(
      const std::array<T, 1>& source_coords, double time,
      const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
      noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> jacobian(
      const std::array<T, 1>& source_coords, double time,
      const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
      noexcept;

  template <typename T>
  tnsr::Iaa<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> hessian(
      const std::array<T, 1>& source_coords, double time,
      const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
      noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT

  bool is_identity() const noexcept { return false; }

 private:
  friend bool operator==(const CubicScale& lhs, const CubicScale& rhs) noexcept;

  std::string f_of_t_a_ = "expansion_a";
  std::string f_of_t_b_ = "expansion_b";
  double outer_boundary_{std::numeric_limits<double>::signaling_NaN()};
};

inline bool operator!=(const CoordMapsTimeDependent::CubicScale& lhs,
                       const CoordMapsTimeDependent::CubicScale& rhs) noexcept {
  return not(lhs == rhs);
}

}  // namespace CoordMapsTimeDependent
}  // namespace domain
