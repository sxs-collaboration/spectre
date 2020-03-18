// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace domain {
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain {
namespace CoordMapsTimeDependent {

/// \cond HIDDEN_SYMBOLS
template <size_t Dim>
class Rotation;
/// \endcond

/*!
 * \ingroup CoordMapsTimeDependentGroup
 * \brief Time-dependent spatial rotation in two dimensions.
 *
 * Let \f$(R,\Phi)\f$ be the polar coordinates associated with
 * \f$(\xi,\eta)\f$, where \f$\xi\f$ and \f$\eta\f$ are the unmapped
 * coordiantes. Let \f$(r,\phi)\f$ be the polar coordinates associated with
 * \f$(x,y)\f$, where \f$x\f$ and \f$y\f$ are the mapped coordinates.
 * This map applies the spatial rotation \f$\phi = \Phi + \alpha(t)\f$.
 *
 * The formula for the mapping is:
 *\f{eqnarray*}
  x &=& \xi \cos \alpha(t) - \eta \sin \alpha(t), \\
  y &=& \xi \sin \alpha(t) + \eta \cos \alpha(t).
  \f}
 *
 * \note Currently, only a rotation in two-dimensional space is implemented
 * here. In the future, this class should be extended to also support
 * three-dimensional rotations using quaternions.
 */
template <>
class Rotation<2> {
 public:
  static constexpr size_t dim = 2;

  explicit Rotation(std::string function_of_time_name) noexcept;
  Rotation() = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 2> operator()(
      const std::array<T, 2>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  boost::optional<std::array<double, 2>> inverse(
      const std::array<double, 2>& target_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 2> frame_velocity(
      const std::array<T, 2>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> jacobian(
      const std::array<T, 2>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> inv_jacobian(
      const std::array<T, 2>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const noexcept;

  void pup(PUP::er& p) noexcept;  // NOLINT

  bool is_identity() const noexcept { return false; }

 private:
  friend bool operator==(const Rotation<2>& lhs,
                         const Rotation<2>& rhs) noexcept;
  std::string f_of_t_name_;
};

bool operator!=(const Rotation<2>& lhs, const Rotation<2>& rhs) noexcept;

}  // namespace CoordMapsTimeDependent
}  // namespace domain
