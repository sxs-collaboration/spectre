// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class Rotation.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TypeTraits/RemoveReferenceWrapper.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace domain {
namespace CoordinateMaps {

/// \cond HIDDEN_SYMBOLS
template <size_t Dim>
class Rotation;
/// \endcond

/*!
 * \ingroup CoordinateMapsGroup
 * \brief Spatial rotation in two dimensions.
 *
 * Let \f$(R,\Phi)\f$ be the polar coordinates associated with
 * \f$(\xi,\eta)\f$.
 * Let \f$(r,\phi)\f$ be the polar coordinates associated with \f$(x,y)\f$.
 * Applies the spatial rotation \f$\phi = \Phi + \alpha\f$.
 *
 * The formula for the mapping is:
 *\f{eqnarray*}
  x &=& \xi \cos \alpha - \eta \sin \alpha \\
  y &=& \xi \sin \alpha + \eta \cos \alpha
  \f}.
 */
template <>
class Rotation<2> {
 public:
  static constexpr size_t dim = 2;

  /// Constructor.
  ///
  /// \param rotation_angle the angle \f$\alpha\f$ (in radians).
  explicit Rotation(double rotation_angle);
  Rotation() = default;
  ~Rotation() = default;
  Rotation(const Rotation&) = default;
  Rotation& operator=(const Rotation&) = default;
  Rotation(Rotation&&) = default;  // NOLINT
  Rotation& operator=(Rotation&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 2> operator()(
      const std::array<T, 2>& source_coords) const;

  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  std::optional<std::array<double, 2>> inverse(
      const std::array<double, 2>& target_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> jacobian(
      const std::array<T, 2>& source_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 2, Frame::NoFrame> inv_jacobian(
      const std::array<T, 2>& source_coords) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  bool is_identity() const { return is_identity_; }

 private:
  friend bool operator==(const Rotation<2>& lhs, const Rotation<2>& rhs);

  double rotation_angle_{std::numeric_limits<double>::signaling_NaN()};
  tnsr::ij<double, 2, Frame::Grid> rotation_matrix_{
      std::numeric_limits<double>::signaling_NaN()};
  bool is_identity_{false};
};

bool operator!=(const Rotation<2>& lhs, const Rotation<2>& rhs);

/*!
 * \ingroup CoordinateMapsGroup
 * \brief Spatial rotation in three dimensions using Euler angles
 *
 * Rotation angles should be specified in degrees.
 * First rotation \f$\alpha\f$ is about z axis.
 * Second rotation \f$\beta\f$ is about rotated y axis.
 * Third rotation \f$\gamma\f$ is about rotated z axis.
 * These rotations are of the \f$(\xi,\eta,\zeta)\f$ coordinate system with
 * respect
 * to the grid coordinates \f$(x,y,z)\f$.
 *
 * The formula for the mapping is:
 * \f{eqnarray*}
 *   x &=& \xi (\cos\gamma \cos\beta \cos\alpha - \sin\gamma \sin\alpha)
 * + \eta (-\sin\gamma \cos\beta \cos\alpha - \cos\gamma \sin\alpha)
 * + \zeta \sin\beta \cos\alpha \\
 * y &=& \xi (\cos\gamma \cos\beta \sin\alpha + \sin\gamma \cos\alpha)
 * + \eta (-\sin\gamma \cos\beta \sin\alpha + \cos\gamma \cos\alpha)
 * + \zeta \sin\beta \sin\alpha \\
 *        z &=& -\xi \cos\gamma \sin\beta + \eta \sin\gamma \sin\beta
 *        +  \zeta \cos\beta
 *  \f}
 */
template <>
class Rotation<3> {
 public:
  static constexpr size_t dim = 3;

  /// Constructor.
  ///
  /// \param rotation_about_z the angle \f$\alpha\f$ (in radians).
  /// \param rotation_about_rotated_y the angle \f$\beta\f$ (in radians).
  /// \param rotation_about_rotated_z the angle \f$\gamma\f$ (in radians).
  Rotation(double rotation_about_z, double rotation_about_rotated_y,
           double rotation_about_rotated_z);
  Rotation() = default;
  ~Rotation() = default;
  Rotation(const Rotation&) = default;
  Rotation& operator=(const Rotation&) = default;
  Rotation(Rotation&&) = default;  // NOLINT
  Rotation& operator=(Rotation&&) = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, 3> operator()(
      const std::array<T, 3>& source_coords) const;

  std::optional<std::array<double, 3>> inverse(
      const std::array<double, 3>& target_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> jacobian(
      const std::array<T, 3>& source_coords) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 3, Frame::NoFrame> inv_jacobian(
      const std::array<T, 3>& source_coords) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  bool is_identity() const { return is_identity_; }

 private:
  friend bool operator==(const Rotation<3>& lhs, const Rotation<3>& rhs);

  double rotation_about_z_{std::numeric_limits<double>::signaling_NaN()};
  double rotation_about_rotated_y_{
      std::numeric_limits<double>::signaling_NaN()};
  double rotation_about_rotated_z_{
      std::numeric_limits<double>::signaling_NaN()};
  tnsr::ij<double, 3, Frame::Grid> rotation_matrix_{
      std::numeric_limits<double>::signaling_NaN()};
  bool is_identity_{false};
};

bool operator!=(const Rotation<3>& lhs, const Rotation<3>& rhs);

}  // namespace CoordinateMaps
}  // namespace domain
