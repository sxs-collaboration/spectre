// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class Rotation.

#pragma once

#include <memory>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/EmbeddingMaps/EmbeddingMap.hpp"
#include "Parallel/CharmPupable.hpp"

namespace EmbeddingMaps {

template <size_t Dim>
class Rotation;

/*! \ingroup EmbeddingMaps
 * Spatial rotation in two dimensions.
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
class Rotation<2> : public EmbeddingMap<2, 2> {
 public:
  /// Constructor.
  ///
  /// \param rotation_angle the angle \f$\alpha\f$ (in radians).
  explicit Rotation(double rotation_angle);
  Rotation() = default;
  ~Rotation() override = default;
  Rotation(const Rotation&) = delete;
  Rotation& operator=(const Rotation&) = delete;
  Rotation(Rotation&&) noexcept = default;  // NOLINT
  Rotation& operator=(Rotation&&) = delete;

  std::unique_ptr<EmbeddingMap<2, 2>> get_clone() const override;

  Point<2, Frame::Grid> operator()(
      const Point<2, Frame::Logical>& xi) const override;

  Point<2, Frame::Logical> inverse(
      const Point<2, Frame::Grid>& x) const override;

  double jacobian(const Point<2, Frame::Logical>& /*xi*/, size_t ud,
                  size_t ld) const override;

  double inv_jacobian(const Point<2, Frame::Logical>& /*xi*/, size_t ud,
                      size_t ld) const override;

  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(EmbeddingMap<2, 2>),  // NOLINT
                                     Rotation);                       // NOLINT

  explicit Rotation(CkMigrateMessage* /* m */);

  void pup(PUP::er& p) override;  // NOLINT

 private:
  double rotation_angle_{std::numeric_limits<double>::signaling_NaN()};
  tnsr::ij<double, 2, Frame::Grid> rotation_matrix_{
      std::numeric_limits<double>::signaling_NaN()};
};

/*! Spatial rotation in three dimensions using Euler angles
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
class Rotation<3> : public EmbeddingMap<3, 3> {
 public:
  /// Constructor.
  ///
  /// \param rotation_about_z the angle \f$\alpha\f$ (in radians).
  /// \param rotation_about_rotated_y the angle \f$\beta\f$ (in radians).
  /// \param rotation_about_rotated_z the angle \f$\gamma\f$ (in radians).
  Rotation(double rotation_about_z, double rotation_about_rotated_y,
           double rotation_about_rotated_z);
  Rotation() = default;
  ~Rotation() override = default;
  Rotation(const Rotation&) = delete;
  Rotation& operator=(const Rotation&) = delete;
  Rotation(Rotation&&) noexcept = default;  // NOLINT
  Rotation& operator=(Rotation&&) = delete;

  std::unique_ptr<EmbeddingMap<3, 3>> get_clone() const override;

  Point<3, Frame::Grid> operator()(
      const Point<3, Frame::Logical>& xi) const override;

  Point<3, Frame::Logical> inverse(
      const Point<3, Frame::Grid>& x) const override;

  double jacobian(const Point<3, Frame::Logical>& xi, size_t ud,
                  size_t ld) const override;

  double inv_jacobian(const Point<3, Frame::Logical>& xi, size_t ud,
                      size_t ld) const override;

  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(EmbeddingMap<3, 3>),  // NOLINT
                                     Rotation);                       // NOLINT

  explicit Rotation(CkMigrateMessage* m);

  void pup(PUP::er& p) override; // NOLINT

 private:
  double rotation_about_z_{std::numeric_limits<double>::signaling_NaN()};
  double rotation_about_rotated_y_{
      std::numeric_limits<double>::signaling_NaN()};
  double rotation_about_rotated_z_{
      std::numeric_limits<double>::signaling_NaN()};
  tnsr::ij<double, 3, Frame::Grid> rotation_matrix_{
      std::numeric_limits<double>::signaling_NaN()};
};
}  // namespace EmbeddingMaps
