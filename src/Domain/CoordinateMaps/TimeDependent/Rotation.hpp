// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
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
namespace CoordinateMaps {
namespace TimeDependent {

/*!
 * \ingroup CoordMapsTimeDependentGroup
 * \brief Time-dependent spatial rotation in two or three dimensions.
 *
 * ### General Transformation
 *
 * Let the source coordinates \f$ \vec{\xi} \f$ be mapped to coordinates \f$
 * \vec{x} \f$ using the transformation
 *
 * \f[ \vec{x} = R(t)\vec{\xi}, \f]
 *
 * where \f$ R(t) \f$ is a rotation matrix of proper dimensionality (defined
 * below) and \f$ A\vec{v} \f$ is the standard matrix-vector multiplicaton. For
 * 2D rotation, \f$ \vec{\xi} = \left(\xi, \eta\right) \f$ and \f$ \vec{x} =
 * \left(x, y\right) \f$ while for 3D rotations \f$ \vec{\xi} = \left(\xi, \eta,
 * \zeta\right) \f$ and \f$ \vec{x} = \left(x, y, z\right) \f$.
 *
 * The inverse transformation is
 *
 * \f[ \vec{\xi} = R^T(t) \vec{x} \f]
 *
 * because the inverse of a rotation matrix is its transpose.
 *
 * The frame velocity \f$ \vec{v} = d\vec{x}/dt \f$ is
 *
 * \f[ \vec{v} = \frac{d}{dt}\big( R(t) \big) \vec{\xi} \f]
 *
 * where \f$ d(R(t))/dt \f$ is the time derivative of the rotation matrix.
 *
 * The components of the Jacobian \f$ \partial x^i/\partial\xi^j \f$ are
 * trivially related to the components of the rotation matrix by
 *
 * \f[ \partial x^i/\partial\xi^j = R_{ij}, \f]
 *
 * and similarly the components of the inverse Jacobian \f$ \partial
 * \xi^i/\partial x^j \f$ are
 *
 * \f[ \partial \xi^i/\partial x^j = R^{-1}_{ij} = R^T_{ij} = R_{ji}. \f]
 *
 * ### 2D Rotation Matrix
 *
 * The 2D rotaion matrix is defined in the usual way as
 *
 * \f[
 * R(t) =
 *   \begin{bmatrix}
 *   \cos(\theta(t)) & -\sin(\theta(t)) \\
 *   \sin(\theta(t)) &  \cos(\theta(t)) \\
 *   \end{bmatrix}.
 * \f]
 *
 * We associate the polar coordinates \f$ \left( \mathrm{P}, \Phi\right) \f$
 * with the unmapped coordinates \f$ \left(\xi, \eta\right) \f$ and the polar
 * coordinates \f$ \left(r,\phi\right) \f$ with the mapped coordinates \f$
 * \left(x, y\right) \f$. We then have \f$ \phi = \Phi + \theta(t) \f$.
 *
 * The derivative of the rotation matrix is then
 *
 * \f[
 * R(t) =
 *   \begin{bmatrix}
 *   -\omega(t) \sin(\theta(t)) & -\omega(t)\cos(\theta(t)) \\
 *    \omega(t) \cos(\theta(t)) & -\omega(t)\sin(\theta(t)) \\
 *   \end{bmatrix}.
 * \f]
 *
 * where \f$ \omega(t) = d\theta(t)/dt \f$.
 *
 * \note This 2D rotation is assumed to be in the \f$ xy \f$-plane (about the
 * \f$ z \f$-axis).
 *
 * ### 3D Rotation Matrix
 *
 * For 3D rotations, we use quaternions to represent rotations about an
 * arbitrary axis. We define a unit quaternion as
 *
 * \f[
 *   \mathbf{q}
 *     = \left(q_0, q_1, q_2, q_3\right)
 *     = \left(q_0, \vec{q}\right)
 *     = \left(\cos(\frac{\theta(t)}{2}),
 *             \hat{n}\sin(\frac{\theta(t)}{2})\right)
 * \f]
 *
 * where \f$ \hat{n} \f$ is our arbitrary rotation axis and \f$ \theta(t) \f$ is
 * the angle rotated about that axis. A rotation in 3D is then defined as
 *
 * \f[ \mathbf{x} = \mathbf{q}\mathbf{\xi}\mathbf{q}^* \f]
 *
 * where \f$ \mathbf{q}^* = \left(\cos(\theta(t)/2),
 * -\hat{n}\sin(\theta(t)/2)\right) \f$ and we promote the vectors to
 * quaternions as \f$ \mathbf{x} = \left(0, \vec{x}\right) \f$ and \f$
 * \mathbf{\xi} = \left(0, \vec{\xi}\right) \f$. This will rotate the vector \f$
 * \vec{\xi} \f$ about \f$ \hat{n} \f$ by an angle \f$ \theta(t) \f$,
 * transforming it into \f$ \vec{x} \f$.
 *
 * We can represent this rotation using quaternions as a rotation matrix of
 * the form
 *
 * \f[
 * R(t) =
 *   \begin{bmatrix}
 *   q_0^2 + q_1^2 - q_2^2 - q_3^2 & 2(q_1q_2 - q_0q_3) & 2(q_1q_3 + q_0q_2) \\
 *   2(q_1q_2 + q_0q_3) & q_0^2 + q_2^2 - q_1^2 - q_3^2 & 2(q_2q_3 - q_0q_1) \\
 *   2(q_1q_3 - q_0q_2) & 2(q_2q_3 + q_0q_1) & q_0^2 + q_3^2 - q_1^2 - q_2^2 \\
 *   \end{bmatrix}.
 * \f]
 *
 * The derivative of this rotation matrix can expressed in a similar form
 *
 * \f[
 * R(t) =
 *   \begin{bmatrix}
 *   2(q_0\dot{q_0} + q_1\dot{q_1} - q_2\dot{q_2} - q_3\dot{q_3})
 *     & 2(\dot{q_1}q_2 + q_1\dot{q_2} - \dot{q_0}q_3 - q_0\dot{q_3})
 *     & 2(\dot{q_1}q_3 + q_1\dot{q_3} + \dot{q_0}q_2 + q_0\dot{q_2}) \\
 *   2(\dot{q_1}q_2 + q_1\dot{q_2} + \dot{q_0}q_3 + q_0\dot{q_3})
 *     & 2(q_0\dot{q_0} + q_2\dot{q_2} - q_1\dot{q_1} - q_3\dot{q_3})
 *     & 2(\dot{q_2}q_3 + q_2\dot{q_3} - \dot{q_0}q_1 - q_0\dot{q_1}) \\
 *   2(\dot{q_1}q_3 + q_1\dot{q_3} - \dot{q_0}q_2 - q_0\dot{q_2})
 *     & 2(\dot{q_2}q_3 + q_2\dot{q_3} + \dot{q_0}q_1 + q_0\dot{q_1})
 *     & 2(q_0\dot{q_0} + q_3\dot{q_3} - q_1\dot{q_1} - q_2\dot{q_2}) \\
 *   \end{bmatrix}.
 * \f]
 *
 * \note If you choose \f$ \hat{n} = (0, 0, 1) \f$, this rotation will be
 * equivalent to the 2D rotation.
 */
template <size_t Dim>
class Rotation {
 public:
  static_assert(Dim == 2 or Dim == 3,
                "Rotation map can only be constructed in 2 or 3 dimensions.");
  static constexpr size_t dim = Dim;

  explicit Rotation(std::string function_of_time_name);
  Rotation() = default;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> operator()(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;

  /// The inverse function is only callable with doubles because the inverse
  /// might fail if called for a point out of range, and it is unclear
  /// what should happen if the inverse were to succeed for some points in a
  /// DataVector but fail for other points.
  std::optional<std::array<double, Dim>> inverse(
      const std::array<double, Dim>& target_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;

  template <typename T>
  std::array<tt::remove_cvref_wrap_t<T>, Dim> frame_velocity(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> jacobian(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;

  template <typename T>
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, Dim, Frame::NoFrame> inv_jacobian(
      const std::array<T, Dim>& source_coords, double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  static bool is_identity() { return false; }

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const Rotation<LocalDim>& lhs,
                         const Rotation<LocalDim>& rhs);
  std::string f_of_t_name_;
};

template <size_t Dim>
bool operator!=(const Rotation<Dim>& lhs, const Rotation<Dim>& rhs);

}  // namespace TimeDependent
}  // namespace CoordinateMaps
}  // namespace domain
