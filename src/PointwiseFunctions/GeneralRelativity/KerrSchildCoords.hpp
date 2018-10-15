// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"

// IWYU pragma:  no_include <pup.h>

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace gr {
/*!
 * \brief Contains helper functions for transforming tensors in Kerr
 * spacetime to Kerr-Schild coordinates.
 *
 * Consider the Kerr-Schild form of the Kerr metric in Cartesian
 * coordinates \f$ (t, x, y, z)\f$,
 *
 * \f{align*}
 * ds^2 = -dt^2 + dx^2 + dy^2 + dz^2 + \dfrac{2Mr^3}{r^4 + a^2 z^2}
 * \left[dt + \dfrac{r(x\,dx + y\,dy)}{r^2 + a^2} +
 * \dfrac{a(y\,dx - x\,dy)}{r^2 + a^2} + \dfrac{z}{r}dz\right]^2,
 * \f}
 *
 * where \f$r\f$ is the function defined through
 *
 * \f{align*}
 * r^2(x, y, z) = \frac{x^2 + y^2 + z^2 - a^2}{2} +
 * \sqrt{\left(\frac{x^2 + y^2 + z^2 - a^2}{2}\right)^2 + a^2 z^2}.
 * \f}
 *
 * Depending on the physical context, the following coordinate transformations
 * are usually adopted:
 *
 * - \em Spherical \em coordinates, defined by the usual transformation
 *
 *   \f{align*}
 *   x &= r_\text{sph}\sin\theta\cos\phi,\\
 *   y &= r_\text{sph}\sin\theta\sin\phi,\\
 *   z &= r_\text{sph}\cos\theta.
 *   \f}
 *
 *   Note that \f$r_\text{sph} \neq r\f$, so the
 *   horizon of a Kerr black hole is \em not at constant \f$r_\text{sph}\f$.
 *
 * - \em Oblate \em spheroidal \em coordinates,
 *
 *   \f{align*}
 *   x &= \sqrt{r^2 + a^2}\sin\vartheta\cos\phi,\\
 *   y &= \sqrt{r^2 + a^2}\sin\vartheta\sin\phi,\\
 *   z &= r\cos\vartheta.
 *   \f}
 *
 *   Notice that \f$r\f$ is that defined in the Kerr-Schild form of the Kerr
 *   metric, so the horizon is at constant \f$r\f$. (As it will be noted below,
 *   \f$r\f$ corresponds to the Boyer-Lindquist radial coordinate.) Also, note
 *   that the azimuthal angle \f$\phi\f$ is the same as the corresponding
 *   angle in spherical coordinates, whereas the zenithal angle
 *   \f$\vartheta\f$ differs from the corresponding angle \f$\theta\f$.
 *
 * - \em Kerr \em coordinates (sometimes also referred to as
 *   "Kerr-Schild coordinates"), defined by
 *
 *   \f{align*}
 *   x &= (r \cos\varphi - a \sin\varphi)\sin\vartheta =
 *   \sqrt{r^2 + a^2}\sin\vartheta\cos\left(\varphi +
 *   \tan^{-1}(a/ r)\right),\\
 *   y &= (r \sin\varphi + a \cos\varphi)\sin\vartheta =
 *   \sqrt{r^2 + a^2}\sin\vartheta\sin\left(\varphi +
 *   \tan^{-1}(a/ r)\right),\\
 *   z &= r \cos\vartheta.
 *   \f}
 *
 *   These are the coordinates used in the gr::KerrSchildCoords class.
 *   Notice that \f$r\f$ and \f$\vartheta\f$ are the same as those in
 *   spheroidal coordinates, whereas the azimuthal angle
 *   \f$\varphi \neq \phi\f$.
 *
 *   \note Kerr's original "advanced Eddington-Finkelstein" form of his metric
 *   is retrieved by performing the additional transformation \f$t = u - r\f$.
 *   Note that Kerr's choice of the letter \f$u\f$ for the time coordinate
 *   is not consistent with the convention of denoting the advanced time
 *   \f$v = t + r\f$ (in flat space), and the retarded time \f$u = t - r\f$
 *   (in flat space). Finally, note that Kerr's original metric has the sign
 *   of \f$a\f$ flipped.
 *
 *   The Kerr coordinates have been used in the hydro community to evolve the
 *   GRMHD equations. They are the intermediate step in getting the Kerr metric
 *   in Boyer-Lindquist coordinates. The relation between both is
 *
 *   \f{align*}
 *   dt_\text{BL} &=
 *   dt - \frac{2M r\,dr}{r^2 - 2Mr + a^2}, \\
 *   dr_\text{BL} &= dr,\\
 *   d\theta_\text{BL} &= d\vartheta,\\
 *   d\phi_\text{BL} &= d\varphi - \frac{a\,dr}{r^2-2Mr + a^2}.
 *   \f}
 *
 *   The above transformation makes explicit that \f$r\f$ is the Boyer-Lindquist
 *   radial coodinate, \f$ r = r_\text{BL}\f$. Likewise,
 *   \f$\vartheta = \theta_\text{BL}\f$.
 *
 *   \note Sometimes (especially in the GRMHD community), the Kerr coordinates
 *   as defined above are referred to as "spherical Kerr-Schild" coordinates.
 *   These should not be confused with the true spherical coordinates defined
 *   in this documentation.
 *
 *   The Kerr metric in Kerr coodinates is sometimes used to write analytic
 *   initial data for hydro simulations. In this coordinate system, the
 *   metric takes the form
 *
 *   \f{align*}
 *   ds^2 &= -(1 - B)\,dt^2 + (1 + B)\,dr^2 + \Sigma\,d\vartheta^2 +
 *   \left(r^2 + a^2 + B a^2\sin^2\vartheta\right)\sin^2\vartheta\,d\varphi^2\\
 *   &\quad + 2B\,dt\,dr - 2aB\sin^2\vartheta\,dt\,d\varphi -
 *   2a\left(1 + B\right)\sin^2\vartheta\,dr\,d\varphi,
 *   \f}
 *
 *   where \f$B = 2M r/ \Sigma\f$ and \f$\Sigma = r^2 + a^2\cos^2\vartheta\f$.
 *   Using the additional relations
 *
 *   \f{align*}
 *   \sin\vartheta &= \sqrt{\frac{x^2 + y^2}{r^2 + a^2}},\quad
 *   \cos\vartheta = \frac{z}{r},\\
 *   \sin\varphi &= \frac{y r - x a}{\sqrt{(x^2 + y^2)(r^2 + a^2)}},\quad
 *   \cos\varphi = \frac{x r + y a}{\sqrt{(x^2 + y^2)(r^2 + a^2)}},
 *   \f}
 *
 *   the Jacobian of the transformation to Cartesian Kerr-Schild coordinates is
 *
 *   \f{align*}
 *   \dfrac{\partial(x, y, z)}{\partial(r, \vartheta, \varphi)} &=
 *   \left(\begin{array}{ccc}
 *   \dfrac{xr + ya}{r^2 + a^2} &
 *   \dfrac{xz}{r}\sqrt{\dfrac{r^2 + a^2}{x^2 + y^2}} & -y\\
 *   \dfrac{yr - ax}{r^2 + a^2} &
 *   \dfrac{yz}{r}\sqrt{\dfrac{r^2 + a^2}{x^2 + y^2}} & x\\
 *   \dfrac{z}{r} & - r\sqrt{\dfrac{x^2 + y^2}{r^2 + a^2}} & 0
 *   \end{array}\right),
 *   \f}
 *
 *   which can be used to transform tensors between both coordinate systems.
 */
class KerrSchildCoords {
 public:
  KerrSchildCoords() = default;
  KerrSchildCoords(const KerrSchildCoords& /*rhs*/) = delete;
  KerrSchildCoords& operator=(const KerrSchildCoords& /*rhs*/) = delete;
  KerrSchildCoords(KerrSchildCoords&& /*rhs*/) noexcept = default;
  KerrSchildCoords& operator=(KerrSchildCoords&& /*rhs*/) noexcept = default;
  ~KerrSchildCoords() = default;

  KerrSchildCoords(double bh_mass, double bh_dimless_spin) noexcept;

  /// Transforms a spatial vector from Kerr (or "spherical Kerr-Schild")
  /// coordinates to Cartesian Kerr-Schild coordinates. If applied on points
  /// on the z-axis, the vector to transform must have a vanishing
  /// \f$v^\vartheta\f$ in order for the transformation to be single-valued.
  template <typename DataType>
  tnsr::I<DataType, 3, Frame::Inertial> cartesian_from_spherical_ks(
      const tnsr::I<DataType, 3, Frame::NoFrame>& spatial_vector,
      const tnsr::I<DataType, 3, Frame::Inertial>& cartesian_coords) const
      noexcept;

  /// Kerr-Schild \f$r^2\f$ in terms of the Cartesian coordinates.
  template <typename DataType>
  Scalar<DataType> r_coord_squared(
      const tnsr::I<DataType, 3, Frame::Inertial>& cartesian_coords) const
      noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& /*p*/) noexcept;  // NOLINT

 private:
  friend bool operator==(const KerrSchildCoords& lhs,
                         const KerrSchildCoords& rhs) noexcept;

  // The spatial components of the Jacobian of the transformation from
  // Kerr coordinates to Cartesian coordinates (x, y, z).
  template <typename DataType>
  tnsr::Ij<DataType, 3, Frame::NoFrame> jacobian_matrix(const DataType& x,
                                                        const DataType& y,
                                                        const DataType& z) const
      noexcept;

  double spin_a_ = std::numeric_limits<double>::signaling_NaN();
};

bool operator!=(const KerrSchildCoords& lhs,
                const KerrSchildCoords& rhs) noexcept;

}  // namespace gr
