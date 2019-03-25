// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
/// \endcond

// IWYU pragma: no_include <pup.h>

namespace gr {
namespace Solutions {

/*!
 * \brief Kerr black hole in Kerr-Schild coordinates
 *
 * \details
 * The metric is \f$g_{\mu\nu} = \eta_{\mu\nu} + 2 H l_\mu l_\nu\f$,
 * where \f$\eta_{\mu\nu}\f$ is the Minkowski metric, \f$H\f$ is a scalar
 * function, and \f$l_\mu\f$ is the outgoing null vector.
 * \f$H\f$ and \f$l_\mu\f$ are known functions of the coordinates
 * and of the mass and spin vector.
 *
 * The following are input file options that can be specified:
 *  - Mass (default: 1.)
 *  - Center (default: {0,0,0})
 *  - Spin (default: {0,0,0})
 *
 * ## Kerr-Schild Coordinates
 *
 *
 * A Kerr-Schild coordinate system is defined by
 * \f{equation}{
 * g_{\mu\nu}  \equiv \eta_{\mu\nu} + 2 H l_\mu l_\nu,
 * \f}
 * where \f$H\f$ is a scalar function of the coordinates, \f$\eta_{\mu\nu}\f$
 * is the Minkowski metric, and \f$l^\mu\f$ is a null vector. Note that the
 * form of the metric along with the nullness of \f$l^\mu\f$ allows one to
 * raise and lower indices of \f$l^\mu\f$ using \f$\eta_{\mu\nu}\f$, and
 * that \f$l^t l^t = l_t l_t = l^i l_i\f$.
 * Note also that
 * \f{equation}{
 * g^{\mu\nu}  \equiv \eta^{\mu\nu} - 2 H l^\mu l^\nu,
 * \f}
 * and that \f$\sqrt{-g}=1\f$.
 * Also, \f$l_\mu\f$ is a geodesic with respect to both the physical metric
 * and the Minkowski metric:
 * \f{equation}{
 * l^\mu \partial_\mu l_\nu = l^\mu\nabla_\mu l_\nu = 0.
 * \f}
 *
 *
 * The corresponding 3+1 quantities are
 * \f{eqnarray}{
 * g_{i j}     &=& \delta_{i j} + 2 H l_i l_j,\\
 * g^{i j}  &=& \delta^{i j} - {2 H l^i l^j \over 1+2H l^t l^t},\\
 * {\rm det} g_{i j}&=& 1+2H l^t l^t,\\
 * \beta^i       &=& - {2 H l^t l^i \over 1+2H l^t l^t},\\
 * N        &=& \left(1+2 H l^t l^t\right)^{-1/2},\quad\hbox{(lapse)}\\
 * \alpha     &=& \left(1+2 H l^t l^t\right)^{-1},
 *                \quad\hbox{(densitized lapse)}\\
 * K_{i j}  &=& - \left(1+2 H l^t l^t\right)^{1/2}
 *       \left[l_i l_j \partial_{t} H + 2 H l_{(i} \partial_{t} l_{j)}\right]
 *                 \nonumber \\
 *                 &&-2\left(1+2 H l^t l^t\right)^{-1/2}
 *                \left[H l^t \partial_{(i}l_{j)} + H l_{(i}\partial_{j)}l^t
 *          + l^t l_{(i}\partial_{j)} H + 2H^2 l^t l_{(i} l^k\partial_{k}l_{j)}
 *          + H l^t l_i l_j l^k \partial_{k} H\right],\\
 * \partial_{k}g_{i j}&=& 2 l_i l_j\partial_{k} H +
 *    4 H l_{(i} \partial_{k}l_{j)},\\
 * \partial_{k}N   &=& -\left(1+2 H l^t l^t\right)^{-3/2}
 *                    \left(l^tl^t\partial_{k}H+2Hl^t\partial_{k}l^t\right),\\
 * \partial_{k}\beta^i  &=& - 2\left(1+2H l^t l^t\right)^{-1}
 *    \left(l^tl^i\partial_{k}H+Hl^t\partial_{k}l^i+Hl^i\partial_{k}l^t\right)
 *                   + 4 H l^t l^i \left(1+2H l^t l^t\right)^{-2}
 *                     \left(l^tl^t\partial_{k}H+2Hl^t\partial_{k}l^t\right),\\
 * \Gamma^k{}_{i j}&=& -\delta^{k m}\left(l_i l_j \partial_{m}H
 *                                    + 2l_{(i} \partial_{m}l_{j)} \right)
 *                   + 2 H l_{(i}\partial_{j)} l^k
 *          \nonumber \\
 *                  &&+\left(1+2 H l^t l^t\right)^{-1}
 *                     \left[2 l^k l_{(i}\partial_{j)} H
 *                          +2 H l_i l_j l^k l^m \partial_{m}H
 *                          +2 H l^k \partial_{(i}l_{j)}
 *                          +4 H^2 l^k l_{(i} (l^m \partial_{m}l_{j)}
 *                                      -\partial_{j)} l^t)
 *                     \right].
 * \f}
 * Note that \f$l^i\f$ is **not** equal to \f$g^{i j} l_j\f$; it is equal
 * to \f${}^{(4)}g^{i \mu} l_\mu\f$.
 *
 * ## Kerr Spacetime
 *
 * ### Spin in the z direction
 *
 * Assume Cartesian coordinates \f$(t,x,y,z)\f$. Then for stationary Kerr
 * spacetime with mass \f$M\f$ and angular momentum \f$a M\f$
 * in the \f$z\f$ direction,
 * \f{eqnarray}{
 * H     &=& {M r^3 \over r^4 + a^2 z^2},\\
 * l_\mu &=&
 * \left(1,{rx+ay\over r^2+a^2},{ry-ax\over r^2+a^2},{z\over r}\right),
 * \f}
 * where \f$r\f$ is defined by
 * \f{equation}{
 * \label{eq:rdefinition1}
 * {x^2+y^2\over a^2+r^2} + {z^2\over r^2} = 1,
 * \f}
 * or equivalently,
 * \f{equation}{
 * r^2 = {1\over 2}(x^2 + y^2 + z^2 - a^2)
 *      + \left({1\over 4}(x^2 + y^2 + z^2 - a^2)^2 + a^2 z^2\right)^{1/2}.
 * \f}
 *
 * Possibly useful formula:
 * \f{equation}{
 * \partial_{i} r = {x_i + z \delta_{i z} \displaystyle {a^2\over r^2} \over
 *   2 r\left(1 - \displaystyle {x^2 + y^2 + z^2 - a^2\over 2 r^2}\right)}.
 * \f}
 *
 * ### Spin in an arbitrary direction
 *
 * For arbitrary spin direction, let \f$\vec{x}\equiv (x,y,z)\f$ and
 * \f$\vec{a}\f$ be a flat-space three-vector with magnitude-squared
 * (\f$\delta_{ij}\f$ norm) equal to \f$a^2\f$.
 * Then the Kerr-Schild quantities for Kerr spacetime are:
 * \f{eqnarray}{
 * H       &=& {M r^3 \over r^4 + (\vec{a}\cdot\vec{x})^2},\\
 * \vec{l} &=& {r\vec{x}-\vec{a}\times\vec{x}+(\vec{a}\cdot\vec{x})\vec{a}/r
 *              \over r^2+a^2 },\\
 * l_t     &=& 1,\\
 * \label{eq:rdefinition2}
 * r^2     &=& {1\over 2}(\vec{x}\cdot\vec{x}-a^2)
 *      + \left({1\over 4}(\vec{x}\cdot\vec{x}-a^2)^2
 *              + (\vec{a}\cdot\vec{x})^2\right)^{1/2},
 * \f}
 * where \f$\vec{l}\equiv (l_x,l_y,l_z)\f$, and
 * all dot and cross products are evaluated as flat-space 3-vector operations.
 *
 * Possibly useful formulae:
 * \f{equation}{
 * \partial_{i} r = {x_i + (\vec{a}\cdot\vec{x})a_i/r^2 \over
 *   2 r\left(1 - \displaystyle {\vec{x}\cdot\vec{x}-a^2\over 2 r^2}\right)},
 * \f}
 * \f{equation}{
 * {\partial_{i} H \over H} =
 *  {3\partial_{i}r\over r} - {4 r^3 \partial_{i}r +
 *                     2(\vec{a}\cdot\vec{x})\vec{a}
 *                     \over r^4 + (\vec{a}\cdot\vec{x})^2},
 * \f}
 * \f{equation}{
 * (r^2+a^2)\partial_{j} l_i =
 *                 (x_i-2 r l_i-(\vec{a}\cdot\vec{x})a_i/r^2)\partial_{j}r
 *                 + r\delta_{ij} + a_i a_j/r + \epsilon^{ijk} a_k.
 * \f}
 *
 * ## Cartesian and Spherical Coordinates for Kerr
 *
 * The Kerr-Schild coordinates are defined in terms of the Cartesian
 * coordinates \f$(x,y,z)\f$.  If one wishes to express Kerr-Schild
 * coordinates in terms of the spherical polar coordinates
 * \f$(\tilde{r},\theta,\phi)\f$ then one can make the obvious and
 * usual transformation
 * \f{equation}{
 * \label{eq:sphertocartsimple}
 * x=\tilde{r}\sin\theta\cos\phi,\quad
 * y=\tilde{r}\sin\theta\sin\phi,\quad
 * z=\tilde{r}\cos\theta.
 * \f}
 *
 * This is simple, and has the advantage that in this coordinate system
 * for \f$M\to0\f$, Kerr spacetime becomes Minkowski space in spherical
 * coordinates \f$(\tilde{r},\theta,\phi)\f$. However, the disadvantage is
 * that the horizon of a Kerr hole is **not** located at constant
 * \f$\tilde{r}\f$, but is located instead at constant \f$r\f$,
 * where \f$r\f$ is the radial
 * Boyer-Lindquist coordinate defined in (\f$\ref{eq:rdefinition2}\f$).
 *
 * For spin in the \f$z\f$ direction, one could use the transformation
 * \f{equation}{
 * x=\sqrt{r^2+a^2}\sin\theta\cos\phi,\quad
 * y=\sqrt{r^2+a^2}\sin\theta\sin\phi,\quad
 * z=r\cos\theta.
 * \f}
 * In this case, for \f$M\to0\f$, Kerr spacetime becomes Minkowski space in
 * spheroidal coordinates, but now the horizon is on a constant-coordinate
 * surface.
 *
 * Right now we use (\f$\ref{eq:sphertocartsimple}\f$), but we may
 * wish to use the other transformation in the future.
 */
class KerrSchild {
 public:
  static constexpr size_t volume_dim = 3;
  struct Mass {
    using type = double;
    static constexpr OptionString help = {"Mass of the black hole"};
    static type default_value() noexcept { return 1.; }
    static type lower_bound() noexcept { return 0.; }
  };
  struct Spin {
    using type = std::array<double, volume_dim>;
    static constexpr OptionString help = {
        "The [x,y,z] dimensionless spin of the black hole"};
    static type default_value() noexcept { return {{0., 0., 0.}}; }
  };
  struct Center {
    using type = std::array<double, volume_dim>;
    static constexpr OptionString help = {
        "The [x,y,z] center of the black hole"};
    static type default_value() noexcept { return {{0., 0., 0.}}; }
  };
  using options = tmpl::list<Mass, Spin, Center>;
  static constexpr OptionString help{"Black hole in Kerr-Schild coordinates"};

  KerrSchild(double mass, Spin::type dimensionless_spin, Center::type center,
             const OptionContext& context = {});

  explicit KerrSchild(CkMigrateMessage* /*unused*/) noexcept {}

  KerrSchild() = default;
  KerrSchild(const KerrSchild& /*rhs*/) = default;
  KerrSchild& operator=(const KerrSchild& /*rhs*/) = default;
  KerrSchild(KerrSchild&& /*rhs*/) noexcept = default;
  KerrSchild& operator=(KerrSchild&& /*rhs*/) noexcept = default;
  ~KerrSchild() = default;

  template <typename DataType>
  using DerivLapse = ::Tags::deriv<gr::Tags::Lapse<DataType>,
                                   tmpl::size_t<volume_dim>, Frame::Inertial>;
  template <typename DataType>
  using DerivShift =
      ::Tags::deriv<gr::Tags::Shift<volume_dim, Frame::Inertial, DataType>,
                    tmpl::size_t<volume_dim>, Frame::Inertial>;
  template <typename DataType>
  using DerivSpatialMetric = ::Tags::deriv<
      gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataType>,
      tmpl::size_t<volume_dim>, Frame::Inertial>;
  template <typename DataType>
  using tags = tmpl::list<
      gr::Tags::Lapse<DataType>, ::Tags::dt<gr::Tags::Lapse<DataType>>,
      DerivLapse<DataType>,
      gr::Tags::Shift<volume_dim, Frame::Inertial, DataType>,
      ::Tags::dt<gr::Tags::Shift<volume_dim, Frame::Inertial, DataType>>,
      DerivShift<DataType>,
      gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataType>,
      ::Tags::dt<
          gr::Tags::SpatialMetric<volume_dim, Frame::Inertial, DataType>>,
      DerivSpatialMetric<DataType>, gr::Tags::SqrtDetSpatialMetric<DataType>,
      gr::Tags::ExtrinsicCurvature<volume_dim, Frame::Inertial, DataType>,
      gr::Tags::InverseSpatialMetric<volume_dim, Frame::Inertial, DataType>>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<tags<DataType>> variables(
      const tnsr::I<DataType, volume_dim>& x, double t,
      tags<DataType> /*meta*/) const noexcept;

  // clang-tidy: no runtime references
  void pup(PUP::er& p) noexcept;  // NOLINT

  SPECTRE_ALWAYS_INLINE double mass() const noexcept { return mass_; }
  SPECTRE_ALWAYS_INLINE const std::array<double, volume_dim>& center() const
      noexcept {
    return center_;
  }
  SPECTRE_ALWAYS_INLINE const std::array<double, volume_dim>&
  dimensionless_spin() const noexcept {
    return dimensionless_spin_;
  }

 private:
  double mass_{1.0};
  std::array<double, volume_dim> dimensionless_spin_{{0.0, 0.0, 0.0}},
      center_{{0.0, 0.0, 0.0}};
};

SPECTRE_ALWAYS_INLINE bool operator==(const KerrSchild& lhs,
                                      const KerrSchild& rhs) noexcept {
  return lhs.mass() == rhs.mass() and
         lhs.dimensionless_spin() == rhs.dimensionless_spin() and
         lhs.center() == rhs.center();
}

SPECTRE_ALWAYS_INLINE bool operator!=(const KerrSchild& lhs,
                                      const KerrSchild& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace Solutions
}  // namespace gr
