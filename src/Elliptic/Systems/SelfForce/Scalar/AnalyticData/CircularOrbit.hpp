// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <pup.h>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarSelfForce::AnalyticData {

/*!
 * \brief Scalar self force for a scalar charge on a circular equatorial orbit.
 *
 * This class implements Eq. (2.9) in \cite Osburn:2022bby . It does so by
 * defining the background fields $\alpha$, $\beta$, and $\gamma_i$ in the
 * general form of the equations
 * \begin{equation}
 * -\partial_i F^i + \beta \Psi_m + \gamma_i \partial_i \Psi_m  = S_m
 * \text{.}
 * \end{equation}
 * with the flux [Eq. (2.39) in \cite Vu:2026ypc]
 * \begin{equation}
 * F^i = \{\frac{\Delta}{r^2+a^2}\partial_{r}, \frac{1-z^2}{r^2+a^2}
 \partial_{z}\} \Psi_m
 * \text{.}
 * \end{equation}
 * We make the following changes compared to \cite Osburn:2022bby :
 *
 * - Multiply by the factor $\Sigma^2 / (r^2 + a^2)^2$ so that we can easily
 *   write the equations in first-order flux form.
 * - Use $z = \cos\theta$ as angular coordinate instead of $\theta$. This avoids
 *   the $\cot\theta$ term by rewriting the angular derivatives as:
 *   \f[
 *   \partial_\theta^2+\cot\theta\partial_\theta =
 *   \partial_{z}\sin^2\theta\partial_{z}
 *   \f]
 * - Decompose $\Psi_m = \sin(\theta)^m u_m(r_\star, \theta)$. This avoids the
 *   $m^2/\sin^2\theta$ term by factoring out the singular behavior at the
 *   poles. The equations transform as:
 *   \f[
 *   -\partial_{z}\sin^2\theta\partial_{z} \Psi_m +
 *   \frac{m^2}{\sin^2\theta}\Psi_m = \sin(\theta)^m \left( m(m+1)
 *   + 2m z \partial_{z}
 *   - \partial_{z}\sin^2\theta\partial_{z} \right) u_m
 *   \f]
 *   We divide by $\sin(\theta)^m$ to get the equations for $u_m$.
 *   With the above three changes, Eq. (2.9) in \cite Osburn:2022bby becomes
 *   Eq. (2.19) in \cite Vu:2026ypc .
 * - Use $r$ as the radial coordinate instead of $r_\star$. This has two
 *   advantages: first, the horizon is placed at a finite radius rather than
 *   $r_\star \rightarrow -\infty$; second, the flux component normal to the
 *   boundary vanishes at the horizon ($r=r_{\plus}$), which reduces to
 *   simple regularity conditions.

 * Written this way, the equations are regular at the poles and converge
 * exponentially. We also don't have to apply angular boundary conditions
 * because regularity at the poles is automatically enforced by the
 * $\sin^2\theta$ factor in the flux.
 *
 * The resulting factors in the equation are:
 * \begin{align}
 * &\beta = \left(\frac{1}{\Delta}\left(-m^2\Omega^2 \Sigma^2 + 4a m^2 \Omega M
 r\right) +
 *   m (m + 1) + \frac{2M}{r}(1-\frac{a^2}{Mr}) + \frac{2iam}{r}
 *   \right) \frac{1}{r^2 + a^2} \\
 * &\gamma_{r} = -\frac{2iam}{r^2+a^2} + \frac{2 a^2 \Delta}{r(r^2+a^2)^2} \\
 * &\gamma_{z} =  \frac{2 m z}{r^2 + a^2}
 * \end{align}
 *
 * This class also provides the effective source $S_m^\mathrm{eff} = \Delta_m
 * \Psi_m^P$ and the singular field $\Psi_m^P$ in the regularized region (see
 * `ScalarSelfForce::FirstOrderSystem` and Sec. III in \cite Vu:2026ypc ).
 * The effective source is computed using the scalar EffectiveSource code by
 * Wardell et. al. (https://github.com/barrywardell/EffectiveSource and
 * \cite Wardell:2011gb ). It is then transformed to correspond to the m-mode
 * decomposition used in \cite Vu:2026ypc Eq. (2.16) as:
 * \begin{align}
 *   \Psi_m^P &= \frac{r}{2 \pi \sin(\theta)^{|m|}}
 *     e^{i m \left( \varphi - \phi\right)} \Phi_m^\mathrm{Wardell} \\
 *   S_m^\mathrm{eff} &= \frac{r}{2 \pi \sin(\theta)^{|m|}}
 *     e^{-i m \left( \varphi - \phi\right)}
 *     \frac{\Delta\,(r^2 + a^2\cos^2\theta)}{(r^2 + a^2)^2}
 *     S_m^\mathrm{Wardell}
 *   \text{,}
 * \end{align}
 * where $\varphi = \phi + \frac{a}{r_+ - r_-} \ln(\frac{r-r_+}{r-r_-})$
 * (Eq. (2.14) in \cite Vu:2026ypc ).
 * Compared to Eq. (2.8) in \cite Osburn:2022bby, we additionally divide by
 * $\sin(\theta)^{|m|}$ to account for the change of variable from $\Psi_m$ to
 * $u_m$ described above.

 *
 * \par Impose equatorial symmetry
 * Since this work is restricted to equatorial orbits, we can enforce equatorial
 * symmetry by reformulating the equations in terms of the coordinate
 * $z^2 = \cos^2\theta$ instead of $z = \cos\theta$. This transforms the factors
 * of first-order flux formulation as [Eqs. (2.44)-(2.45) in \cite Vu:2026ypc]:
 * \begin{equation}
 * F^{\cos^2\theta}  = 4 \cos^2\theta F^{\cos\theta}
 * \text{.}
 * \end{equation}
 * \begin{equation}
 * \gamma_{\cos^2\theta} = 2 \cos\theta \gamma_{\cos\theta} + 2 \alpha
 * \text{.}
 * \end{equation}
 *
 * \par Hyperboloidal slicing
 * Transforming to a hyperboloidal time coordinate $s = t - h(r_*)$ can simplify
 * the solution a lot because the slice will only intersect a finite number of
 * wave fronts. In frequency domain this transformation means that partial
 * derivatives transform as [Eqs. (2.31)-(2.32) in \cite Vu:2026ypc]:
 * \begin{align}
 *   &\partial_{r_*} \rightarrow \partial_{r_*} + i m\Omega H(r_*) \\
 *   &\partial_{r_*}^2 \rightarrow \partial_{r_*}^2 +
 *     2 i m\Omega H(r_*) \partial_{r_*} + i m\Omega H'(r_*)
 *     -m^2\Omega^2 H(r_*)^2
 * \end{align}
 * where $H(r_*) = h'(r_*)$ is the boost function that asymptotes to
 * $H(r_* \rightarrow \pm \infty) = \pm 1$.
 * This maps to the following additional terms in $\beta$ and $\gamma_{r_*}$:
 * \begin{align}
 *   &\beta \rightarrow \beta + i m\Omega \gamma_{r_*} H(r_*)
 *     - i m\Omega H'(r_*) + m^2\Omega^2 H(r_*)^2 \\
 *   &\gamma_{r_*} \rightarrow \gamma_{r_*} - 2 i m\Omega H(r_*)
 * \end{align}
 * The complete expressions for these factors are given in Eqs. (2.41)-(2.43)
 * in \cite Vu:2026ypc .
 * For the boost function we use the piecewise-constant "vtu" slicing
 * [Eq. (2.34) in \cite Vu:2026ypc]:
 * $H = -1$ in the $v$ domain ($r_* \leq r_*^v$), $H = 0$ in the $t$ domain
 * ($r_*^v \leq r_* \leq r_*^u$), and $H = +1$ in the $u$ domain
 * ($r_* \geq r_*^u$), where $r_*^v < r_{*0} < r_*^u$ and $r_{*0}$ is
 * the particle location. The jump in $H$ at the domain interfaces produces a
 * jump in the radial derivative of $\Psi_m$, which we impose as junction
 * conditions [Eqs. (2.36)-(2.37) in \cite Vu:2026ypc]:
 * \begin{align}
 *   &\left.\frac{\partial \Psi_m}{\partial r_*}\right|_{r_*=(r_*^v)^+} -
 *     \left.\frac{\partial \Psi_m}{\partial r_*}\right|_{r_*=(r_*^v)^-} =
 *     -im\Omega\Psi_m\big|_{r_*=r_*^v} \\
 *   &\left.\frac{\partial \Psi_m}{\partial r_*}\right|_{r_*=(r_*^u)^+} -
 *     \left.\frac{\partial \Psi_m}{\partial r_*}\right|_{r_*=(r_*^u)^-} =
 *     -im\Omega\Psi_m\big|_{r_*=r_*^u}
 * \end{align}
 */
class CircularOrbit : public elliptic::analytic_data::Background,
                      public elliptic::analytic_data::InitialGuess {
 public:
  struct BlackHoleMass {
    static constexpr Options::String help =
        "Kerr mass parameter 'M' of the black hole";
    using type = double;
  };
  struct BlackHoleSpin {
    static constexpr Options::String help =
        "Kerr dimensionless spin parameter 'chi' of the black hole";
    using type = double;
  };
  struct OrbitalRadius {
    static constexpr Options::String help =
        "Radius 'r_0' of the circular orbit";
    using type = double;
  };
  struct MModeNumber {
    static constexpr Options::String help =
        "Mode number 'm' of the scalar field";
    using type = int;
  };
  struct HyperboloidalSlicingTransitions {
    static constexpr Options::String help =
        "Enable hyperboloidal slicing by specifying the transition points for "
        "the boost function. The boost function transitions from -1 to zero "
        "between the first two points and from zero to 1 between the last "
        "two points. The two points can be the same, in which case the "
        "transition "
        "is discontinuous (vtu slicing) and the jump must be handled by the DG "
        "scheme (see option 'NullSlicingBlocks'). "
        "The effective source can only be evaluated where the "
        "boost function is zero, so the regularized region must be between "
        "the second and third points.";
    using type = Options::Auto<std::array<double, 4>, Options::AutoLabel::None>;
  };
  struct PenetratingHorizon {
    static constexpr Options::String help =
        "If 'False', use tortoise radial coordinate where the Kerr horizon is "
        "at negative infinity. If 'True', use Boyer-Lindquist radial "
        "coordinate where the Kerr horizon is at r_+.";
    using type = bool;
  };
  struct ImposeEquatorialSymmetry {
    static constexpr Options::String help =
        "Impose symmetry across the equatorial plane by using cos(theta)^2 "
        "as the angular coordinate instead of cos(theta). This means the "
        "domain should span [0, 1] instead of [-1, 1].";
    using type = bool;
  };
  using options = tmpl::list<BlackHoleMass, BlackHoleSpin, OrbitalRadius,
                             MModeNumber, HyperboloidalSlicingTransitions,
                             PenetratingHorizon, ImposeEquatorialSymmetry>;
  static constexpr Options::String help =
      "Quasicircular orbit of a scalar point charge in Kerr spacetime";

  CircularOrbit() = default;
  CircularOrbit(const CircularOrbit&) = default;
  CircularOrbit& operator=(const CircularOrbit&) = default;
  CircularOrbit(CircularOrbit&&) = default;
  CircularOrbit& operator=(CircularOrbit&&) = default;
  ~CircularOrbit() override = default;

  CircularOrbit(
      double black_hole_mass, double black_hole_spin, double orbital_radius,
      int m_mode_number,
      std::optional<std::array<double, 4>> hyperboloidal_slicing_transitions,
      bool penetrating_horizon, bool impose_equatorial_symmetry);

  explicit CircularOrbit(CkMigrateMessage* m);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(CircularOrbit);

  tnsr::I<double, 2> puncture_position() const;
  double black_hole_mass() const { return black_hole_mass_; }
  double black_hole_spin() const { return black_hole_spin_; }
  double orbital_radius() const { return orbital_radius_; }
  int m_mode_number() const { return m_mode_number_; }
  double omega() const;
  std::optional<std::array<double, 4>> hyperboloidal_slicing_transitions()
      const {
    return hyperboloidal_slicing_transitions_;
  }
  bool penetrating_horizon() const { return penetrating_horizon_; }
  bool impose_equatorial_symmetry() const {
    return impose_equatorial_symmetry_;
  }

  using background_tags = tmpl::list<Tags::Alpha, Tags::Beta, Tags::Gamma>;
  using source_tags = tmpl::list<
      ::Tags::FixedSource<Tags::MMode>, Tags::SingularField,
      ::Tags::deriv<Tags::SingularField, tmpl::size_t<2>, Frame::Inertial>,
      Tags::BoyerLindquistRadius>;

  // Background
  tuples::tagged_tuple_from_typelist<background_tags> variables(
      const tnsr::I<DataVector, 2>& x, background_tags /*meta*/) const;

  // Initial guess
  static tuples::TaggedTuple<Tags::MMode> variables(
      const tnsr::I<DataVector, 2>& x, tmpl::list<Tags::MMode> /*meta*/);

  // Fixed sources
  tuples::tagged_tuple_from_typelist<source_tags> variables(
      const tnsr::I<DataVector, 2>& x, source_tags /*meta*/) const;

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 2>& x, const Mesh<2>& /*mesh*/,
      const InverseJacobian<DataVector, 2, Frame::ElementLogical,
                            Frame::Inertial>& /*inv_jacobian*/,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables(x, tmpl::list<RequestedTags...>{});
  }

  // NOLINTNEXTLINE
  void pup(PUP::er& p) override;

 private:
  friend bool operator==(const CircularOrbit& lhs, const CircularOrbit& rhs);

  double black_hole_mass_{std::numeric_limits<double>::signaling_NaN()};
  double black_hole_spin_{std::numeric_limits<double>::signaling_NaN()};
  double orbital_radius_{std::numeric_limits<double>::signaling_NaN()};
  int m_mode_number_{};
  std::optional<std::array<double, 4>> hyperboloidal_slicing_transitions_{};
  bool penetrating_horizon_{false};
  bool impose_equatorial_symmetry_{false};
};

bool operator!=(const CircularOrbit& lhs, const CircularOrbit& rhs);

}  // namespace ScalarSelfForce::AnalyticData
