// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/MakeArray.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er; // IWYU pragma: keep
}  // namespace PUP
/// \endcond

// IWYU pragma: no_include <pup.h>

namespace RelativisticEuler {
namespace Solutions {

/*!
 * \brief Fluid disk orbiting a Kerr black hole
 *
 * The Fishbone-Moncrief solution to the 3D relativistic Euler system,
 * representing the isentropic flow of a thick fluid disk orbiting a Kerr black
 * hole. In Boyer-Lindquist coordinates \f$(t, r, \theta, \phi)\f$, the flow is
 * assumed to be purely toroidal,
 *
 * \f{align*}
 * u^\mu = (u^t, 0, 0, u^\phi),
 * \f}
 *
 * where \f$u^\mu\f$ is the 4-velocity. Then, all the fluid quantities are
 * assumed to share the same symmetries as those of the background spacetime,
 * namely they are stationary (independent of \f$t\f$), and axially symmetric
 * (independent of \f$\phi\f$). Self-gravity is neglected, so that the fluid
 * variables are determined as functions of the metric. Following the treatment
 * by Kozlowski et al. (1978) (but using signature +2) the solution is
 * expresssed in terms of the quantities
 *
 * \f{align*}
 * \Omega &= \dfrac{u^\phi}{u^t},\\
 * W &= W_\text{in} - \int_{p_\text{in}}^p\frac{dp}{e + p},
 * \f}
 *
 * where \f$\Omega\f$ is the angular velocity, \f$p\f$ is the fluid pressure,
 * \f$e\f$ is the energy density, and \f$W\f$ is an auxiliary quantity
 * interpreted in the Newtonian limit as the total (gravitational + centrifugal)
 * potential. \f$W_\text{in}\f$ and \f$p_\text{in}\f$ are the potential and
 * the pressure at the radius of the inner edge, i.e. the closest edge to the
 * black hole. Here we assume \f$p_\text{in} = 0.\f$ The solution to the Euler
 * equation is then
 *
 * \f{align*}
 * (u^t)^2 &= \frac{A}{2\Delta\Sigma}\left(1 +
 * \sqrt{1 + \frac{4l^2\Delta \Sigma^2}{A^2\sin^2\theta}}\right)\\
 * \Omega &= \frac{\Sigma}{A (u^t)^2}\frac{l}{\sin^2\theta} + \frac{2Mra}{A}\\
 * u^\phi &= \Omega u^t\\
 * W &= l\Omega - \ln u^t,
 * \f}
 *
 * where
 *
 * \f{align*}
 * \Sigma = r^2 + a^2\cos^2\theta\qquad
 * \Delta = r^2 - 2Mr + a^2\qquad
 * A = (r^2 + a^2)^2 - \Delta a^2 \sin^2\theta
 * \f}
 *
 * and \f$l = u_\phi u^t\f$ is the so-called angular momentum per unit
 * intertial mass, which is a parameter defining an
 * individual disk. In deriving the solution, an integration constant has been
 * chosen so that \f$ W\longrightarrow 0\f$ as \f$r\longrightarrow \infty\f$,
 * in accordance with the Newtonian limit. Note that, from its definition,
 * equipotential contours coincide with isobaric contours. Physically, the
 * matter can fill each of the closed surfaces \f$W = \text{const}\f$, giving
 * rise to an orbiting thick disk. For \f$W > 0\f$, all equipotentials are open,
 * whereas for \f$W < 0\f$, some of them will be closed. Should a disk exist,
 * the pressure reaches a maximum value on the equator at a coordinate radius
 * \f$r_\text{max}\f$ that is related to the angular momentum per unit inertial
 * mass via
 *
 * \f{align*}
 * l = \dfrac{M^{1/2}(r_\text{max}^{3/2} + aM^{1/2})(a^2 - 2aM^{1/2}
 * r_\text{max}^{1/2} + r_\text{max}^2)}{2aM^{1/2}r_\text{max}^{3/2} +
 * (r_\text{max} - 3M)r_\text{max}^2}.
 * \f}
 *
 * Once \f$W\f$ is determined, an equation of state is required in order to
 * obtain the thermodynamic variables. If the flow is isentropic, the specific
 * enthalpy can readily be obtained from the first and second laws of
 * thermodynamics: one has
 *
 * \f{align*}
 * \frac{dp}{e + p} = \frac{dh}{h}
 * \f}
 *
 * so that
 *
 * \f{align*}
 * h = h_\text{in}\exp(W_\text{in} - W),
 * \f}
 *
 * and the pressure can be obtained from a thermodynamic relation of the form
 * \f$h = h(p)\f$. Here we assume a polytropic relation
 *
 * \f{align*}
 * p = K\rho^\gamma.
 * \f}
 *
 * Once all the variables are known in Boyer-Lindquist coordinates, it is
 * straightforward to write them in Kerr-Schild coordinates. The coordinate
 * transformation
 *
 * \f{align*}
 * t_\text{KS} &= t\\
 * x &= \sqrt{r^2 + a^2}\sin\theta\cos\phi\\
 * y &= \sqrt{r^2 + a^2}\sin\theta\sin\phi\\
 * z &= r\cos\theta
 * \f}
 *
 * helps read the Jacobian matrix, which, applied to the azimuthal flow of the
 * disk, gives
 *
 * \f{align*}
 * u_\text{KS}^\mu = u^t(1, -y\Omega, x\Omega, 0),
 * \f}
 *
 * where \f$u^t\f$ and \f$\Omega\f$ are now understood as functions of the
 * Kerr-Schild coordinates, for which the relations
 *
 * \f{align*}
 * r^2 &= \frac{1}{2}\left(x^2 + y^2 + z^2 - a^2 +
 * \sqrt{(x^2 + y^2 + z^2 - a^2)^2 + 4a^2z^2}\right)\\
 * \theta &= \cos^{-1}\left(\frac{z}{r}\right)
 * \f}
 *
 * are needed. Finally, the spatial velocity can be readily obtained from its
 * definition,
 *
 * \f{align*}
 * \alpha v^i = \frac{u^i}{u^t} + \beta^i,
 * \f}
 *
 * where \f$\alpha\f$ and \f$\beta^i\f$ are the lapse and the shift,
respectively.
 *
 * \note Kozlowski et al. (1978) denote \f$l_* = u_\phi u^t\f$ in order to
 * distinguish this quantity from their own definition \f$l = - u_\phi/u_t\f$.
 */
class FishboneMoncriefDisk {
  template <typename DataType, bool NeedSpacetime>
  struct IntermediateVariables;

 public:
  /// The mass of the black hole.
  struct BlackHoleMass {
    using type = double;
    static constexpr OptionString help = {"The mass of the black hole."};
    static type lower_bound() { return 0.0; }
  };
  /// The black hole spin magnitude in units of the black hole mass.
  struct BlackHoleSpin {
    using type = double;
    static constexpr OptionString help = {
        "The dimensionless black hole spin magnitude."};
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };
  /// The radial coordinate of the inner edge of the disk.
  struct InnerEdgeRadius {
    using type = double;
    static constexpr OptionString help = {
        "The radial coordinate of the inner edge of the disk."};
  };
  /// The radial coordinate at which the pressure reaches its maximum.
  struct MaxPressureRadius {
    using type = double;
    static constexpr OptionString help = {
        "The radial coordinate of the maximum pressure."};
  };
  /// The polytropic constant of the fluid.
  struct PolytropicConstant {
    using type = double;
    static constexpr OptionString help = {
        "The polytropic constant of the fluid."};
    static type lower_bound() { return 0.; }
  };
  /// The polytropic exponent of the fluid.
  struct PolytropicExponent {
    using type = double;
    static constexpr OptionString help = {
        "The polytropic exponent of the fluid."};
    static type lower_bound() { return 1.; }
  };

  using options =
      tmpl::list<BlackHoleMass, BlackHoleSpin, InnerEdgeRadius,
                 MaxPressureRadius, PolytropicConstant, PolytropicExponent>;
  static constexpr OptionString help = {
      "Fluid disk orbiting a Kerr black hole."};

  FishboneMoncriefDisk() = default;
  FishboneMoncriefDisk(const FishboneMoncriefDisk& /*rhs*/) = delete;
  FishboneMoncriefDisk& operator=(const FishboneMoncriefDisk& /*rhs*/) = delete;
  FishboneMoncriefDisk(FishboneMoncriefDisk&& /*rhs*/) noexcept = default;
  FishboneMoncriefDisk& operator=(FishboneMoncriefDisk&& /*rhs*/) noexcept =
      default;
  ~FishboneMoncriefDisk() = default;

  FishboneMoncriefDisk(double black_hole_mass, double black_hole_spin,
                       double inner_edge_radius, double max_pressure_radius,
                       double polytropic_constant,
                       double polytropic_exponent) noexcept;

  // Eventually, if we implement a gr::Solutions::BoyerLindquist
  // black hole, the following two functions aren't needed, and the algebra
  // of the three functions after can be simplified by using the corresponding
  // lapse, shift, and spatial metric.
  template <typename DataType>
  DataType sigma(const DataType& r_sqrd, const DataType& sin_theta_sqrd) const
      noexcept;

  template <typename DataType>
  DataType inv_ucase_a(const DataType& r_sqrd, const DataType& sin_theta_sqrd,
                       const DataType& delta) const noexcept;

  template <typename DataType>
  DataType four_velocity_t_sqrd(const DataType& r_sqrd,
                                const DataType& sin_theta_sqrd,
                                double angular_momentum) const noexcept;

  template <typename DataType>
  DataType angular_velocity(const DataType& r_sqrd,
                            const DataType& sin_theta_sqrd,
                            double angular_momentum) const noexcept;

  template <typename DataType>
  DataType potential(const DataType& r_sqrd, const DataType& sin_theta_sqrd,
                     double angular_momentum) const noexcept;

  // @{
  /// The fluid variables in Cartesian Kerr-Schild coordinates at `(x, t)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         const double t,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    IntermediateVariables<
        DataType, tmpl2::flat_any_v<cpp17::is_same_v<
                      Tags, hydro::Tags::SpatialVelocity<DataType, 3>>...>>
        vars(black_hole_mass_, black_hole_spin_, max_pressure_radius_,
             background_spacetime_, x, t);
    return {std::move(get<Tags>(variables(x, tmpl::list<Tags>{}, vars)))...};
  }

  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     const double t,  // NOLINT
                                     tmpl::list<Tag> /*meta*/) const noexcept {
    return variables(
        x, tmpl::list<Tag>{},
        IntermediateVariables<
            DataType,
            cpp17::is_same_v<Tag, hydro::Tags::SpatialVelocity<DataType, 3>>>(
            black_hole_mass_, black_hole_spin_, max_pressure_radius_,
            background_spacetime_, x, t));
  }
  // @}

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

  double black_hole_mass() const noexcept { return black_hole_mass_; }
  double black_hole_spin() const noexcept { return black_hole_spin_; }
  double inner_edge_radius() const noexcept { return inner_edge_radius_; }
  double max_pressure_radius() const noexcept { return max_pressure_radius_; }
  double polytropic_constant() const noexcept { return polytropic_constant_; }
  double polytropic_exponent() const noexcept { return polytropic_exponent_; }

  const EquationsOfState::PolytropicFluid<true>& equation_of_state() const
      noexcept {
    return equation_of_state_;
  }

  const gr::Solutions::KerrSchild& background_spacetime() const noexcept {
    return background_spacetime_;
  }

 private:
  template <typename DataType, bool NeedSpacetime>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/,
      const IntermediateVariables<DataType, NeedSpacetime>& vars) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType, bool NeedSpacetime>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/,
      const IntermediateVariables<DataType, NeedSpacetime>& vars) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType, bool NeedSpacetime>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/,
                 const IntermediateVariables<DataType, NeedSpacetime>& vars)
      const noexcept -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType, bool NeedSpacetime>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/,
      const IntermediateVariables<DataType, NeedSpacetime>& vars) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/,
                 const IntermediateVariables<DataType, true>& vars) const
      noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/,
                 const IntermediateVariables<DataType, true>& vars) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType, bool NeedSpacetime>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<
          hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>> /*meta*/,
      const IntermediateVariables<DataType, NeedSpacetime>& vars) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>;

  template <typename DataType, bool NeedSpacetime>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/,
      const IntermediateVariables<DataType, NeedSpacetime>& vars) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType, bool NeedSpacetime, typename Func>
  void variables_impl(
      const IntermediateVariables<DataType, NeedSpacetime>& vars, Func f) const
      noexcept;

  // Intermediate variables needed to set several of the Fishbone-Moncrief
  // solution's variables.
  template <typename DataType, bool NeedSpacetime>
  struct IntermediateVariables {
    IntermediateVariables(double black_hole_mass, double black_hole_spin,
                          double max_pressure_radius,
                          const gr::Solutions::KerrSchild& background_spacetime,
                          const tnsr::I<DataType, 3>& x, double t) noexcept;

    DataType r_squared{};
    DataType sin_theta_squared{};
    double angular_momentum{};
    Scalar<DataType> inv_lapse{};
    tnsr::I<DataType, 3, Frame::Inertial> shift{};
    tnsr::ii<DataType, 3, Frame::Inertial> spatial_metric{};
  };

  double black_hole_mass_ = std::numeric_limits<double>::signaling_NaN();
  double black_hole_spin_ = std::numeric_limits<double>::signaling_NaN();
  double inner_edge_radius_ = std::numeric_limits<double>::signaling_NaN();
  double max_pressure_radius_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::PolytropicFluid<true> equation_of_state_{};
  gr::Solutions::KerrSchild background_spacetime_{};
};

bool operator==(const FishboneMoncriefDisk& lhs,
                const FishboneMoncriefDisk& rhs) noexcept;
bool operator!=(const FishboneMoncriefDisk& lhs,
                const FishboneMoncriefDisk& rhs) noexcept;
}  // namespace Solutions
}  // namespace RelativisticEuler
