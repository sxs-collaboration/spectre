// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/SphericalKerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/Solutions.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/Temperature.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
namespace grmhd::AnalyticData {
class MagnetizedFmDisk;
}  // namespace grmhd::AnalyticData
/// \endcond

// IWYU pragma: no_include <pup.h>

namespace RelativisticEuler {
namespace Solutions {

/*!
 * \brief Fluid disk orbiting a Kerr black hole
 *
 * The Fishbone-Moncrief solution to the 3D relativistic Euler system
 * \cite Fishbone1976apj, representing the isentropic flow of a thick fluid disk
 * orbiting a Kerr black hole. In Boyer-Lindquist coordinates
 * \f$(t, r, \theta, \phi)\f$, the flow is assumed to be purely toroidal,
 *
 * \f{align*}
 * u^\mu = (u^t, 0, 0, u^\phi),
 * \f}
 *
 * where \f$u^\mu\f$ is the 4-velocity. Then, all the fluid quantities are
 * assumed to share the same symmetries as those of the background spacetime,
 * namely they are stationary (independent of \f$t\f$), and axially symmetric
 * (independent of \f$\phi\f$).
 *
 * Self-gravity is neglected, so that the fluid
 * variables are determined as functions of the metric. Following the treatment
 * by Kozlowski et al. \cite Kozlowski1978aa (but using signature +2)
 * the solution is expressed in terms of the quantities
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
 * Once all the variables are known in Boyer-Lindquist (or Kerr) coordinates, it
 * is straightforward to write them in Cartesian Kerr-Schild coordinates. The
 * coordinate transformation in gr::KerrSchildCoords helps read the Jacobian
 * matrix, which, applied to the azimuthal flow of the disk, gives
 *
 * \f{align*}
 * u_\text{KS}^\mu = u^t(1, -y\Omega, x\Omega, 0),
 * \f}
 *
 * where \f$u^t\f$ and \f$\Omega\f$ are now understood as functions of the
 * Kerr-Schild coordinates. Finally, the spatial velocity can be readily
 * obtained from its definition,
 *
 * \f{align*}
 * \alpha v^i = \frac{u^i}{u^t} + \beta^i,
 * \f}
 *
 * where \f$\alpha\f$ and \f$\beta^i\f$ are the lapse and the shift,
 * respectively.
 *
 * \note Kozlowski et al. \cite Kozlowski1978aa denote
 * \f$l_* = u_\phi u^t\f$ in order to
 * distinguish this quantity from their own definition \f$l = - u_\phi/u_t\f$.
 *
 * \note When using Kerr-Schild coordinates, the horizon that is at
 * constant \f$r\f$ is not spherical, but instead spheroidal. This could make
 * application of boundary condition and computing various fluxes
 * across the horizon more complicated than they need to be.
 * Thus, we use Spherical Kerr-Schild coordinates,
 * see gr::Solutions::SphericalKerrSchild, in which constant \f$r\f$
 * is spherical. Because we compute variables in Kerr-Schild coordinates,
 * there is a necessary extra step of transforming them back to
 * Spherical Kerr-Schild coordinates.
 *
 */
class FishboneMoncriefDisk
    : public virtual evolution::initial_data::InitialData,
      public MarkAsAnalyticSolution,
      public AnalyticSolution<3>,
      public hydro::TemperatureInitialization<FishboneMoncriefDisk> {
 protected:
  template <typename DataType>
  struct IntermediateVariables;

 public:
  using equation_of_state_type = EquationsOfState::PolytropicFluid<true>;

  /// The mass of the black hole, \f$M\f$.
  struct BhMass {
    using type = double;
    static constexpr Options::String help = {"The mass of the black hole."};
    static type lower_bound() { return 0.0; }
  };
  /// The dimensionless black hole spin, \f$\chi = a/M\f$.
  struct BhDimlessSpin {
    using type = double;
    static constexpr Options::String help = {
        "The dimensionless black hole spin."};
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
  };
  /// The radial coordinate of the inner edge of the disk, in units of \f$M\f$.
  struct InnerEdgeRadius {
    using type = double;
    static constexpr Options::String help = {
        "The radial coordinate of the inner edge of the disk."};
  };
  /// The radial coordinate of the maximum pressure, in units of \f$M\f$.
  struct MaxPressureRadius {
    using type = double;
    static constexpr Options::String help = {
        "The radial coordinate of the maximum pressure."};
  };
  /// The polytropic constant of the fluid.
  struct PolytropicConstant {
    using type = double;
    static constexpr Options::String help = {
        "The polytropic constant of the fluid."};
    static type lower_bound() { return 0.; }
  };
  /// The polytropic exponent of the fluid.
  struct PolytropicExponent {
    using type = double;
    static constexpr Options::String help = {
        "The polytropic exponent of the fluid."};
    static type lower_bound() { return 1.; }
  };

  using options =
      tmpl::list<BhMass, BhDimlessSpin, InnerEdgeRadius, MaxPressureRadius,
                 PolytropicConstant, PolytropicExponent>;
  static constexpr Options::String help = {
      "Fluid disk orbiting a Kerr black hole."};

  FishboneMoncriefDisk() = default;
  FishboneMoncriefDisk(const FishboneMoncriefDisk& /*rhs*/) = default;
  FishboneMoncriefDisk& operator=(const FishboneMoncriefDisk& /*rhs*/) =
      default;
  FishboneMoncriefDisk(FishboneMoncriefDisk&& /*rhs*/) = default;
  FishboneMoncriefDisk& operator=(FishboneMoncriefDisk&& /*rhs*/) = default;
  ~FishboneMoncriefDisk() override = default;

  FishboneMoncriefDisk(double bh_mass, double bh_dimless_spin,
                       double inner_edge_radius, double max_pressure_radius,
                       double polytropic_constant, double polytropic_exponent);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit FishboneMoncriefDisk(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(FishboneMoncriefDisk);
  /// \endcond

  // Eventually, if we implement a gr::Solutions::BoyerLindquist
  // black hole, the following two functions aren't needed, and the algebra
  // of the three functions after can be simplified by using the corresponding
  // lapse, shift, and spatial metric.
  template <typename DataType>
  DataType sigma(const DataType& r_sqrd, const DataType& sin_theta_sqrd) const;

  template <typename DataType>
  DataType inv_ucase_a(const DataType& r_sqrd, const DataType& sin_theta_sqrd,
                       const DataType& delta) const;

  template <typename DataType>
  DataType four_velocity_t_sqrd(const DataType& r_sqrd,
                                const DataType& sin_theta_sqrd) const;

  template <typename DataType>
  DataType angular_velocity(const DataType& r_sqrd,
                            const DataType& sin_theta_sqrd) const;

  template <typename DataType>
  DataType potential(const DataType& r_sqrd,
                     const DataType& sin_theta_sqrd) const;

  template <typename DataType>
  using tags =
      tmpl::append<hydro::grmhd_tags<DataType>,
                   typename gr::Solutions::SphericalKerrSchild::tags<DataType>>;

  /// @{
  /// The variables in Cartesian Spherical-Kerr-Schild coordinates at `(x, t)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         const double /*t*/,
                                         tmpl::list<Tags...> /*meta*/) const {
    // Can't store IntermediateVariables as member variable because we need to
    // be threadsafe.
    IntermediateVariables<DataType> vars(x);
    return {std::move(
        get<Tags>(variables(x, tmpl::list<Tags>{}, make_not_null(&vars))))...};
  }

  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     const double /*t*/,
                                     tmpl::list<Tag> /*meta*/) const {
    // Can't store IntermediateVariables as member variable because we need to
    // be threadsafe.
    IntermediateVariables<DataType> vars(x);
    return variables(x, tmpl::list<Tag>{}, make_not_null(&vars));
  }
  /// @}

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  const EquationsOfState::PolytropicFluid<true>& equation_of_state() const {
    return equation_of_state_;
  }

 protected:
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/,
                 gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/,
                 gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/,
                 gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/,
                 gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/,
                 gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::Temperature<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/,
      gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/,
                 gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/,
                 gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/,
                 gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/,
      gsl::not_null<IntermediateVariables<DataType>*> vars) const
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  // Grab the metric variables
  template <typename DataType, typename Tag,
            Requires<not tmpl::list_contains_v<
                tmpl::push_back<hydro::grmhd_tags<DataType>,
                                hydro::Tags::SpecificEnthalpy<DataType>,
                                hydro::Tags::SpatialVelocity<DataType, 3>,
                                hydro::Tags::LorentzFactor<DataType>>,
                Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(
      const tnsr::I<DataType, 3>& x, tmpl::list<Tag> /*meta*/,
      gsl::not_null<IntermediateVariables<DataType>*> vars) const {
    return {get<Tag>(background_spacetime_.variables(
        x, 0.0, tmpl::list<Tag>{},
        make_not_null(&vars->sph_kerr_schild_cache)))};
  }

  template <typename DataType, typename Func>
  void variables_impl(gsl::not_null<IntermediateVariables<DataType>*> vars,
                      Func f) const;

  // Intermediate variables needed to set several of the Fishbone-Moncrief
  // solution's variables.

  template <typename DataType>
  struct IntermediateVariables {
    explicit IntermediateVariables(const tnsr::I<DataType, 3>& x);

    DataType r_squared{};
    DataType sin_theta_squared{};
    gr::Solutions::SphericalKerrSchild::IntermediateVars<DataType,
                                                         Frame::Inertial>
        sph_kerr_schild_cache =
            gr::Solutions::SphericalKerrSchild::IntermediateVars<
                DataType, Frame::Inertial>(0);
  };

  friend bool operator==(const FishboneMoncriefDisk& lhs,
                         const FishboneMoncriefDisk& rhs);
  friend class grmhd::AnalyticData::MagnetizedFmDisk;

  double bh_mass_ = std::numeric_limits<double>::signaling_NaN();
  double bh_spin_a_ = std::numeric_limits<double>::signaling_NaN();
  double inner_edge_radius_ = std::numeric_limits<double>::signaling_NaN();
  double max_pressure_radius_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
  double angular_momentum_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::PolytropicFluid<true> equation_of_state_{};
  gr::Solutions::SphericalKerrSchild background_spacetime_{};
};

bool operator!=(const FishboneMoncriefDisk& lhs,
                const FishboneMoncriefDisk& rhs);
}  // namespace Solutions
}  // namespace RelativisticEuler
