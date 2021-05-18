// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <string>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include <pup.h>

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace grmhd::AnalyticData {
/*!
 * \brief Initial conditions for relativistic MHD Riemann problems
 *
 * The standard problems were first collected in \cite Balsara2001. A complete
 * Riemann solver for the RMHD equations is presented in \cite Giacomazzo2006
 * and can be downloaded from https://www.brunogiacomazzo.org/?page_id=395
 *
 * The domain is from \f$[-0.5,0.5]^3\f$ with periodic boundary conditions in
 * \f$y\f$ and \f$z\f$. The problems are usually run at a resolution of 1600
 * finite difference/finite volume grid points and the initial discontinuity is
 * located at \f$x=0\f$.
 *
 * While the problems were originally run in Minkowski space, changing the lapse
 * \f$\alpha\f$ to a non-unity constant value and/or the shift \f$\beta^x\f$ to
 * a non-zero constant value allows testing some of the metric terms in the
 * evolution equations in a fairly simple setup.
 *
 * Below are the initial conditions for the 5 different Balsara Riemann
 * problems. Please note that RP5 has a different final time than the rest, and
 * that RP1 has a different adiabatic index than the rest.
 *
 * RP1:
 *  - AdiabaticIndex: 2.0
 *  - LeftDensity: 1.0
 *  - LeftPressure: 1.0
 *  - LeftVelocity: [0.0, 0.0, 0.0]
 *  - LeftMagneticField: [0.5, 1.0, 0.0]
 *  - RightDensity: 0.125
 *  - RightPressure: 0.1
 *  - RightVelocity: [0.0, 0.0, 0.0]
 *  - RightMagneticField: [0.5, -1.0, 0.0]
 *  - Lapse: 1.0
 *  - ShiftX: 0.0
 *  - Final time: 0.4
 *
 * RP2:
 *  - AdiabaticIndex: 1.66666666666666666
 *  - LeftDensity: 1.0
 *  - LeftPressure: 30.0
 *  - LeftVelocity: [0.0, 0.0, 0.0]
 *  - LeftMagneticField: [5.0, 6.0, 6.0]
 *  - RightDensity: 1.0
 *  - RightPressure: 1.0
 *  - RightVelocity: [0.0, 0.0, 0.0]
 *  - RightMagneticField: [5.0, 0.7, 0.7]
 *  - Lapse: 1.0
 *  - ShiftX: 0.0
 *  - Final time: 0.4
 *
 * RP3:
 *  - AdiabaticIndex: 1.66666666666666666
 *  - LeftDensity: 1.0
 *  - LeftPressure: 1000.0
 *  - LeftVelocity: [0.0, 0.0, 0.0]
 *  - LeftMagneticField: [10.0, 7.0, 7.0]
 *  - RightDensity: 1.0
 *  - RightPressure: 0.1
 *  - RightVelocity: [0.0, 0.0, 0.0]
 *  - RightMagneticField: [10.0, 0.7, 0.7]
 *  - Lapse: 1.0
 *  - ShiftX: 0.0
 *  - Final time: 0.4
 *
 * RP4:
 *  - AdiabaticIndex: 1.66666666666666666
 *  - LeftDensity: 1.0
 *  - LeftPressure: 0.1
 *  - LeftVelocity: [0.999, 0.0, 0.0]
 *  - LeftMagneticField: [10.0, 7.0, 7.0]
 *  - RightDensity: 1.0
 *  - RightPressure: 0.1
 *  - RightVelocity: [-0.999, 0.0, 0.0]
 *  - RightMagneticField: [10.0, -7.0, -7.0]
 *  - Lapse: 1.0
 *  - ShiftX: 0.0
 *  - Final time: 0.4
 *
 * RP5:
 *  - AdiabaticIndex: 1.66666666666666666
 *  - LeftDensity: 1.08
 *  - LeftPressure: 0.95
 *  - LeftVelocity: [0.4, 0.3, 0.2]
 *  - LeftMagneticField: [2.0, 0.3, 0.3]
 *  - RightDensity: 1.0
 *  - RightPressure: 1.0
 *  - RightVelocity: [-0.45, -0.2, 0.2]
 *  - RightMagneticField: [2.0, -0.7, 0.5]
 *  - Lapse: 1.0
 *  - ShiftX: 0.0
 *  - Final time: 0.55
 */
class RiemannProblem : public MarkAsAnalyticData {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;

  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {
        "The adiabatic index of the ideal fluid"};
    static type lower_bound() noexcept { return 1.0; }
  };
  struct LeftRestMassDensity {
    using type = double;
    static std::string name() noexcept { return "LeftDensity"; };
    static constexpr Options::String help = {
        "Fluid rest mass density in the left half-domain"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct RightRestMassDensity {
    using type = double;
    static std::string name() noexcept { return "RightDensity"; };
    static constexpr Options::String help = {
        "Fluid rest mass density in the right half-domain"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct LeftPressure {
    using type = double;
    static constexpr Options::String help = {
        "Fluid pressure in the left half-domain"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct RightPressure {
    using type = double;
    static constexpr Options::String help = {
        "Fluid pressure in the right half-domain"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct LeftSpatialVelocity {
    using type = std::array<double, 3>;
    static std::string name() noexcept { return "LeftVelocity"; };
    static constexpr Options::String help = {
        "Fluid spatial velocity in the left half-domain"};
  };
  struct RightSpatialVelocity {
    using type = std::array<double, 3>;
    static std::string name() noexcept { return "RightVelocity"; };
    static constexpr Options::String help = {
        "Fluid spatial velocity in the right half-domain"};
  };
  struct LeftMagneticField {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Magnetic field in the left half-domain"};
  };
  struct RightMagneticField {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Magnetic field in the right half-domain"};
  };
  struct Lapse {
    using type = double;
    static constexpr Options::String help = {
        "The value of the lapse. Standard is 1."};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct ShiftX {
    using type = double;
    static constexpr Options::String help = {
        "The value of the x-component of the shift, beta^x. Standard is 0."};
  };

  using options =
      tmpl::list<AdiabaticIndex, LeftRestMassDensity, RightRestMassDensity,
                 LeftPressure, RightPressure, LeftSpatialVelocity,
                 RightSpatialVelocity, LeftMagneticField, RightMagneticField,
                 Lapse, ShiftX>;

  static constexpr Options::String help = {
      "Analytic initial data for a GRMHD Riemann problem. The fluid variables "
      "are set homogeneously on either half of the domain left and right of "
      "x=0."};

  RiemannProblem() = default;
  RiemannProblem(const RiemannProblem& /*rhs*/) = delete;
  RiemannProblem& operator=(const RiemannProblem& /*rhs*/) = delete;
  RiemannProblem(RiemannProblem&& /*rhs*/) noexcept = default;
  RiemannProblem& operator=(RiemannProblem&& /*rhs*/) noexcept = default;
  ~RiemannProblem() = default;

  RiemannProblem(double adiabatic_index, double left_rest_mass_density,
                 double right_rest_mass_density, double left_pressure,
                 double right_pressure,
                 const std::array<double, 3>& left_spatial_velocity,
                 const std::array<double, 3>& right_spatial_velocity,
                 const std::array<double, 3>& left_magnetic_field,
                 const std::array<double, 3>& right_magnetic_field,
                 double lapse, double shift) noexcept;

  explicit RiemannProblem(CkMigrateMessage* /*unused*/) noexcept;

  // @{
  /// Retrieve the GRMHD variables at a given position.
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<gr::Tags::Lapse<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<gr::Tags::Shift<3, Frame::Inertial, DataType>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<gr::Tags::Shift<3, Frame::Inertial, DataType>>;
  // @}

  /// Retrieve a collection of hydrodynamic variables at position x
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const noexcept {
    return background_spacetime_.variables(x, 0.0, tmpl::list<Tag>{});
  }

  const EquationsOfState::IdealFluid<true>& equation_of_state() const noexcept {
    return equation_of_state_;
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

 private:
  friend bool operator==(const RiemannProblem& lhs,
                         const RiemannProblem& rhs) noexcept;

  friend bool operator!=(const RiemannProblem& lhs,
                         const RiemannProblem& rhs) noexcept;

  EquationsOfState::IdealFluid<true> equation_of_state_{};
  gr::Solutions::Minkowski<3> background_spacetime_{};

  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  double left_rest_mass_density_ = std::numeric_limits<double>::signaling_NaN();
  double right_rest_mass_density_ =
      std::numeric_limits<double>::signaling_NaN();
  double left_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double right_pressure_ = std::numeric_limits<double>::signaling_NaN();
  static constexpr double discontinuity_location_ = 0.0;
  std::array<double, 3> left_spatial_velocity_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  std::array<double, 3> right_spatial_velocity_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  std::array<double, 3> left_magnetic_field_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  std::array<double, 3> right_magnetic_field_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  double lapse_ = std::numeric_limits<double>::signaling_NaN();
  double shift_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace grmhd::AnalyticData
