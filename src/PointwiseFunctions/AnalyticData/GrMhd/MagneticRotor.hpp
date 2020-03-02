// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma:  no_include <pup.h>

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace grmhd {
namespace AnalyticData {

/*!
 * \brief Analytic initial data for a magnetic rotor.
 *
 * This is a test first described in \cite Balsara1999 for classical MHD and
 * later generalised to relativistic MHD in \cite DelZanna2002rv
 *
 * This effectively 2D test initially consists of an infinitely long cylinder of
 * radius `RotorRadius` rotating about the z-axis with the given
 * `AngularVelocity`. The rest mass density of the fluid inside the rotor,
 * `RotorDensity`, is higher than the `BackgroundDensity` outside of the rotor.
 * The fluid is at a constant `Pressure`.  The rotor is embedded in a constant
 * `MagneticField` (usually taken to be along the x-axis).  The fluid is an
 * ideal fluid with the given `AdiabaticIndex`.  Evolving the initial data,
 * magnetic braking will slow down the rotor, while dragging the magnetic field
 * lines.
 *
 * The standard test setup is done on a unit cube with the following values
 * given for the options:
 * -  RotorRadius: 0.1
 * -  RotorDensity: 10.0
 * -  BackgroundDensity: 1.0
 * -  Pressure: 1.0
 * -  AngularVelocity: 9.95
 * -  MagneticField: [3.54490770181103205, 0.0, 0.0]
 * -  AdiabaticIndex: 1.66666666666666667
 *
 * The magnetic field in the disk should rotate by about 90 degrees by t = 0.4.
 *
 */
class MagneticRotor : public MarkAsAnalyticData {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;

  /// Radius of the rotor.
  struct RotorRadius {
    using type = double;
    static constexpr OptionString help = {"The initial radius of the rotor."};
    static type lower_bound() noexcept { return 0.0; }
  };
  /// Density inside the rotor.
  struct RotorDensity {
    using type = double;
    static constexpr OptionString help = {"Density inside RotorRadius."};
    static type lower_bound() noexcept { return 0.0; }
  };
  /// Density outside the rotor.
  struct BackgroundDensity {
    using type = double;
    static constexpr OptionString help = {"Density outside RotorRadius."};
    static type lower_bound() noexcept { return 0.0; }
  };
  /// Uniform pressure inside and outside the rotor.
  struct Pressure {
    using type = double;
    static constexpr OptionString help = {"Pressure."};
    static type lower_bound() noexcept { return 0.0; }
  };
  /// Angular velocity inside the rotor.
  struct AngularVelocity {
    using type = double;
    static constexpr OptionString help = {
        "Angular velocity of matter inside RotorRadius"};
  };
  /// The x,y,z components of the uniform magnetic field threading the matter.
  struct MagneticField {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {
        "The x,y,z components of the uniform magnetic field."};
  };
  /// The adiabatic index of the ideal fluid.
  struct AdiabaticIndex {
    using type = double;
    static constexpr OptionString help = {
        "The adiabatic index of the ideal fluid."};
    static type lower_bound() noexcept { return 1.0; }
  };

  using options =
      tmpl::list<RotorRadius, RotorDensity, BackgroundDensity, Pressure,
                 AngularVelocity, MagneticField, AdiabaticIndex>;

  static constexpr OptionString help = {
      "Magnetic rotor analytic initial data."};

  MagneticRotor() = default;
  MagneticRotor(const MagneticRotor& /*rhs*/) = delete;
  MagneticRotor& operator=(const MagneticRotor& /*rhs*/) = delete;
  MagneticRotor(MagneticRotor&& /*rhs*/) noexcept = default;
  MagneticRotor& operator=(MagneticRotor&& /*rhs*/) noexcept = default;
  ~MagneticRotor() = default;

  MagneticRotor(double rotor_radius, double rotor_density,
                double background_density, double pressure,
                double angular_velocity, std::array<double, 3> magnetic_field,
                double adiabatic_index, const OptionContext& context = {});

  explicit MagneticRotor(CkMigrateMessage* /*unused*/) noexcept {}

  // @{
  /// Retrieve the GRMHD variables at a given position.
  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;
  // @}

  /// Retrieve a collection of hydrodynamic variables at position x
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const noexcept {
    constexpr double dummy_time = 0.0;
    return background_spacetime_.variables(x, dummy_time, tmpl::list<Tag>{});
  }

  const EquationsOfState::IdealFluid<true>& equation_of_state() const noexcept {
    return equation_of_state_;
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

 private:
  double rotor_radius_ = std::numeric_limits<double>::signaling_NaN();
  double rotor_density_ = std::numeric_limits<double>::signaling_NaN();
  double background_density_ = std::numeric_limits<double>::signaling_NaN();
  double pressure_ = std::numeric_limits<double>::signaling_NaN();
  double angular_velocity_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> magnetic_field_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::IdealFluid<true> equation_of_state_{};
  gr::Solutions::Minkowski<3> background_spacetime_{};

  friend bool operator==(const MagneticRotor& lhs,
                         const MagneticRotor& rhs) noexcept;

  friend bool operator!=(const MagneticRotor& lhs,
                         const MagneticRotor& rhs) noexcept;
};

}  // namespace AnalyticData
}  // namespace grmhd
