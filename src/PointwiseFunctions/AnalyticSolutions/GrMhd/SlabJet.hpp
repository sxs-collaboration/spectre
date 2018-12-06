// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/MakeArray.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma:  no_include <pup.h>

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace grmhd {
namespace Solutions {

/*!
 * \brief SlabJet
 */
class SlabJet {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;
  using background_spacetime_type = gr::Solutions::Minkowski<3>;

  struct AmbientDensity {
    using type = double;
    static constexpr OptionString help = {
        "Fluid rest mass density outside the jet"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct AmbientPressure {
    using type = double;
    static constexpr OptionString help = {
        "Fluid pressure outside the jet"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct JetDensity {
    using type = double;
    static constexpr OptionString help = {
        "Fluid rest mass density of the jet inlet"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct JetPressure {
    using type = double;
    static constexpr OptionString help = {
        "Fluid pressure of the jet inlet"};
    static type lower_bound() noexcept { return 0.0; }
  };
  struct JetVelocity {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {
        "Fluid spatial velocity of the jet inlet"};
  };
  struct InletRadius {
    using type = double;
    static constexpr OptionString help = {
        "Radius of the jet inlet around y=0"};
  };
  struct MagneticField {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {
        "Initially uniform magnetic field"};
  };

  using options =
      tmpl::list<AmbientDensity, AmbientPressure, JetDensity, JetPressure,
                 JetVelocity, InletRadius, MagneticField>;

  static constexpr OptionString help = {
      "Analytic initial data for a jet test."};

  SlabJet() = default;
  SlabJet(const SlabJet& /*rhs*/) = delete;
  SlabJet& operator=(const SlabJet& /*rhs*/) = delete;
  SlabJet(SlabJet&& /*rhs*/) noexcept = default;
  SlabJet& operator=(SlabJet&& /*rhs*/) noexcept = default;
  ~SlabJet() = default;

  SlabJet(AmbientDensity::type ambient_density,
          AmbientPressure::type ambient_pressure, JetDensity::type jet_density,
          JetPressure::type jet_pressure, JetVelocity::type jet_velocity,
          InletRadius::type inlet_radius,
          MagneticField::type magnetic_field) noexcept;

  explicit SlabJet(CkMigrateMessage* /*unused*/) noexcept {}

  // @{
  /// Retrieve the GRMHD variables at a given position.
  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double t,
      tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double t,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double t,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double t,
                 tmpl::list<hydro::Tags::SpatialVelocity<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double t,
                 tmpl::list<hydro::Tags::MagneticField<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double t,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double t,
      tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double t,
      tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;
  // @}

  /// Retrieve a collection of hydrodynamic variables at position x
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x, double t,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {tuples::get<Tags>(variables(x, t, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x, double t,
                                     tmpl::list<Tag> /*meta*/) const noexcept {
    // constexpr double dummy_time = 0.0;
    return background_spacetime_.variables(x, t, tmpl::list<Tag>{});
  }

  const EquationsOfState::IdealFluid<true>& equation_of_state() const noexcept {
    return equation_of_state_;
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

 private:
  EquationsOfState::IdealFluid<true> equation_of_state_{4. / 3.};
  gr::Solutions::Minkowski<3> background_spacetime_{};

  AmbientDensity::type ambient_density_ =
      std::numeric_limits<double>::signaling_NaN();
  AmbientPressure::type ambient_pressure_ =
      std::numeric_limits<double>::signaling_NaN();
  JetDensity::type jet_density_ = std::numeric_limits<double>::signaling_NaN();
  JetPressure::type jet_pressure_ =
      std::numeric_limits<double>::signaling_NaN();
  JetVelocity::type jet_velocity_ =
      std::array<double, 3>{{std::numeric_limits<double>::signaling_NaN(),
                             std::numeric_limits<double>::signaling_NaN(),
                             std::numeric_limits<double>::signaling_NaN()}};
  InletRadius::type inlet_radius_ =
      std::numeric_limits<double>::signaling_NaN();
  MagneticField::type magnetic_field_ =
      std::array<double, 3>{{std::numeric_limits<double>::signaling_NaN(),
                             std::numeric_limits<double>::signaling_NaN(),
                             std::numeric_limits<double>::signaling_NaN()}};

  friend bool operator==(const SlabJet& lhs, const SlabJet& rhs) noexcept;

  friend bool operator!=(const SlabJet& lhs, const SlabJet& rhs) noexcept;
};

}  // namespace Solutions
}  // namespace grmhd
