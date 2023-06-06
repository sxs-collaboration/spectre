// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::AnalyticData {

/*!
 * \brief Analytic initial data for a slab jet
 *
 * This test problem is described in \cite Komissarov1999, Section 7.4 and
 * Fig. 13. It involves a high Lorentz factor jet injected into an ambient fluid
 * with an initially uniform magnetic field.
 *
 * The parameters used in \cite Komissarov1999 are:
 *
 * ```yaml
 * AdiabaticIndex: 4. / 3.
 * AmbientDensity: 10.
 * AmbientPressure: 0.01
 * JetDensity: 0.1
 * JetPressure: 0.01
 * # u^i = [20, 0, 0] or W = sqrt(401)
 * JetVelocity: [0.9987523388778445, 0., 0.]
 * InletRadius: 1.
 * MagneticField: [1., 0., 0.]
 * ```
 *
 * In \cite Komissarov1999 an artificial dissipation of $\eta_u=0.2$ and
 * $\eta_b=0.15$ is used, which we don't use here (yet).
 *
 * \note The inlet is currently modeled as the region of the domain where
 * $x <= 0$ and $|y| <= \mathrm{inlet_radius}$. Since this class only sets
 * the initial data, no fluid will flow into the domain during the evolution.
 * This has to be modeled as a boundary condition and is not implemented yet.
 */
class SlabJet : public evolution::initial_data::InitialData,
                public MarkAsAnalyticData,
                public AnalyticDataBase {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;

  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {
        "The adiabatic index of the ideal fluid"};
    static double lower_bound() { return 1.; }
  };
  struct AmbientDensity {
    using type = double;
    static constexpr Options::String help = {
        "Fluid rest mass density outside the jet"};
    static double lower_bound() { return 0.; }
    static double suggested_value() { return 10.; }
  };
  struct AmbientPressure {
    using type = double;
    static constexpr Options::String help = {"Fluid pressure outside the jet"};
    static double lower_bound() { return 0.; }
    static double suggested_value() { return 0.01; }
  };
  struct AmbientElectronFraction {
    using type = double;
    static constexpr Options::String help = {
        "Electron fraction outside the jet"};
    static double lower_bound() { return 0.; }
    static double upper_bound() { return 1.; }
  };
  struct JetDensity {
    using type = double;
    static constexpr Options::String help = {
        "Fluid rest mass density of the jet inlet"};
    static double lower_bound() { return 0.; }
    static double suggested_value() { return 0.1; }
  };
  struct JetPressure {
    using type = double;
    static constexpr Options::String help = {"Fluid pressure of the jet inlet"};
    static double lower_bound() { return 0.; }
    static double suggested_value() { return 0.01; }
  };
  struct JetElectronFraction {
    using type = double;
    static constexpr Options::String help = {
        "Electron fraction of the jet inlet"};
    static double lower_bound() { return 0.; }
    static double upper_bound() { return 1.; }
  };
  struct JetVelocity {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Fluid spatial velocity of the jet inlet"};
  };
  struct InletRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of the jet inlet around y=0"};
    static double lower_bound() { return 0.; }
    static double suggested_value() { return 1.; }
  };
  struct MagneticField {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Initially uniform magnetic field"};
    static std::array<double, 3> suggested_value() { return {{1., 0., 0.}}; }
  };

  using options =
      tmpl::list<AdiabaticIndex, AmbientDensity, AmbientPressure,
                 AmbientElectronFraction, JetDensity, JetPressure,
                 JetElectronFraction, JetVelocity, InletRadius, MagneticField>;

  static constexpr Options::String help = {
      "Analytic initial data for a jet test."};

  SlabJet() = default;
  SlabJet(const SlabJet& /*rhs*/) = default;
  SlabJet& operator=(const SlabJet& /*rhs*/) = default;
  SlabJet(SlabJet&& /*rhs*/) = default;
  SlabJet& operator=(SlabJet&& /*rhs*/) = default;
  ~SlabJet() = default;

  SlabJet(double adiabatic_index, double ambient_density,
          double ambient_pressure, double ambient_electron_fraction,
          double jet_density, double jet_pressure, double jet_electron_fraction,
          std::array<double, 3> jet_velocity, double inlet_radius,
          std::array<double, 3> magnetic_field);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit SlabJet(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SlabJet);
  /// \endcond

  /// @{
  /// Retrieve the GRMHD variables at a given position.
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;
  /// @}

  /// Retrieve a collection of hydrodynamic variables at position x
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         tmpl::list<Tags...> /*meta*/) const {
    static_assert(sizeof...(Tags) > 1, "The requested tag is not implemented.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const {
    return background_spacetime_.variables(
        x, std::numeric_limits<double>::signaling_NaN(), tmpl::list<Tag>{});
  }

  const EquationsOfState::IdealFluid<true>& equation_of_state() const {
    return equation_of_state_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) override;

 private:
  EquationsOfState::IdealFluid<true> equation_of_state_{};
  gr::Solutions::Minkowski<3> background_spacetime_{};

  double ambient_density_ = std::numeric_limits<double>::signaling_NaN();
  double ambient_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double ambient_electron_fraction_ =
      std::numeric_limits<double>::signaling_NaN();
  double jet_density_ = std::numeric_limits<double>::signaling_NaN();
  double jet_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double jet_electron_fraction_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> jet_velocity_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  double inlet_radius_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> magnetic_field_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};

  friend bool operator==(const SlabJet& lhs, const SlabJet& rhs);

  friend bool operator!=(const SlabJet& lhs, const SlabJet& rhs);
};

}  // namespace grmhd::AnalyticData
