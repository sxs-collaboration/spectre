// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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
 * \brief Analytic initial data for a Kelvin-Helmholtz instability simulation.
 *
 * This is similar to the data from Section 4.7 of \cite Beckwith2011iy.
 * The initial state consists of a horizontal strip of mass
 * density \f$\rho_\text{in}\f$ moving with horizontal speed
 * \f$v_{\text{in}}\f$. The rest of the fluid possesses mass density
 * \f$\rho_\text{out}\f$, and its horizontal velocity is \f$v_{\text{out}}\f$,
 * both constant. Mathematically,
 *
 * \f{align*}
 * \rho(x, y) =
 * \begin{cases}
 * \rho_\text{in}, & \left|y - y_\text{mid}\right| < b/2\\
 * \rho_\text{out}, & \text{otherwise},
 * \end{cases}
 * \f}
 *
 * and
 *
 * \f{align*}
 * v_x(x, y) =
 * \begin{cases}
 * v_{\text{in}}, & \left|y - y_\text{mid}\right| < b/2\\
 * v_{\text{out}}, & \text{otherwise},
 * \end{cases}
 * \f}
 *
 * where \f$b > 0\f$ is the thickness of the strip, and \f$y = y_\text{mid}\f$
 * is its horizontal bimedian. The initial pressure is set equal to a constant,
 * and the system is evolved assuming an ideal fluid of known adiabatic index.
 * Finally, in order to excite the instability, the vertical velocity is
 * initialized to
 *
 * \f{align*}
 * v_y(x, y) = A\sin(4\pi x)
 * \left[\exp\left(-\dfrac{(y - y_\text{top})^2}{2\sigma^2}\right) +
 * \exp\left(-\dfrac{(y - y_\text{bot})^2}{2\sigma^2}\right)\right],
 * \f}
 *
 * whose net effect is to perturb the horizontal boundaries of the strip
 * periodically along the \f$x-\f$axis. Here \f$A\f$ is the amplitude,
 * \f$\sigma\f$ is a characteristic length for the perturbation width,
 * and \f$y_\text{top} = y_\text{mid} + b/2\f$ and
 * \f$y_\text{bot} = y_\text{mid} - b/2\f$ are the vertical coordinates
 * of the top and bottom boundaries of the strip, respectively.
 *
 * A uniform magnetic field can be added.
 */
class KhInstability : public evolution::initial_data::InitialData,
                      public MarkAsAnalyticData,
                      public AnalyticDataBase {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;

  /// The adiabatic index of the fluid.
  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {
        "The adiabatic index of the fluid."};
  };

  /// The vertical coordinate of the horizontal bimedian of the strip.
  struct StripBimedianHeight {
    using type = double;
    static constexpr Options::String help = {"The height of the strip center."};
  };

  /// The thickness of the strip.
  struct StripThickness {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The thickness of the horizontal strip."};
  };

  /// The mass density in the strip
  struct StripDensity {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The mass density in the horizontal strip."};
  };

  /// The velocity along \f$x\f$ in the strip
  struct StripVelocity {
    using type = double;
    static constexpr Options::String help = {
        "The velocity along x in the horizontal strip."};
  };

  /// The mass density outside of the strip
  struct BackgroundDensity {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The mass density outside of the strip."};
  };

  /// The velocity along \f$x\f$ outside of the strip
  struct BackgroundVelocity {
    using type = double;
    static constexpr Options::String help = {
        "The velocity along x outside of the strip."};
  };

  /// The initial (constant) pressure of the fluid
  struct Pressure {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The initial (constant) pressure."};
  };

  /// The amplitude of the perturbation
  struct PerturbAmplitude {
    using type = double;
    static constexpr Options::String help = {
        "The amplitude of the perturbation."};
  };

  /// The characteristic length for the width of the perturbation
  struct PerturbWidth {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The characteristic length for the width of the perturbation."};
  };

  /// The uniform magnetic field
  struct MagneticField {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {"The uniform magnetic field."};
  };

  using options = tmpl::list<AdiabaticIndex, StripBimedianHeight,
                             StripThickness, StripDensity, StripVelocity,
                             BackgroundDensity, BackgroundVelocity, Pressure,
                             PerturbAmplitude, PerturbWidth, MagneticField>;

  static constexpr Options::String help = {
      "Initial data to simulate the magnetized KH instability."};

  KhInstability() = default;
  KhInstability(const KhInstability& /*rhs*/) = default;
  KhInstability& operator=(const KhInstability& /*rhs*/) = default;
  KhInstability(KhInstability&& /*rhs*/) = default;
  KhInstability& operator=(KhInstability&& /*rhs*/) = default;
  ~KhInstability() override = default;

  KhInstability(double adiabatic_index, double strip_bimedian_height,
                double strip_thickness, double strip_density,
                double strip_velocity, double background_density,
                double background_velocity, double pressure,
                double perturbation_amplitude, double perturbation_width,
                const std::array<double, 3>& magnetic_field);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit KhInstability(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(KhInstability);
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
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const {
    constexpr double dummy_time = 0.0;
    return background_spacetime_.variables(x, dummy_time, tmpl::list<Tag>{});
  }

  const EquationsOfState::IdealFluid<true>& equation_of_state() const {
    return equation_of_state_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) override;

 private:
  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  double strip_bimedian_height_ = std::numeric_limits<double>::signaling_NaN();
  double strip_half_thickness_ = std::numeric_limits<double>::signaling_NaN();
  double strip_density_ = std::numeric_limits<double>::signaling_NaN();
  double strip_velocity_ = std::numeric_limits<double>::signaling_NaN();
  double background_density_ = std::numeric_limits<double>::signaling_NaN();
  double background_velocity_ = std::numeric_limits<double>::signaling_NaN();
  double pressure_ = std::numeric_limits<double>::signaling_NaN();
  double perturbation_amplitude_ = std::numeric_limits<double>::signaling_NaN();
  double perturbation_width_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> magnetic_field_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  EquationsOfState::IdealFluid<true> equation_of_state_{};
  gr::Solutions::Minkowski<3> background_spacetime_{};

  friend bool operator==(const KhInstability& lhs, const KhInstability& rhs);

  friend bool operator!=(const KhInstability& lhs, const KhInstability& rhs);
};
}  // namespace grmhd::AnalyticData
