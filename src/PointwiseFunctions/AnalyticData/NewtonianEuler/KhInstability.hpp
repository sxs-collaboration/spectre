// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/NoSource.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

namespace NewtonianEuler {
namespace AnalyticData {

/*!
 * \brief Initial data to simulate the Kelvin-Helmholtz instability.
 *
 * For comparison purposes, this class implements the planar shear of
 * \cite Schaal2015, which is evolved using periodic boundary conditions.
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
 * \note The periodic form of the perturbation enforces the horizontal
 * extent of the domain to be a multiple of 0.5. The domain chosen in
 * \cite Schaal2015 is the square \f$[0, 1]^2\f$, which covers two wavelengths
 * of the perturbation. In addition, the authors ran their simulations using
 *
 * ```
 * AdiabaticIndex: 1.4
 * StripBimedianHeight: 0.5
 * StripThickness: 0.5
 * StripDensity: 2.0
 * StripVelocity: 0.5
 * BackgroundDensity: 1.0
 * BackgroundVelocity: -0.5
 * Pressure: 2.5
 * PerturbAmplitude: 0.1
 * PerturbWidth: 0.035355339059327376 # 0.05/sqrt(2)
 * ```
 *
 * \note This class can be used to initialize 2D and 3D data. In 3D, the strip
 * is aligned to the \f$xy-\f$plane, and the vertical direction is taken to be
 * the \f$z-\f$axis.
 */
template <size_t Dim>
class KhInstability : public MarkAsAnalyticData {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<false>;
  using source_term_type = Sources::NoSource;

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

  using options =
      tmpl::list<AdiabaticIndex, StripBimedianHeight, StripThickness,
                 StripDensity, StripVelocity, BackgroundDensity,
                 BackgroundVelocity, Pressure, PerturbAmplitude, PerturbWidth>;

  static constexpr Options::String help = {
      "Initial data to simulate the KH instability."};

  KhInstability() = default;
  KhInstability(const KhInstability& /*rhs*/) = delete;
  KhInstability& operator=(const KhInstability& /*rhs*/) = delete;
  KhInstability(KhInstability&& /*rhs*/) = default;
  KhInstability& operator=(KhInstability&& /*rhs*/) = default;
  ~KhInstability() = default;

  KhInstability(double adiabatic_index, double strip_bimedian_height,
                double strip_thickness, double strip_density,
                double strip_velocity, double background_density,
                double background_velocity, double pressure,
                double perturbation_amplitude, double perturbation_width);

  /// Retrieve a collection of hydrodynamic variables at position x
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, Dim, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const {
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  const EquationsOfState::IdealFluid<false>& equation_of_state() const {
    return equation_of_state_;
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/);  //  NOLINT

 private:
  /// @{
  /// Retrieve hydro variable at `x`
  template <typename DataType>
  auto variables(const tnsr::I<DataType, Dim, Frame::Inertial>& x,
                 tmpl::list<Tags::MassDensity<DataType>> /*meta*/
  ) const -> tuples::TaggedTuple<Tags::MassDensity<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, Dim, Frame::Inertial>& x,
      tmpl::list<Tags::Velocity<DataType, Dim, Frame::Inertial>> /*meta*/) const
      -> tuples::TaggedTuple<Tags::Velocity<DataType, Dim, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, Dim, Frame::Inertial>& x,
                 tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/
  ) const -> tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, Dim, Frame::Inertial>& x,
                 tmpl::list<Tags::Pressure<DataType>> /*meta*/
  ) const -> tuples::TaggedTuple<Tags::Pressure<DataType>>;
  /// @}

  template <size_t SpatialDim>
  friend bool
  operator==(  // NOLINT (clang-tidy: readability-redundant-declaration)
      const KhInstability<SpatialDim>& lhs,
      const KhInstability<SpatialDim>& rhs);

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
  EquationsOfState::IdealFluid<false> equation_of_state_{};
};

template <size_t Dim>
bool operator!=(const KhInstability<Dim>& lhs, const KhInstability<Dim>& rhs);

}  // namespace AnalyticData
}  // namespace NewtonianEuler
