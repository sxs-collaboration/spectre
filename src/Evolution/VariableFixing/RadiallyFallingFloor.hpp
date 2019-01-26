// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace PUP {
class er;
}  // namespace PUP
class DataVector;
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace Tags
namespace hydro {
namespace Tags {
template <typename DataType>
struct Pressure;
template <typename DataType>
struct RestMassDensity;
}  // namespace Tags
}  // namespace hydro
/// \endcond

/// Contains all variable fixers.
namespace VariableFixing {

/// \ingroup VariableFixingGroup
/// \brief Applies a pressure and density floor dependent on the distance
/// to the origin.
///
/// Applies the floors:
/// \f$\rho(r) \geq \rho_{\mathrm{fl}}(r) = C_\rho r^{k_\rho}\f$
/// and \f$P(r) \geq P_{\mathrm{fl}}(r) = C_p r^{k_p}\f$
/// when \f$ r > r_{min}\f$, where \f$C_\rho\f$ is given by the option
/// `ScaleDensityFloor`, \f$k_\rho\f$ is given by the option
/// `PowerDensityFloor`,  \f$C_p\f$ is given by the option
/// `ScalePressureFloor`, \f$k_p\f$ is given by the option
/// `PowerPressureFloor`, and \f$r_{min}\f$ is given by the option
/// `MinimumRadius`.
///
/// \note In \cite Porth2016rfi, the following floors are applied:
/// \f$\rho(r) \geq \rho_{\mathrm{fl}}(r) = 10^{-5}r^{-3/2}\f$
/// and \f$P(r) \geq P_{\mathrm{fl}}(r) = \frac{1}{3} \times 10^{-7}r^{-5/2}\f$
template <size_t Dim>
class RadiallyFallingFloor {
 public:
  /// \brief The minimum radius at which to begin applying the floors on the
  /// density and pressure.
  struct MinimumRadius {
    static constexpr OptionString help =
        "The radius at which to begin applying the lower bound.";
    using type = double;
    static double lower_bound() noexcept { return 0.0; }
  };

  /// \brief The scale of the floor of the rest mass density.
  struct ScaleDensityFloor {
    static constexpr OptionString help =
        "The rest mass density floor at r = 1.";
    using type = double;
    static double lower_bound() noexcept { return 0.0; }
  };

  /// \brief The power of the radius of the floor of the rest mass density.
  struct PowerDensityFloor {
    static constexpr OptionString help =
        "Radial power for the floor of the rest mass density.";
    using type = double;
  };

  /// \brief The scale of the floor of the pressure.
  struct ScalePressureFloor {
    static constexpr OptionString help = "The pressure floor at r = 1.";
    using type = double;
    static double lower_bound() noexcept { return 0.0; }
  };

  /// \brief The power of the radius of the floor of the pressure.
  struct PowerPressureFloor {
    static constexpr OptionString help =
        "The radial power for the floor of the pressure.";
    using type = double;
  };

  using options =
      tmpl::list<MinimumRadius, ScaleDensityFloor, PowerDensityFloor,
                 ScalePressureFloor, PowerPressureFloor>;
  static constexpr OptionString help = {
      "Applies a pressure and density floor dependent on the distance to the "
      "origin."};

  RadiallyFallingFloor(double minimum_radius_at_which_to_apply_floor,
                       double rest_mass_density_scale,
                       double rest_mass_density_power, double pressure_scale,
                       double pressure_power) noexcept;

  RadiallyFallingFloor() noexcept = default;
  RadiallyFallingFloor(const RadiallyFallingFloor& /*rhs*/) = default;
  RadiallyFallingFloor& operator=(const RadiallyFallingFloor& /*rhs*/) =
      default;
  RadiallyFallingFloor(RadiallyFallingFloor&& /*rhs*/) noexcept = default;
  RadiallyFallingFloor& operator=(RadiallyFallingFloor&& /*rhs*/) noexcept =
      default;
  ~RadiallyFallingFloor() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

  using return_tags = tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                                 hydro::Tags::Pressure<DataVector>>;
  using argument_tags = tmpl::list<::Tags::Coordinates<Dim, Frame::Inertial>>;

  void operator()(gsl::not_null<Scalar<DataVector>*> density,
                  gsl::not_null<Scalar<DataVector>*> pressure,
                  const tnsr::I<DataVector, Dim, Frame::Inertial>& coords) const
      noexcept;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const RadiallyFallingFloor<LocalDim>& lhs,
                         const RadiallyFallingFloor<LocalDim>& rhs) noexcept;

  double minimum_radius_at_which_to_apply_floor_{
      std::numeric_limits<double>::signaling_NaN()};
  double rest_mass_density_scale_{std::numeric_limits<double>::signaling_NaN()};
  double rest_mass_density_power_{std::numeric_limits<double>::signaling_NaN()};
  double pressure_scale_{std::numeric_limits<double>::signaling_NaN()};
  double pressure_power_{std::numeric_limits<double>::signaling_NaN()};
};

template <size_t Dim>
bool operator!=(const RadiallyFallingFloor<Dim>& lhs,
                const RadiallyFallingFloor<Dim>& rhs) noexcept;

}  // namespace VariableFixing
