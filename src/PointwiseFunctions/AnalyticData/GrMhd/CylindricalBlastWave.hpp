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
namespace AnalyticData {

/*!
 * \brief Analytic initial data for a cylindrical blast wave.
 *
 * This class implements analytic initial data for a cylindrical blast wave,
 * as described, e.g., in \cite Kidder2016hev Sec. 6.2.3.
 * A uniform magnetic field threads an ideal fluid. The solution begins with
 * material at fixed (typically high) density and pressure at rest inside a
 * cylinder of radius \f$r < r_{\rm in}\f$ and material at fixed (typically low)
 * density and pressure at rest in a cylindrical shell with radius
 * \f$r > r_{\rm out}\f$. In the region \f$ r_{\rm in} < r < r_{\rm out}\f$,
 * the solution transitions such that the logarithms of the density and
 * pressure vary linearly. E.g., if \f$\rho(r < r_{\rm in}) = \rho_{\rm in}\f$
 * and \f$\rho(r > r_{\rm out}) = \rho_{\rm out}\f$, then
 * \f[
 * \log \rho = [(r_{\rm in} - r) \log(\rho_{\rm out})
 *              + (r - r_{\rm out}) \log(\rho_{\rm in})]
 *              / (r_{\rm in} - r_{\rm out}).
 * \f]
 * Note that the cylinder's axis is the \f$z\f$ axis. To evolve this analytic
 * initial data, use a cubic or cylindrical domain with periodic boundary
 * conditions applied to the outer boundaries whose normals are parallel or
 * antiparallel to the z axis. In the transverse (e.g., x and y) dimensions, the
 * domain should be large enough that the blast wave doesn't reach the boundary
 * at the final time. E.g., if `InnerRadius = 0.8`, `OuterRadius = 1.0`, and
 * the final time is 4.0, a good domain extends from `(x,y)=(-6.0, -6.0)` to
 * `(x,y)=(6.0, 6.0)`.
 */
class CylindricalBlastWave {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;
  using background_spacetime_type = gr::Solutions::Minkowski<3>;

  /// Inside InnerRadius, density is InnerDensity.
  struct InnerRadius {
    using type = double;
    static constexpr OptionString help = {
        "Inside InnerRadius, density is InnerDensity."};
    static type lower_bound() noexcept { return 0.0; }
  };
  /// Outside OuterRadius, density is OuterDensity.
  struct OuterRadius {
    using type = double;
    static constexpr OptionString help = {
        "Outside OuterRadius, density is OuterDensity."};
    static type lower_bound() noexcept { return 0.0; }
  };
  /// Density at radii less than InnerRadius.
  struct InnerDensity {
    using type = double;
    static constexpr OptionString help = {
        "Density at radii less than InnerRadius."};
    static type lower_bound() noexcept { return 0.0; }
  };
  /// Density at radii greater than OuterRadius.
  struct OuterDensity {
    using type = double;
    static constexpr OptionString help = {
        "Density at radii greater than OuterRadius."};
    static type lower_bound() noexcept { return 0.0; }
  };
  /// Pressure at radii less than InnerRadius.
  struct InnerPressure {
    using type = double;
    static constexpr OptionString help = {
        "Pressure at radii less than InnerRadius."};
    static type lower_bound() noexcept { return 0.0; }
  };
  /// Pressure at radii greater than OuterRadius.
  struct OuterPressure {
    using type = double;
    static constexpr OptionString help = {
        "Pressure at radii greater than OuterRadius."};
    static type lower_bound() noexcept { return 0.0; }
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
      tmpl::list<InnerRadius, OuterRadius, InnerDensity, OuterDensity,
                 InnerPressure, OuterPressure, MagneticField, AdiabaticIndex>;

  static constexpr OptionString help = {
      "Cylindrical blast wave analytic initial data."};

  CylindricalBlastWave() = default;
  CylindricalBlastWave(const CylindricalBlastWave& /*rhs*/) = delete;
  CylindricalBlastWave& operator=(const CylindricalBlastWave& /*rhs*/) = delete;
  CylindricalBlastWave(CylindricalBlastWave&& /*rhs*/) noexcept = default;
  CylindricalBlastWave& operator=(CylindricalBlastWave&& /*rhs*/) noexcept =
      default;
  ~CylindricalBlastWave() = default;

  CylindricalBlastWave(
      InnerRadius::type inner_radius, OuterRadius::type outer_radius,
      InnerDensity::type inner_density, OuterDensity::type outer_density,
      InnerPressure::type inner_pressure, OuterPressure::type outer_pressure,
      MagneticField::type magnetic_field, AdiabaticIndex::type adiabatic_index,
      const OptionContext& context = {});

  explicit CylindricalBlastWave(CkMigrateMessage* /*unused*/) noexcept {}

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
                 tmpl::list<hydro::Tags::SpatialVelocity<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::MagneticField<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>;

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
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
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
    constexpr double dummy_time = 0.0;
    return background_spacetime_.variables(x, dummy_time, tmpl::list<Tag>{});
  }

  const EquationsOfState::IdealFluid<true>& equation_of_state() const noexcept {
    return equation_of_state_;
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

 private:
  InnerRadius::type inner_radius_ =
      std::numeric_limits<double>::signaling_NaN();
  OuterRadius::type outer_radius_ =
      std::numeric_limits<double>::signaling_NaN();
  InnerDensity::type inner_density_ =
      std::numeric_limits<double>::signaling_NaN();
  OuterDensity::type outer_density_ =
      std::numeric_limits<double>::signaling_NaN();
  InnerPressure::type inner_pressure_ =
      std::numeric_limits<double>::signaling_NaN();
  OuterPressure::type outer_pressure_ =
      std::numeric_limits<double>::signaling_NaN();
  MagneticField::type magnetic_field_ =
      std::array<double, 3>{{std::numeric_limits<double>::signaling_NaN(),
                             std::numeric_limits<double>::signaling_NaN(),
                             std::numeric_limits<double>::signaling_NaN()}};
  AdiabaticIndex::type adiabatic_index_ =
      std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::IdealFluid<true> equation_of_state_{};
  gr::Solutions::Minkowski<3> background_spacetime_{};

  friend bool operator==(const CylindricalBlastWave& lhs,
                         const CylindricalBlastWave& rhs) noexcept;

  friend bool operator!=(const CylindricalBlastWave& lhs,
                         const CylindricalBlastWave& rhs) noexcept;
};

}  // namespace AnalyticData
}  // namespace grmhd
