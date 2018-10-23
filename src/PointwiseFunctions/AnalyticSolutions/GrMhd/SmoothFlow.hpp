// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/MakeArray.hpp"            // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace grmhd {
namespace Solutions {

/*!
 * \brief Periodic GrMhd solution in Minkowski spacetime.
 *
 * An analytic solution to the 3-D GrMhd system. The user specifies the mean
 * flow velocity of the fluid, the wavevector of the density profile, and the
 * amplitude \f$A\f$ of the density profile. The magnetic field is taken to be
 * zero everywhere. In Cartesian coordinates \f$(x, y, z)\f$, and using
 * dimensionless units, the primitive quantities at a given time \f$t\f$ are
 * then
 *
 * \f{align*}
 * \rho(\vec{x},t) &= 1 + A \sin(\vec{k}\cdot(\vec{x} - \vec{v}t)) \\
 * \vec{v}(\vec{x},t) &= [v_x, v_y, v_z]^{T},\\
 * P(\vec{x},t) &= P, \\
 * \epsilon(\vec{x}, t) &= \frac{P}{(\gamma - 1)\rho}\\
 * \vec{B}(\vec{x},t) &= [0, 0, 0]^{T}
 * \f}
 */
class SmoothFlow {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;
  using background_spacetime_type = gr::Solutions::Minkowski<3>;

  /// The mean flow velocity.
  struct MeanVelocity {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {"The mean flow velocity."};
  };

  /// The wave vector of the profile.
  struct WaveVector {
    using type = std::array<double, 3>;
    static constexpr OptionString help = {"The wave vector of the profile."};
  };

  /// The constant pressure throughout the fluid.
  struct Pressure {
    using type = double;
    static constexpr OptionString help = {
        "The constant pressure throughout the fluid."};
    static type lower_bound() noexcept { return 0.0; }
  };

  /// The adiabatic index for the ideal fluid.
  struct AdiabaticIndex {
    using type = double;
    static constexpr OptionString help = {
        "The adiabatic index for the ideal fluid."};
    static type lower_bound() noexcept { return 1.0; }
  };

  /// The perturbation amplitude of the rest mass density of the fluid.
  struct PerturbationSize {
    using type = double;
    static constexpr OptionString help = {
        "The perturbation size of the rest mass density."};
    static type lower_bound() noexcept { return -1.0; }
    static type upper_bound() noexcept { return 1.0; }
  };

  using options = tmpl::list<MeanVelocity, WaveVector, Pressure, AdiabaticIndex,
                             PerturbationSize>;
  static constexpr OptionString help = {
      "Periodic smooth flow in Minkowski spacetime with zero magnetic field."};

  SmoothFlow() = default;
  SmoothFlow(const SmoothFlow& /*rhs*/) = delete;
  SmoothFlow& operator=(const SmoothFlow& /*rhs*/) = delete;
  SmoothFlow(SmoothFlow&& /*rhs*/) noexcept = default;
  SmoothFlow& operator=(SmoothFlow&& /*rhs*/) noexcept = default;
  ~SmoothFlow() = default;

  SmoothFlow(MeanVelocity::type mean_velocity, WaveVector::type wavevector,
             Pressure::type pressure, AdiabaticIndex::type adiabatic_index,
             PerturbationSize::type perturbation_size) noexcept;

  explicit SmoothFlow(CkMigrateMessage* /*unused*/) noexcept {}

  // @{
  /// Retrieve hydro variable at `(x, t)`
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
  auto variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                 tmpl::list<hydro::Tags::SpatialVelocity<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::SpatialVelocity<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x, double /*t*/,
                 tmpl::list<hydro::Tags::MagneticField<
                     DataType, 3, Frame::Inertial>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<
          hydro::Tags::MagneticField<DataType, 3, Frame::Inertial>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double /*t*/,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double /*t*/,
      tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x, double t,
      tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;
  // @}

  /// Retrieve a collection of hydro variables at `(x, t)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         double t,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "The generic template will recurse infinitely if only one "
                  "tag is being retrieved.");
    return {get<Tags>(variables(x, t, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename DataType, typename Tag>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x, double t,
                                     tmpl::list<Tag> /*meta*/) const noexcept {
    return background_spacetime_.variables(x, t, tmpl::list<Tag>{});
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

  const EquationsOfState::IdealFluid<true>& equation_of_state() const noexcept {
    return equation_of_state_;
  }

 private:
  friend bool operator==(const SmoothFlow& lhs, const SmoothFlow& rhs) noexcept;

  // Computes the phase.
  template <typename DataType>
  DataType k_dot_x_minus_vt(const tnsr::I<DataType, 3>& x, double t) const
      noexcept;
  MeanVelocity::type mean_velocity_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  WaveVector::type wavevector_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  Pressure::type pressure_ = std::numeric_limits<double>::signaling_NaN();
  AdiabaticIndex::type adiabatic_index_ =
      std::numeric_limits<double>::signaling_NaN();
  PerturbationSize::type perturbation_size_ =
      std::numeric_limits<double>::signaling_NaN();
  // The angular frequency.
  double k_dot_v_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::IdealFluid<true> equation_of_state_{};
  gr::Solutions::Minkowski<3> background_spacetime_{};
};

bool operator!=(const SmoothFlow& lhs, const SmoothFlow& rhs) noexcept;
}  // namespace Solutions
}  // namespace grmhd
