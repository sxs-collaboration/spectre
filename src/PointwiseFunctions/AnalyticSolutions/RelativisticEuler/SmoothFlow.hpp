// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace RelativisticEuler {
namespace Solutions {

/*!
 * \brief Smooth wave propagating in Minkowski spacetime.
 *
 * The relativistic Euler equations in Minkowski spacetime accept a
 * solution with constant pressure and uniform spatial velocity provided
 * that the rest mass density satisfies the advection equation
 *
 * \f{align*}
 * \partial_t\rho + v^i\partial_i\rho = 0,
 * f}
 *
 * and the specific internal energy is a function of the rest mass density only,
 * \f$\epsilon = \epsilon(\rho)\f$. For testing purposes, this class implements
 * this solution for the case where \f$\rho\f$ is a sine wave. The user
 * specifies the mean flow velocity of the fluid, the wavevector of the density
 * profile, and the amplitude \f$A\f$ of the density profile. In Cartesian
 * coordinates \f$(x, y, z)\f$, and using dimensionless units, the primitive
 * variables at a given time \f$t\f$ are then
 *
 * \f{align*}
 * \rho(\vec{x},t) &= 1 + A \sin(\vec{k}\cdot(\vec{x} - \vec{v}t)) \\
 * \vec{v}(\vec{x},t) &= [v_x, v_y, v_z]^{T},\\
 * P(\vec{x},t) &= P, \\
 * \epsilon(\vec{x}, t) &= \frac{P}{(\gamma - 1)\rho}\\
 * \f}
 *
 * where we have assumed \f$\epsilon\f$ and \f$\rho\f$ to be related through an
 * equation mathematically equivalent to the equation of state of an ideal gas,
 * where the pressure is held constant.
 */
template <size_t Dim>
class SmoothFlow : virtual public MarkAsAnalyticSolution {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;

  /// The mean flow velocity.
  struct MeanVelocity {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = {"The mean flow velocity."};
  };

  /// The wave vector of the profile.
  struct WaveVector {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = {"The wave vector of the profile."};
  };

  /// The constant pressure throughout the fluid.
  struct Pressure {
    using type = double;
    static constexpr Options::String help = {
        "The constant pressure throughout the fluid."};
    static type lower_bound() noexcept { return 0.0; }
  };

  /// The adiabatic index for the ideal fluid.
  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {
        "The adiabatic index for the ideal fluid."};
    static type lower_bound() noexcept { return 1.0; }
  };

  /// The perturbation amplitude of the rest mass density of the fluid.
  struct PerturbationSize {
    using type = double;
    static constexpr Options::String help = {
        "The perturbation size of the rest mass density."};
    static type lower_bound() noexcept { return -1.0; }
    static type upper_bound() noexcept { return 1.0; }
  };

  using options = tmpl::list<MeanVelocity, WaveVector, Pressure, AdiabaticIndex,
                             PerturbationSize>;
  static constexpr Options::String help = {
      "Smooth flow in Minkowski spacetime."};

  SmoothFlow() = default;
  SmoothFlow(const SmoothFlow& /*rhs*/) = delete;
  SmoothFlow& operator=(const SmoothFlow& /*rhs*/) = delete;
  SmoothFlow(SmoothFlow&& /*rhs*/) noexcept = default;
  SmoothFlow& operator=(SmoothFlow&& /*rhs*/) noexcept = default;
  ~SmoothFlow() = default;

  SmoothFlow(const std::array<double, Dim>& mean_velocity,
             const std::array<double, Dim>& wavevector, double pressure,
             double adiabatic_index, double perturbation_size) noexcept;

  explicit SmoothFlow(CkMigrateMessage* /*unused*/) noexcept {}

  // @{
  /// Retrieve hydro variable at `(x, t)`
  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, Dim>& x, double t,
      tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, Dim>& x, double t,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<hydro::Tags::SpatialVelocity<DataType, Dim>> /*meta*/) const
      noexcept
      -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, Dim>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, Dim>& x, double t,
      tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const
      noexcept -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;
  // @}

  /// Retrieve a collection of hydro variables at `(x, t)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, Dim>& x,
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
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, Dim>& x, double t,
                                     tmpl::list<Tag> /*meta*/) const noexcept {
    return background_spacetime_.variables(x, t, tmpl::list<Tag>{});
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

  const EquationsOfState::IdealFluid<true>& equation_of_state() const noexcept {
    return equation_of_state_;
  }

 private:
  template <size_t SpatialDim>
  friend bool
  operator==(  // NOLINT (clang-tidy: readability-redundant-declaration)
      const SmoothFlow<SpatialDim>& lhs,
      const SmoothFlow<SpatialDim>& rhs) noexcept;

  // Computes the phase.
  template <typename DataType>
  DataType k_dot_x_minus_vt(const tnsr::I<DataType, Dim>& x, double t) const
      noexcept;

  std::array<double, Dim> mean_velocity_ =
      make_array<Dim>(std::numeric_limits<double>::signaling_NaN());
  std::array<double, Dim> wavevector_ =
      make_array<Dim>(std::numeric_limits<double>::signaling_NaN());
  double pressure_ = std::numeric_limits<double>::signaling_NaN();
  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  double perturbation_size_ = std::numeric_limits<double>::signaling_NaN();
  // The angular frequency.
  double k_dot_v_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::IdealFluid<true> equation_of_state_{};
  gr::Solutions::Minkowski<Dim> background_spacetime_{};
};

template <size_t Dim>
bool operator!=(const SmoothFlow<Dim>& lhs,
                const SmoothFlow<Dim>& rhs) noexcept;
}  // namespace Solutions
}  // namespace RelativisticEuler
