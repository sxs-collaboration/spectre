// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/NoSource.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
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
namespace Solutions {

/*!
 * \brief Analytic solution to the Riemann Problem
 *
 * This class implements the exact Riemann solver described in detail in
 * Chapter 4 of \cite Toro2009. We follow the notation there.
 * The algorithm implemented here allows for 1, 2 and 3D wave propagation
 * along any coordinate axis. Typical initial data for test cases
 * (see \cite Toro2009) include:
 *
 * - Sod's Shock Tube (shock on the right, rarefaction on the left):
 *   - \f$(\rho_L, u_L, p_L) = (1.0, 0.0, 1.0)\f$
 *   - \f$(\rho_R, u_R, p_R) = (0.125, 0.0, 0.1)\f$
 *   - Recommended setup for sample run:
 *     - InitialTimeStep: 0.0001
 *     - Final time: 0.2
 *     - DomainCreator along wave propagation (no AMR):
 *       - Interval of length 1
 *       - InitialRefinement: 8
 *       - InitialGridPoints: 2
 *
 * - "123" problem (two symmetric rarefaction waves):
 *   - \f$(\rho_L, u_L, p_L) = (1.0, -2.0, 0.4)\f$
 *   - \f$(\rho_R, u_R, p_R) = (1.0, 2.0, 0.4)\f$
 *   - Recommended setup for sample run:
 *     - InitialTimeStep: 0.0001
 *     - Final time: 0.15
 *     - DomainCreator along wave propagation (no AMR):
 *       - Interval of length 1
 *       - InitialRefinement: 8
 *       - InitialGridPoints: 2
 *
 * - Collision of two blast waves (this test is challenging):
 *   - \f$(\rho_L, u_L, p_L) = (5.99924, 19.5975, 460.894)\f$
 *   - \f$(\rho_R, u_R, p_R) = (5.99242, -6.19633, 46.0950)\f$
 *   - Recommended setup for sample run:
 *     - InitialTimeStep: 0.00001
 *     - Final time: 0.012
 *     - DomainCreator along wave propagation (no AMR):
 *       - Interval of length 1
 *       - InitialRefinement: 8
 *       - InitialGridPoints: 2
 *
 * - Lax problem:
 *   - \f$(\rho_L, u_L, p_L) = (0.445, 0.698, 3.528)\f$
 *   - \f$(\rho_R, u_R, p_R) = (0.5, 0.0, 0.571)\f$
 *   - Recommended setup for sample run:
 *     - InitialTimeStep: 0.00001
 *     - Final time: 0.1
 *     - DomainCreator along wave propagation (no AMR):
 *       - Interval of length 1
 *       - InitialRefinement: 8
 *       - InitialGridPoints: 2
 *
 * where \f$\rho\f$ is the mass density, \f$p\f$ is the pressure, and
 * \f$u\f$ denotes the normal velocity.
 *
 * \note Currently the propagation axis must be hard-coded as a `size_t`
 * private member variable `propagation_axis_`, which can take one of
 * the three values `PropagationAxis::X`, `PropagationAxis::Y`, and
 * `PropagationAxis::Z`.
 *
 * \details The algorithm makes use of the following recipe:
 *
 * - Given the initial data on both sides of the initial interface of the
 *   discontinuity (here called "left" and "right" sides, where a coordinate
 *   axis points from left to right), we compute the pressure,
 *   \f$p_*\f$, and the normal velocity,
 *   \f$u_*\f$, in the so-called star region. This is done in the constructor.
 *   Here "normal" refers to the normal direction to the initial interface.
 *
 * - Given the pressure and the normal velocity in the star region, two
 *   `Wave` `struct`s are created, which represent the waves propagating
 *   at later times on each side of the contact discontinuity. Each `Wave`
 *   is equipped with two `struct`s named `Shock` and `Rarefaction` which
 *   contain functions that compute the primitive variables depending on whether
 *   the wave is a shock or a rarefaction.
 *
 * - If \f$p_* > p_K\f$, the wave is a shock, otherwise the wave is a
 *   rarefaction. Here \f$K\f$ stands for \f$L\f$ or \f$R\f$: the left
 *   and right initial pressure, respectively. Since this comparison can't be
 *   performed at compile time, each `Wave` holds a `bool` member `is_shock_`
 *   which is `true` if it is a shock, and `false` if it is a
 *   rarefaction wave. This variable is used to evaluate the correct functions
 *   at run time.
 *
 * - In order to obtain the primitives at a certain time and spatial location,
 *   we evaluate whether the spatial location is on the left of the propagating
 *   contact discontinuity \f$(x < u_* t)\f$ or on the right \f$(x > u_* t)\f$,
 *   and we use the corresponding functions for left or right `Wave`s,
 *   respectively.
 *
 * \note The characterization of each propagating wave will only
 * depend on the normal velocity, while the initial jump in the components of
 * the velocity transverse to the wave propagation will be advected at the
 * speed of the contact discontinuity (\f$u_*\f$).
 */
template <size_t Dim>
class RiemannProblem : public MarkAsAnalyticSolution {
  enum class Side { Left, Right };
  enum PropagationAxis { X = 0, Y = 1, Z = 2 };

  struct Wave;
  struct Shock;
  struct Rarefaction;

 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<false>;
  using source_term_type = Sources::NoSource;

  /// The adiabatic index of the fluid.
  struct AdiabaticIndex {
    using type = double;
    static constexpr OptionString help = {"The adiabatic index of the fluid."};
  };

  /// Initial position of the discontinuity
  struct InitialPosition {
    using type = double;
    static constexpr OptionString help = {
        "The initial position of the discontinuity."};
  };

  /// The mass density on the left of the initial discontinuity
  struct LeftMassDensity {
    using type = double;
    static constexpr OptionString help = {"The left mass density."};
  };

  /// The velocity on the left of the initial discontinuity
  struct LeftVelocity {
    using type = std::array<double, Dim>;
    static constexpr OptionString help = {"The left velocity."};
  };

  /// The pressure on the left of the initial discontinuity
  struct LeftPressure {
    using type = double;
    static constexpr OptionString help = {"The left pressure."};
  };

  /// The mass density on the right of the initial discontinuity
  struct RightMassDensity {
    using type = double;
    static constexpr OptionString help = {"The right mass density."};
  };

  /// The velocity on the right of the initial discontinuity
  struct RightVelocity {
    using type = std::array<double, Dim>;
    static constexpr OptionString help = {"The right velocity."};
  };

  /// The pressure on the right of the initial discontinuity
  struct RightPressure {
    using type = double;
    static constexpr OptionString help = {"The right pressure."};
  };

  /// The tolerance for solving for \f$p_*\f$.
  struct PressureStarTol {
    using type = double;
    static constexpr OptionString help = {
        "The tolerance for the numerical solution for p star"};
    static type default_value() noexcept { return 1.e-9; }
  };

  // Any of the two states that constitute the initial data, including
  // some derived quantities that are used repeatedly at each evaluation of the
  // different waves of the solution.
  /// Holds initial data on a side of the discontinuity and related quantities
  struct InitialData {
    InitialData() = default;
    InitialData(const InitialData& /*rhs*/) = default;
    InitialData& operator=(const InitialData& /*rhs*/) = default;
    InitialData(InitialData&& /*rhs*/) noexcept = default;
    InitialData& operator=(InitialData&& /*rhs*/) noexcept = default;
    ~InitialData() = default;

    InitialData(double mass_density, const std::array<double, Dim>& velocity,
                double pressure, double adiabatic_index,
                size_t propagation_axis) noexcept;

    // clang-tidy: no runtime references
    void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

    double mass_density_ = std::numeric_limits<double>::signaling_NaN();
    std::array<double, Dim> velocity_ =
        make_array<Dim>(std::numeric_limits<double>::signaling_NaN());
    double pressure_ = std::numeric_limits<double>::signaling_NaN();
    double sound_speed_ = std::numeric_limits<double>::signaling_NaN();
    double normal_velocity_ = std::numeric_limits<double>::signaling_NaN();

    // Data-dependent constants A and B in Eqns. (4.8) of Toro.
    double constant_a_ = std::numeric_limits<double>::signaling_NaN();
    double constant_b_ = std::numeric_limits<double>::signaling_NaN();

    friend bool operator==(const InitialData& lhs,
                           const InitialData& rhs) noexcept {
      return lhs.mass_density_ == rhs.mass_density_ and
             lhs.velocity_ == rhs.velocity_ and lhs.pressure_ == rhs.pressure_;
    }
  };

  using options = tmpl::list<AdiabaticIndex, InitialPosition, LeftMassDensity,
                             LeftVelocity, LeftPressure, RightMassDensity,
                             RightVelocity, RightPressure, PressureStarTol>;

  static constexpr OptionString help = {
      "Riemann Problem in 1, 2 or 3D along any coordinate axis."};

  RiemannProblem() = default;
  RiemannProblem(const RiemannProblem& /*rhs*/) = delete;
  RiemannProblem& operator=(const RiemannProblem& /*rhs*/) = delete;
  RiemannProblem(RiemannProblem&& /*rhs*/) noexcept = default;
  RiemannProblem& operator=(RiemannProblem&& /*rhs*/) noexcept = default;
  ~RiemannProblem() = default;

  RiemannProblem(
      double adiabatic_index, double initial_position, double left_mass_density,
      const std::array<double, Dim>& left_velocity, double left_pressure,
      double right_mass_density, const std::array<double, Dim>& right_velocity,
      double right_pressure,
      double pressure_star_tol = PressureStarTol::default_value()) noexcept;

  /// Retrieve a collection of hydrodynamic variables at position `x`
  /// and time `t`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, Dim, Frame::Inertial>& x, double t,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    const Wave left(left_initial_data_, pressure_star_, velocity_star_,
                    adiabatic_index_, Side::Left);
    const Wave right(right_initial_data_, pressure_star_, velocity_star_,
                     adiabatic_index_, Side::Right);

    tnsr::I<DataType, Dim, Frame::Inertial> x_shifted(x);
    x_shifted.get(propagation_axis_) -= initial_position_;

    return {tuples::get<Tags>(
        variables(x_shifted, t, tmpl::list<Tags>{}, left, right))...};
  }

  const EquationsOfState::IdealFluid<false>& equation_of_state() const
      noexcept {
    return equation_of_state_;
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

  // Retrieve these member variables for testing purposes.
  constexpr std::array<double, 2> diagnostic_star_region_values() const
      noexcept {
    return make_array(pressure_star_, velocity_star_);
  }

 private:
  // @{
  /// Retrieve hydro variable at `(x, t)`
  template <typename DataType>
  auto variables(const tnsr::I<DataType, Dim, Frame::Inertial>& x_shifted,
                 double t, tmpl::list<Tags::MassDensity<DataType>> /*meta*/,
                 const Wave& left, const Wave& right) const noexcept
      -> tuples::TaggedTuple<Tags::MassDensity<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, Dim, Frame::Inertial>& x_shifted, double t,
      tmpl::list<Tags::Velocity<DataType, Dim, Frame::Inertial>> /*meta*/,
      const Wave& left, const Wave& right) const noexcept
      -> tuples::TaggedTuple<Tags::Velocity<DataType, Dim, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, Dim, Frame::Inertial>& x_shifted,
                 double t, tmpl::list<Tags::Pressure<DataType>> /*meta*/,
                 const Wave& left, const Wave& right) const noexcept
      -> tuples::TaggedTuple<Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, Dim, Frame::Inertial>& x_shifted,
                 double t,
                 tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/,
                 const Wave& left, const Wave& right) const noexcept
      -> tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>>;
  // @}

  // Any of the two waves propagating on each side of the contact discontinuity.
  // Depending on whether p_* is larger or smaller
  // than the initial pressure on the corresponding side, the wave is a
  // shock (larger) or a rarefaction (smaller) wave. Since p_*
  // is computed at run time, the characterization of the wave
  // must also be done at run time. Shock and rarefaction waves will provide
  // different functions to retrieve the primitive variables at a given (x, t).
  // Here normal velocity means velocity along the wave propagation.
  struct Wave {
    Wave(const InitialData& data, double pressure_star, double velocity_star,
         double adiabatic_index, const Side& side) noexcept;

    double mass_density(double x_shifted, double t) const noexcept;

    double normal_velocity(double x_shifted, double t,
                           double velocity_star) const noexcept;

    double pressure(double x_shifted, double t, double pressure_star) const
        noexcept;

   private:
    // p_* over initial pressure on the corresponding side
    double pressure_ratio_ = std::numeric_limits<double>::signaling_NaN();
    // false if rarefaction wave
    bool is_shock_ = true;

    InitialData data_{};
    Shock shock_{};
    Rarefaction rarefaction_{};
  };

  struct Shock {
    Shock(const InitialData& data, double pressure_ratio,
          double adiabatic_index, const Side& side) noexcept;

    double mass_density(double x_shifted, double t,
                        const InitialData& data) const noexcept;
    double normal_velocity(double x_shifted, double t, const InitialData& data,
                           double velocity_star) const noexcept;

    double pressure(double x_shifted, double t, const InitialData& data,
                    double pressure_star) const noexcept;

   private:
    double direction_ = std::numeric_limits<double>::signaling_NaN();
    double mass_density_star_ = std::numeric_limits<double>::signaling_NaN();
    double shock_speed_ = std::numeric_limits<double>::signaling_NaN();
  };

  struct Rarefaction {
    Rarefaction(const InitialData& data, double pressure_ratio,
                double velocity_star, double adiabatic_index,
                const Side& side) noexcept;

    double mass_density(double x_shifted, double t,
                        const InitialData& data) const noexcept;
    double normal_velocity(double x_shifted, double t, const InitialData& data,
                           double velocity_star) const noexcept;

    double pressure(double x_shifted, double t, const InitialData& data,
                    double pressure_star) const noexcept;

   private:
    double direction_ = std::numeric_limits<double>::signaling_NaN();
    double gamma_mm_ = std::numeric_limits<double>::signaling_NaN();
    double gamma_pp_ = std::numeric_limits<double>::signaling_NaN();
    double mass_density_star_ = std::numeric_limits<double>::signaling_NaN();
    double sound_speed_star_ = std::numeric_limits<double>::signaling_NaN();
    double head_speed_ = std::numeric_limits<double>::signaling_NaN();
    double tail_speed_ = std::numeric_limits<double>::signaling_NaN();
  };

  template <size_t SpatialDim>
  friend bool
  operator==(  // NOLINT (clang-tidy: readability-redundant-declaration)
      const RiemannProblem<SpatialDim>& lhs,
      const RiemannProblem<SpatialDim>& rhs) noexcept;

  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  double initial_position_ = std::numeric_limits<double>::signaling_NaN();
  size_t propagation_axis_ = PropagationAxis::X;
  InitialData left_initial_data_{};
  InitialData right_initial_data_{};

  double pressure_star_tol_ = std::numeric_limits<double>::signaling_NaN();
  // the pressure in the star region, p_*
  double pressure_star_ = std::numeric_limits<double>::signaling_NaN();
  // the velocity in the star region, u_*
  double velocity_star_ = std::numeric_limits<double>::signaling_NaN();

  EquationsOfState::IdealFluid<false> equation_of_state_{};
};

template <size_t Dim>
bool operator!=(const RiemannProblem<Dim>& lhs,
                const RiemannProblem<Dim>& rhs) noexcept;

}  // namespace Solutions
}  // namespace NewtonianEuler
