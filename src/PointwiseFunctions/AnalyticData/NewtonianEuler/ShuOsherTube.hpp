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

namespace NewtonianEuler::AnalyticData {
/*!
 * \brief Initial data for the Shu-Osher oscillatory shock tube \cite Shu1988439
 *
 * The general initial data is given by:
 *
 * \f{align*}{
 *  \{\rho,v^x,p\}=
 * \left\{
 * \begin{array}{ll}
 *   \left\{\rho_L, v^x_L, p_L\right\} &\mathrm{if} \;\;\; x<\Delta \\
 *   \left\{1+\epsilon\sin(\lambda x), v^x_R, p_R\right\} &\mathrm{if} \;\;\;
 *    x\ge \Delta
 * \end{array}\right.
 * \f}
 *
 * with the adiabatic index being 1.4.
 *
 * With the standard parameters given below, this is a Mach-3 shock moving into
 * a sinusoidal density profile.
 *
 * \f{align*}{
 *  \{\rho,v^x,p\}=
 * \left\{
 * \begin{array}{ll}
 *   \left\{3.857143, 2.629369, 10.33333\right\} &\mathrm{if} \;\;\; x<-4 \\
 *   \left\{1+0.2\sin(5x), 0, 1\right\} &\mathrm{if} \;\;\; x\ge -4
 * \end{array}\right.
 * \f}
 *
 * With these values the usual final time is 1.8.
 */
class ShuOsherTube : public MarkAsAnalyticData {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<false>;
  using source_term_type = Sources::NoSource;

  /// Initial postition of the discontinuity
  struct JumpPosition {
    using type = double;
    static constexpr Options::String help = {
        "The initial position of the discontinuity."};
  };

  struct LeftMassDensity {
    using type = double;
    static constexpr Options::String help = {"The left mass density."};
    static type lower_bound() noexcept { return 0.0; }
  };

  struct LeftVelocity {
    using type = double;
    static constexpr Options::String help = {"The left velocity."};
  };

  struct LeftPressure {
    using type = double;
    static constexpr Options::String help = {"The left pressure."};
    static type lower_bound() noexcept { return 0.0; }
  };

  struct RightVelocity {
    using type = double;
    static constexpr Options::String help = {"The right velocity."};
  };

  struct RightPressure {
    using type = double;
    static constexpr Options::String help = {"The right pressure."};
    static type lower_bound() noexcept { return 0.0; }
  };

  struct Epsilon {
    using type = double;
    static constexpr Options::String help = {"Sinusoid amplitude."};
    static type lower_bound() noexcept { return 0.0; }
    static type upper_bound() noexcept { return 1.0; }
  };

  struct Lambda {
    using type = double;
    static constexpr Options::String help = {"Sinusoid wavelength."};
  };

  static constexpr Options::String help = {
      "1D Shu-Osher oscillatory shock tube."};

  using options =
      tmpl::list<JumpPosition, LeftMassDensity, LeftVelocity, LeftPressure,
                 RightVelocity, RightPressure, Epsilon, Lambda>;

  ShuOsherTube(double jump_position, double mass_density_l, double velocity_l,
               double pressure_l, double velocity_r, double pressure_r,
               double epsilon, double lambda) noexcept;
  ShuOsherTube() = default;
  ShuOsherTube(const ShuOsherTube& /*rhs*/) = delete;
  ShuOsherTube& operator=(const ShuOsherTube& /*rhs*/) = delete;
  ShuOsherTube(ShuOsherTube&& /*rhs*/) noexcept = default;
  ShuOsherTube& operator=(ShuOsherTube&& /*rhs*/) noexcept = default;
  ~ShuOsherTube() = default;

  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, 1, Frame::Inertial>& x,
      tmpl::list<Tags...> /*meta*/) const noexcept {
    return {tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  const EquationsOfState::IdealFluid<false>& equation_of_state()
      const noexcept {
    return equation_of_state_;
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& p) noexcept;  //  NOLINT

 private:
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 1, Frame::Inertial>& x,
                 tmpl::list<Tags::MassDensity<DataType>> /*meta*/)
      const noexcept -> tuples::TaggedTuple<Tags::MassDensity<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 1, Frame::Inertial>& x,
      tmpl::list<Tags::Velocity<DataType, 1, Frame::Inertial>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<Tags::Velocity<DataType, 1, Frame::Inertial>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 1, Frame::Inertial>& x,
                 tmpl::list<Tags::Pressure<DataType>> /*meta*/) const noexcept
      -> tuples::TaggedTuple<Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 1, Frame::Inertial>& x,
                 tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/)
      const noexcept
      -> tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>>;

  friend bool
  operator==(  // NOLINT (clang-tidy: readability-redundant-declaration)
      const ShuOsherTube& lhs, const ShuOsherTube& rhs) noexcept;

  double mass_density_l_ = std::numeric_limits<double>::signaling_NaN();
  double velocity_l_ = std::numeric_limits<double>::signaling_NaN();
  double pressure_l_ = std::numeric_limits<double>::signaling_NaN();
  double jump_position_ = std::numeric_limits<double>::signaling_NaN();
  double epsilon_ = std::numeric_limits<double>::signaling_NaN();
  double lambda_ = std::numeric_limits<double>::signaling_NaN();
  double velocity_r_ = std::numeric_limits<double>::signaling_NaN();
  double pressure_r_ = std::numeric_limits<double>::signaling_NaN();
  double adiabatic_index_ = 1.4;
  EquationsOfState::IdealFluid<false> equation_of_state_{adiabatic_index_};
};

bool operator!=(const ShuOsherTube& lhs, const ShuOsherTube& rhs) noexcept;
}  // namespace NewtonianEuler::AnalyticData
