// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"  // for IdealFluid

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd {
namespace AnalyticData {

/*!
 * \brief Analytic initial data for the relativistic Orszag-Tang vortex.
 *
 * The relativistic version of the Orszag-Tang vortex is a
 * 2-dimensional test case for relativistic MHD systems (see, e.g.,
 * \cite Beckwith2011iy).  It describes the flow of an ideal fluid with
 * adiabatic index \f$5/3\f$.  The initial conditions (and hence the
 * states at later times) are periodic in both \f$x\f$ and \f$y\f$
 * with period 1.  The initial conditions are:
 * \f{align*}
 * \rho &= \frac{25}{36 \pi} \\
 * p &= \frac{5}{12 \pi} \\
 * v_x &= -\frac{1}{2} \sin(2 \pi y) \\
 * v_y &= \frac{1}{2} \sin(2 \pi x) \\
 * B_x &= -\frac{1}{\sqrt{4 \pi}} \sin(2 \pi y) \\
 * B_y &= \frac{1}{\sqrt{4 \pi}} \sin(4 \pi x)
 * \f}
 * with \f$\rho\f$ the rest mass density, \f$p\f$ the pressure,
 * \f$v_i\f$ the spatial velocity, and \f$B_i\f$ the magnetic field.
 *
 * \parblock
 * \note We do not currently support 2-dimensional RMHD, so this class
 * provides 3-dimensional data with no \f$z\f$-dependence.
 * \endparblock
 *
 * \parblock
 * \note There are multiple errors in the description of this test
 * problem in the original SpECTRE paper \cite Kidder2016hev and there
 * is a sign error in the velocity in \cite Beckwith2011iy.  Despite these
 * errors, the actual tests performed for those papers matched the standard
 * problem as presented here.
 * \endparblock
 */
class OrszagTangVortex {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;

  using options = tmpl::list<>;

  static constexpr OptionString help = {"The relativistic Orszag-Tang vortex"};

  OrszagTangVortex();

  // @{
  /// Retrieve hydro variable at `x`
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

  /// Retrieve a collection of hydro variables at `x`
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
  template <typename DataType, typename Tag,
            Requires<not tmpl::list_contains_v<hydro::grmhd_tags<DataType>,
                                               Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const noexcept {
    constexpr double dummy_time = 0.;
    return {std::move(get<Tag>(gr::Solutions::Minkowski<3>{}.variables(
        x, dummy_time, tmpl::list<Tag>{})))};
  }

  const equation_of_state_type& equation_of_state() const noexcept {
    return equation_of_state_;
  }

  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT(google-runtime-references)

 private:
  EquationsOfState::IdealFluid<true> equation_of_state_{5. / 3.};
};

bool operator==(const OrszagTangVortex& lhs,
                const OrszagTangVortex& rhs) noexcept;

bool operator!=(const OrszagTangVortex& lhs,
                const OrszagTangVortex& rhs) noexcept;

}  // namespace AnalyticData
}  // namespace grmhd
