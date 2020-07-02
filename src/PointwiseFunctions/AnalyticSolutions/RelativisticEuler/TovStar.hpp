// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/preprocessor/list/for_each.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

// IWYU pragma: no_include <pup.h>

// IWYU pragma: no_include "Utilities/GenerateInstantiations.hpp"

namespace RelativisticEuler {
namespace Solutions {

/*!
 * \brief A static spherically symmetric star
 *
 * An analytic solution for a static, spherically-symmetric star found by
 * solving the Tolman-Oppenheimer-Volkoff (TOV) equations.  The equation of
 * state is assumed to be that of a polytropic fluid.
 *
 * \tparam RadialSolution selects not only how the TOV equations are solved, but
 * also the radial coordinate that the solution is given in (e.g. areal or
 * isotropic).  See the documentation of the specific `RadialSolution`s for
 * more details.
 */
template <typename RadialSolution>
class TovStar : public MarkAsAnalyticSolution {
 public:
  /*!
   * \brief The radial variables needed to compute the full `TOVStar` solution
   *
   * RadialSolution must provide a method that fills the radial variables
   * given the equation of state and the Cartesian coordinates \f$x^i\f$
   *
   * If the spherically symmetric metric is written as
   *
   * \f[
   * ds^2 = - e^{2 \Phi_t} dt^2 + e^{2 \Phi_r} dr^2 + e^{2 \Phi_\Omega} r^2
   * d\Omega^2
   * \f]
   *
   * where \f$r = \delta_{mn} x^m x^n\f$ is the `radial_coordinate` and
   * \f$\Phi_t\f$, \f$\Phi_r\f$, and \f$\Phi_\Omega\f$ are respectvely the
   * `metric_time_potential`, `metric_radial_potential`, and
   * `metric_angular_potential`, then the lapse, shift, and spatial metric in
   * the Cartesian coordinates are
   *
   * \f{align*}
   * \alpha &= e^{\Phi_t} \\
   * \beta^i &= 0 \\
   * \gamma_{ij} &= \delta_{ij} e^{2 \Phi_r} + \delta_{im} \delta_{jn}
   * \frac{x^m x^n}{r^2} \left( e^{2 \Phi_r} - e^{2 \Phi_\Omega} \right)
   * \f}
   *
   */
  template <typename DataType>
  struct RadialVariables {
    explicit RadialVariables(DataType radial_coordinate_in)
        : radial_coordinate(std::move(radial_coordinate_in)),
          rest_mass_density(
              make_with_value<Scalar<DataType>>(radial_coordinate, 0.0)),
          pressure(make_with_value<Scalar<DataType>>(radial_coordinate, 0.0)),
          specific_internal_energy(
              make_with_value<Scalar<DataType>>(radial_coordinate, 0.0)),
          specific_enthalpy(
              make_with_value<Scalar<DataType>>(radial_coordinate, 0.0)),
          metric_time_potential(
              make_with_value<DataType>(radial_coordinate, 0.0)),
          dr_metric_time_potential(
              make_with_value<DataType>(radial_coordinate, 0.0)),
          metric_radial_potential(
              make_with_value<DataType>(radial_coordinate, 0.0)),
          dr_metric_radial_potential(
              make_with_value<DataType>(radial_coordinate, 0.0)),
          metric_angular_potential(
              make_with_value<DataType>(radial_coordinate, 0.0)),
          dr_metric_angular_potential(
              make_with_value<DataType>(radial_coordinate, 0.0)) {}
    DataType radial_coordinate{};
    Scalar<DataType> rest_mass_density{};
    Scalar<DataType> pressure{};
    Scalar<DataType> specific_internal_energy{};
    Scalar<DataType> specific_enthalpy{};
    DataType metric_time_potential{};
    DataType dr_metric_time_potential{};
    DataType metric_radial_potential{};
    DataType dr_metric_radial_potential{};
    DataType metric_angular_potential{};
    DataType dr_metric_angular_potential{};
  };

  using equation_of_state_type = EquationsOfState::PolytropicFluid<true>;

  /// The central density of the star.
  struct CentralDensity {
    using type = double;
    static constexpr OptionString help = {"The central density of the star."};
    static type lower_bound() noexcept { return 0.; }
  };

  /// The polytropic constant of the polytropic fluid.
  struct PolytropicConstant {
    using type = double;
    static constexpr OptionString help = {
        "The polytropic constant of the fluid."};
    static type lower_bound() noexcept { return 0.; }
  };

  /// The polytropic exponent of the polytropic fluid.
  struct PolytropicExponent {
    using type = double;
    static constexpr OptionString help = {
        "The polytropic exponent of the fluid."};
    static type lower_bound() noexcept { return 1.; }
  };

  using options =
      tmpl::list<CentralDensity, PolytropicConstant, PolytropicExponent>;

  static constexpr OptionString help = {
      "A static, spherically-symmetric star found by solving the \n"
      "Tolman-Oppenheimer-Volkoff (TOV) equations, with a given central \n"
      "density and polytropic fluid."};

  TovStar() = default;
  TovStar(const TovStar& /*rhs*/) = delete;
  TovStar& operator=(const TovStar& /*rhs*/) = delete;
  TovStar(TovStar&& /*rhs*/) noexcept = default;
  TovStar& operator=(TovStar&& /*rhs*/) noexcept = default;
  ~TovStar() = default;

  TovStar(double central_rest_mass_density, double polytropic_constant,
          double polytropic_exponent) noexcept;

  /// Retrieve a collection of variables at `(x, t)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         const double /*t*/,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    auto radial_vars =
        radial_tov_solution().radial_variables(equation_of_state_, x);
    return {get<Tags>(variables(x, tmpl::list<Tags>{}, radial_vars))...};
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept;  //  NOLINT

  const EquationsOfState::PolytropicFluid<true>& equation_of_state() const
      noexcept {
    return equation_of_state_;
  }

 private:
  template <typename LocalRadialSolution>
  friend bool operator==(const TovStar<LocalRadialSolution>& lhs,
                         const TovStar<LocalRadialSolution>& rhs) noexcept;

  const RadialSolution& radial_tov_solution() const noexcept;

  template <typename DataType>
  using SpatialVelocity = hydro::Tags::SpatialVelocity<DataType, 3>;

  template <typename DataType>
  using MagneticField = hydro::Tags::MagneticField<DataType, 3>;

  template <typename DataType>
  using DerivLapse = ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<3>,
                                   Frame::Inertial>;

  template <typename DataType>
  using Shift = gr::Tags::Shift<3, Frame::Inertial, DataType>;

  template <typename DataType>
  using DerivShift =
      ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataType>,
                    tmpl::size_t<3>, Frame::Inertial>;

  template <typename DataType>
  using SpatialMetric = gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>;

  template <typename DataType>
  using DerivSpatialMetric =
      ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>,
                    tmpl::size_t<3>, Frame::Inertial>;

  template <typename DataType>
  using InverseSpatialMetric =
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataType>;

  template <typename DataType>
  using ExtrinsicCurvature =
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataType>;

#define FUNC_DECL(r, data, elem)                                \
  template <typename DataType>                                  \
  tuples::TaggedTuple<elem> variables(                          \
      const tnsr::I<DataType, 3>& x, tmpl::list<elem> /*meta*/, \
      const RadialVariables<DataType>& radial_vars) const noexcept;

#define MY_LIST                                                                \
  BOOST_PP_TUPLE_TO_LIST(                                                      \
      17, (hydro::Tags::RestMassDensity<DataType>,                             \
           hydro::Tags::SpecificInternalEnergy<DataType>,                      \
           hydro::Tags::Pressure<DataType>, SpatialVelocity<DataType>,         \
           MagneticField<DataType>,                                            \
           hydro::Tags::DivergenceCleaningField<DataType>,                     \
           hydro::Tags::LorentzFactor<DataType>,                               \
           hydro::Tags::SpecificEnthalpy<DataType>, gr::Tags::Lapse<DataType>, \
           DerivLapse<DataType>, Shift<DataType>, DerivShift<DataType>,        \
           SpatialMetric<DataType>, DerivSpatialMetric<DataType>,              \
           gr::Tags::SqrtDetSpatialMetric<DataType>,                           \
           InverseSpatialMetric<DataType>, ExtrinsicCurvature<DataType>))

  BOOST_PP_LIST_FOR_EACH(FUNC_DECL, _, MY_LIST)
#undef MY_LIST
#undef FUNC_DECL

  double central_rest_mass_density_ =
      std::numeric_limits<double>::signaling_NaN();
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::PolytropicFluid<true> equation_of_state_{};
};

template <typename RadialSolution>
bool operator!=(const TovStar<RadialSolution>& lhs,
                const TovStar<RadialSolution>& rhs) noexcept;

}  // namespace Solutions
}  // namespace RelativisticEuler
