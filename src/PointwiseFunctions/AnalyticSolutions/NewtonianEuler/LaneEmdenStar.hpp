// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/LaneEmdenGravitationalField.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
/// \endcond

// IWYU pragma: no_include <pup.h>

namespace NewtonianEuler {
namespace Solutions {

/*!
 * \brief A static spherically symmetric star in Newtonian gravity
 *
 * The solution for a static, spherically-symmetric star in 3 dimensions, found
 * by solving the Lane-Emden equation \cite Chandrasekhar1939
 * \cite Shapiro1983 .
 * The Lane-Emden equation has closed-form solutions for certain equations of
 * state; this class implements the solution for a polytropic fluid with
 * polytropic exponent \f$\Gamma=2\f$ (i.e., with polytropic index \f$n=1\f$).
 * The solution is returned in units where \f$G=1\f$, with \f$G\f$ the
 * gravitational constant.
 *
 * The radius and mass of the star are determined by the polytropic constant
 * \f$\kappa\f$ and central density \f$\rho_c\f$.
 * The radius is \f$R = \pi \alpha\f$,
 * and the mass is \f$M = 4 \pi^2 \alpha^3 \rho_c\f$,
 * where \f$\alpha = \sqrt{\kappa / (2 \pi)}\f$ and \f$G=1\f$.
 */
class LaneEmdenStar : public MarkAsAnalyticSolution {
 public:
  using equation_of_state_type = EquationsOfState::PolytropicFluid<false>;
  using source_term_type = Sources::LaneEmdenGravitationalField;

  /// The central mass density of the star.
  struct CentralMassDensity {
    using type = double;
    static constexpr Options::String help = {
        "The central mass density of the star."};
    static type lower_bound() { return 0.; }
  };

  /// The polytropic constant of the polytropic fluid.
  struct PolytropicConstant {
    using type = double;
    static constexpr Options::String help = {
        "The polytropic constant of the fluid."};
    static type lower_bound() { return 0.; }
  };

  using options = tmpl::list<CentralMassDensity, PolytropicConstant>;

  static constexpr Options::String help = {
      "A static, spherically-symmetric star in Newtonian gravity, found by\n"
      "solving the Lane-Emden equations, with a given central density and\n"
      "polytropic fluid. The fluid has polytropic index 1, but the polytropic\n"
      "constant is specifiable"};

  LaneEmdenStar() = default;
  LaneEmdenStar(const LaneEmdenStar& /*rhs*/) = delete;
  LaneEmdenStar& operator=(const LaneEmdenStar& /*rhs*/) = delete;
  LaneEmdenStar(LaneEmdenStar&& /*rhs*/) = default;
  LaneEmdenStar& operator=(LaneEmdenStar&& /*rhs*/) = default;
  ~LaneEmdenStar() = default;

  LaneEmdenStar(double central_mass_density, double polytropic_constant);

  /// Retrieve a collection of variables at `(x, t)`
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, 3>& x,
                                         const double /*t*/,
                                         tmpl::list<Tags...> /*meta*/) const {
    const auto mass_density = precompute_mass_density(x);
    return {tuples::get<Tags>(variables(tmpl::list<Tags>{}, mass_density))...};
  }

  /// \brief Compute the gravitational field for the corresponding source term,
  /// LaneEmdenGravitationalField.
  ///
  /// The result is the vector-field giving the acceleration due to gravity
  /// that is felt by a test particle.
  template <typename DataType>
  tnsr::I<DataType, 3> gravitational_field(const tnsr::I<DataType, 3>& x) const;

  const EquationsOfState::PolytropicFluid<false>& equation_of_state() const {
    return equation_of_state_;
  }

  const Sources::LaneEmdenGravitationalField& source_term() const {
    return source_term_;
  }

  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/);  // NOLINT

 private:
  template <typename DataType>
  Scalar<DataType> precompute_mass_density(const tnsr::I<DataType, 3>& x) const;

  template <typename DataType>
  tuples::TaggedTuple<Tags::MassDensity<DataType>> variables(
      tmpl::list<Tags::MassDensity<DataType>> /*meta*/,
      const Scalar<DataType>& mass_density) const;

  template <typename DataType>
  tuples::TaggedTuple<Tags::Velocity<DataType, 3>> variables(
      tmpl::list<Tags::Velocity<DataType, 3>> /*meta*/,
      const Scalar<DataType>& mass_density) const;

  template <typename DataType>
  tuples::TaggedTuple<Tags::Pressure<DataType>> variables(
      tmpl::list<Tags::Pressure<DataType>> /*meta*/,
      const Scalar<DataType>& mass_density) const;

  template <typename DataType>
  tuples::TaggedTuple<Tags::SpecificInternalEnergy<DataType>> variables(
      tmpl::list<Tags::SpecificInternalEnergy<DataType>> /*meta*/,
      const Scalar<DataType>& mass_density) const;

  friend bool operator==(const LaneEmdenStar& lhs, const LaneEmdenStar& rhs);

  double central_mass_density_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  EquationsOfState::PolytropicFluid<false> equation_of_state_{};
  Sources::LaneEmdenGravitationalField source_term_{};
};

bool operator!=(const LaneEmdenStar& lhs, const LaneEmdenStar& rhs);

}  // namespace Solutions
}  // namespace NewtonianEuler
