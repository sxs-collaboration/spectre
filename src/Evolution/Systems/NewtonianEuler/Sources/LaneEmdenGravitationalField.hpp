// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/Source.hpp"
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace NewtonianEuler {
namespace Solutions {
struct LaneEmdenStar;
}  // namespace Solutions
}  // namespace NewtonianEuler

namespace PUP {
class er;
}  // namespace PUP

namespace Tags {
template <typename SolutionType>
struct AnalyticSolution;
}  // namespace Tags

namespace domain::Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace domain::Tags

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace NewtonianEuler::Sources {

/*!
 * \brief Source giving the acceleration due to gravity in the spherical,
 * Newtonian Lane-Emden star solution.
 *
 * The gravitational field \f$g^i\f$ enters the NewtonianEuler system as source
 * terms for the conserved momentum and energy:
 *
 * \f{align*}
 * \partial_t S^i + \partial_j F^{j}(S^i) &= S(S^i) = \rho g^i
 * \partial_t e + \partial_j F^{j}(e) &= S(e) = S_i g^i,
 * \f}
 *
 * where \f$S^i\f$ is the conserved momentum density, \f$e\f$ is the conserved
 * energy, \f$F^{j}(u)\f$ is the flux of the conserved quantity \f$u\f$, and
 * \f$\rho\f$ is the fluid mass density.
 *
 * \note This source is specialized to the Lane-Emden solution because it
 * queries a LaneEmdenStar analytic solution for the gravitational field that
 * generates the fluid acceleration. This source does *not* integrate the fluid
 * density to compute a self-consistent gravitational field (i.e., as if one
 * were solving a coupled Euler + Poisson system).
 */
class LaneEmdenGravitationalField : public Source<3> {
 public:
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
      "The gravitational field corresponding to a static, "
      "spherically-symmetric star in Newtonian gravity, found by "
      "solving the Lane-Emden equations, with a given central density and "
      "polytropic fluid. The fluid has polytropic index 1, but the polytropic "
      "constant is specifiable"};

  LaneEmdenGravitationalField(double central_mass_density,
                              double polytropic_constant);

  LaneEmdenGravitationalField() = default;
  LaneEmdenGravitationalField(const LaneEmdenGravitationalField& /*rhs*/) =
      default;
  LaneEmdenGravitationalField& operator=(
      const LaneEmdenGravitationalField& /*rhs*/) = default;
  LaneEmdenGravitationalField(LaneEmdenGravitationalField&& /*rhs*/) = default;
  LaneEmdenGravitationalField& operator=(
      LaneEmdenGravitationalField&& /*rhs*/) = default;
  ~LaneEmdenGravitationalField() override = default;

  /// \cond
  explicit LaneEmdenGravitationalField(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(LaneEmdenGravitationalField);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  auto get_clone() const -> std::unique_ptr<Source> override;

  void operator()(
      gsl::not_null<Scalar<DataVector>*> source_mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, 3>*> source_momentum_density,
      gsl::not_null<Scalar<DataVector>*> source_energy_density,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, 3>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, 3>& velocity,
      const Scalar<DataVector>& pressure,
      const Scalar<DataVector>& specific_internal_energy,
      const EquationsOfState::EquationOfState<false, 2>& eos,
      const tnsr::I<DataVector, 3>& coords, double time) const override;

 private:
  tnsr::I<DataVector, 3> gravitational_field(
      const tnsr::I<DataVector, 3>& x) const;

  double central_mass_density_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace NewtonianEuler::Sources
