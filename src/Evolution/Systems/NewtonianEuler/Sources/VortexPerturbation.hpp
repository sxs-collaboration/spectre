// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/Source.hpp"
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace NewtonianEuler {
namespace Solutions {
template <size_t Dim>
struct IsentropicVortex;
}  // namespace Solutions
}  // namespace NewtonianEuler

namespace Tags {
struct Time;
}  // namespace Tags

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace NewtonianEuler::Sources {

/*!
 * \brief Source generating a modified isentropic vortex.
 *
 * If Solutions::IsentropicVortex is modifed so that the flow velocity along
 * the \f$z-\f$axis is not a constant but a function of \f$z\f$, the new vortex
 * will be a solution to the 3-D Newtonian Euler equations with a source term,
 *
 * \f{align*}
 * \partial_t\rho + \partial_i F^i(\rho) &= S(\rho)\\
 * \partial_t S^i + \partial_j F^{j}(S^i) &= S(S^i)\\
 * \partial_t e + \partial_i F^i(e) &= S(e),
 * \f}
 *
 * where \f$F^i(u)\f$ is the volume flux of the conserved quantity \f$u\f$
 * (see ComputeFluxes), and
 *
 * \f{align*}
 * S(\rho) &= \rho \dfrac{dv_z}{dz}\\
 * S(S_x) &= S_x \dfrac{dv_z}{dz}\\
 * S(S_y) &= S_y \dfrac{dv_z}{dz}\\
 * S(S_z) &= 2S_z \dfrac{dv_z}{dz}\\
 * S(e) &= \left(e + p + v_z S_z\right)\dfrac{dv_z}{dz},
 * \f}
 *
 * where \f$\rho\f$ is the mass density of the vortex, \f$S_i\f$ is
 * its momentum density, \f$e\f$ is its energy density,
 * \f$v_z = v_z(z)\f$ is the \f$z-\f$component of its velocity,
 * and \f$p\f$ is its pressure. These quantities are readily obtained
 * from the primitive variables, whose expressions are those in
 * Solutions::IsentropicVortex
 *
 * Currently \f$v_z(z)=sin(z)\f$ is hard-coded.
 */
class VortexPerturbation : public Source<3> {
 public:
  /// The perturbation amplitude
  struct PerturbationAmplitude {
    using type = double;
    static constexpr Options::String help = {"The perturbation amplitude."};
  };

  using options = tmpl::list<PerturbationAmplitude>;

  static constexpr Options::String help = {
      "Source terms corresponding to a vortex perturbation. Should be used "
      "with the IsentropicVortex solution."};

  explicit VortexPerturbation(double perturbation_amplitude);

  VortexPerturbation() = default;
  VortexPerturbation(const VortexPerturbation& /*rhs*/) = default;
  VortexPerturbation& operator=(const VortexPerturbation& /*rhs*/) = default;
  VortexPerturbation(VortexPerturbation&& /*rhs*/) = default;
  VortexPerturbation& operator=(VortexPerturbation&& /*rhs*/) = default;
  ~VortexPerturbation() override = default;

  /// \cond
  explicit VortexPerturbation(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(VortexPerturbation);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  auto get_clone() const -> std::unique_ptr<Source<3>> override;

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
  double perturbation_amplitude_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace NewtonianEuler::Sources
