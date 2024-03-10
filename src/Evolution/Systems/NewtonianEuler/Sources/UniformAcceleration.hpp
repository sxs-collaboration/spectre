// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Sources/Source.hpp"
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"
#include "Options/String.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace PUP {
class er;
}  // namespace PUP

namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace NewtonianEuler::Sources {

/*!
 * \brief Source generated from an external uniform acceleration.
 *
 * The NewtonianEuler system with source terms is written as
 *
 * \f{align*}
 * \partial_t\rho + \partial_i F^i(\rho) &= S(\rho)\\
 * \partial_t S^i + \partial_j F^{j}(S^i) &= S(S^i)\\
 * \partial_t e + \partial_i F^i(e) &= S(e),
 * \f}
 *
 * where \f$F^i(u)\f$ is the volume flux of the conserved quantity \f$u\f$
 * (see ComputeFluxes). For an external acceleration \f$a^i\f$, one has
 *
 * \f{align*}
 * S(\rho) &= 0\\
 * S(S^i) &= \rho a^i\\
 * S(e) &= S_ia^i,
 * \f}
 *
 * where \f$\rho\f$ is the mass density, \f$S^i\f$ is the momentum density,
 * and \f$e\f$ is the energy density.
 */
template <size_t Dim>
class UniformAcceleration : public Source<Dim> {
 public:
  /// The applied acceleration
  struct Acceleration {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = {"The applied accerelation."};
  };

  using options = tmpl::list<Acceleration>;

  static constexpr Options::String help = {
      "Source terms corresponding to a uniform acceleration."};

  UniformAcceleration() = default;
  UniformAcceleration(const UniformAcceleration& /*rhs*/) = default;
  UniformAcceleration& operator=(const UniformAcceleration& /*rhs*/) = default;
  UniformAcceleration(UniformAcceleration&& /*rhs*/) = default;
  UniformAcceleration& operator=(UniformAcceleration&& /*rhs*/) = default;
  ~UniformAcceleration() override = default;

  explicit UniformAcceleration(
      const std::array<double, Dim>& acceleration_field);

  /// \cond
  explicit UniformAcceleration(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(UniformAcceleration);
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  auto get_clone() const -> std::unique_ptr<Source<Dim>> override;

  void operator()(
      gsl::not_null<Scalar<DataVector>*> source_mass_density_cons,
      gsl::not_null<tnsr::I<DataVector, Dim>*> source_momentum_density,
      gsl::not_null<Scalar<DataVector>*> source_energy_density,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const tnsr::I<DataVector, Dim>& velocity,
      const Scalar<DataVector>& pressure,
      const Scalar<DataVector>& specific_internal_energy,
      const EquationsOfState::EquationOfState<false, 2>& eos,
      const tnsr::I<DataVector, Dim>& coords, double time) const override;

 private:
  template <size_t SpatialDim>
  friend bool operator==(  // NOLINT(readability-redundant-declaration)
      const UniformAcceleration<SpatialDim>& lhs,
      const UniformAcceleration<SpatialDim>& rhs);

  std::array<double, Dim> acceleration_field_ =
      make_array<Dim>(std::numeric_limits<double>::signaling_NaN());
};

template <size_t Dim>
bool operator!=(const UniformAcceleration<Dim>& lhs,
                const UniformAcceleration<Dim>& rhs);
}  // namespace NewtonianEuler::Sources
