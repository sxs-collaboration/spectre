// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/TagsDeclarations.hpp"
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

namespace NewtonianEuler {
namespace Sources {

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
struct UniformAcceleration {
  UniformAcceleration() = default;
  UniformAcceleration(const UniformAcceleration& /*rhs*/) = default;
  UniformAcceleration& operator=(const UniformAcceleration& /*rhs*/) = default;
  UniformAcceleration(UniformAcceleration&& /*rhs*/) = default;
  UniformAcceleration& operator=(UniformAcceleration&& /*rhs*/) = default;
  ~UniformAcceleration() = default;

  explicit UniformAcceleration(
      const std::array<double, Dim>& acceleration_field);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/);

  using sourced_variables =
      tmpl::list<Tags::MomentumDensity<Dim>, Tags::EnergyDensity>;

  using argument_tags =
      tmpl::list<Tags::MassDensityCons, Tags::MomentumDensity<Dim>>;

  void apply(gsl::not_null<tnsr::I<DataVector, Dim>*> source_momentum_density,
             gsl::not_null<Scalar<DataVector>*> source_energy_density,
             const Scalar<DataVector>& mass_density_cons,
             const tnsr::I<DataVector, Dim>& momentum_density) const;

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
}  // namespace Sources
}  // namespace NewtonianEuler
