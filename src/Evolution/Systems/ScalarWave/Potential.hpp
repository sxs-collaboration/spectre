// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ScalarWave {

/// @{
/*!
 * \brief Computes the potential energy density of a massive scalar field:
 *
 * \f[
 * V(\psi) = \frac{1}{2} m^2 \psi^2
 * \f]
 */
template <size_t SpatialDim>
void potential(gsl::not_null<Scalar<DataVector>*> result,
               const Scalar<DataVector>& psi, const double& mass_squared);

template <size_t SpatialDim>
Scalar<DataVector> potential(const Scalar<DataVector>& psi,
                             const double& mass_squared);
/// @}

namespace Tags {
/// \brief Computes the potential energy using ScalarWave::potential()
template <size_t SpatialDim>
struct PotentialCompute : Potential<SpatialDim>, db::ComputeTag {
  using argument_tags = tmpl::list<Psi, MassSquared>;
  using return_type = Scalar<DataVector>;

  static constexpr auto function =
      static_cast<void (*)(gsl::not_null<Scalar<DataVector>*>,
                           const Scalar<DataVector>&, const double&)>(
          &potential<SpatialDim>);

  using base = Potential<SpatialDim>;
};
}  // namespace Tags
}  // namespace ScalarWave
