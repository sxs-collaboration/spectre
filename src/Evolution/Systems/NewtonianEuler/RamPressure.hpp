// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler {
/// @{
/*!
 * Compute the ram pressure, \f$\rho v^i v^j\f$, where \f$\rho\f$ is the
 * mass density, and \f$v^i\f$ is the velocity.
 */
template <typename DataType, size_t Dim, typename Fr>
void ram_pressure(gsl::not_null<tnsr::II<DataType, Dim, Fr>*> result,
                  const Scalar<DataType>& mass_density,
                  const tnsr::I<DataType, Dim, Fr>& velocity);

template <typename DataType, size_t Dim, typename Fr>
tnsr::II<DataType, Dim, Fr> ram_pressure(
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim, Fr>& velocity);
/// @}

namespace Tags {
/// Compute item for the ram pressure, \f$\rho v^i v^j\f$.
/// \see NewtonianEuler::ram_pressure
///
/// Can be retrieved using `NewtonianEuler::Tags::RamPressure`
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct RamPressureCompute : RamPressure<DataType, Dim, Fr>, db::ComputeTag {
  using base = RamPressure<DataType, Dim, Fr>;

  using argument_tags =
      tmpl::list<MassDensity<DataType>, Velocity<DataType, Dim, Fr>>;

  using return_type = tnsr::II<DataType, Dim, Fr>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<tnsr::II<DataType, Dim, Fr>*>,
      const Scalar<DataType>&, const tnsr::I<DataType, Dim, Fr>&)>(
      &ram_pressure<DataType, Dim, Fr>);
};
}  // namespace Tags
}  // namespace NewtonianEuler
