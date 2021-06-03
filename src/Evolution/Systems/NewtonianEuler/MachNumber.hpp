// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler {
/// @{
/*!
 * Compute the local Mach number, \f$\text{Ma} = v/c_s\f$,
 * where \f$v\f$ is the magnitude of the velocity, and
 * \f$c_s\f$ is the sound speed.
 */
template <typename DataType, size_t Dim, typename Fr>
void mach_number(gsl::not_null<Scalar<DataType>*> result,
                 const tnsr::I<DataType, Dim, Fr>& velocity,
                 const Scalar<DataType>& sound_speed) noexcept;

template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> mach_number(const tnsr::I<DataType, Dim, Fr>& velocity,
                             const Scalar<DataType>& sound_speed) noexcept;
/// @}

namespace Tags {
/// Compute item for the local Mach number, \f$\text{Ma}\f$.
/// \see NewtonianEuler::mach_number
///
/// Can be retrieved using `NewtonianEuler::Tags::MachNumber`
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct MachNumberCompute : MachNumber<DataType>, db::ComputeTag {
  using base = MachNumber<DataType>;

  using argument_tags =
      tmpl::list<Velocity<DataType, Dim, Fr>, SoundSpeed<DataType>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<Scalar<DataType>*>, const tnsr::I<DataType, Dim, Fr>&,
      const Scalar<DataType>&)>(&mach_number<DataType, Dim, Fr>);
};
}  // namespace Tags
}  // namespace NewtonianEuler
