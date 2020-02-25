// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl

namespace NewtonianEuler {
//@{
/*!
 * Compute the internal energy density, \f$\rho \epsilon\f$,
 * where \f$\rho\f$ is the mass density, and \f$\epsilon\f$ is the
 * specific internal energy.
 */
template <typename DataType>
void internal_energy_density(
    gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& mass_density,
    const Scalar<DataType>& specific_internal_energy) noexcept;

template <typename DataType>
Scalar<DataType> internal_energy_density(
    const Scalar<DataType>& mass_density,
    const Scalar<DataType>& specific_internal_energy) noexcept;
//@}

//@{
/*!
 * Compute the kinetic energy density, \f$\rho v^2/2\f$,
 * where \f$\rho\f$ is the mass density, and \f$v\f$ is the
 * magnitude of the velocity.
 */
template <typename DataType, size_t Dim, typename Fr>
void kinetic_energy_density(
    gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim, Fr>& velocity) noexcept;

template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> kinetic_energy_density(
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim, Fr>& velocity) noexcept;
//@}

//@{
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
//@}

//@{
/*!
 * Compute the ram pressure, \f$\rho v^i v^j\f$, where \f$\rho\f$ is the
 * mass density, and \f$v^i\f$ is the velocity.
 */
template <typename DataType, size_t Dim, typename Fr>
void ram_pressure(gsl::not_null<tnsr::II<DataType, Dim, Fr>*> result,
                  const Scalar<DataType>& mass_density,
                  const tnsr::I<DataType, Dim, Fr>& velocity) noexcept;

template <typename DataType, size_t Dim, typename Fr>
tnsr::II<DataType, Dim, Fr> ram_pressure(
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim, Fr>& velocity) noexcept;
//@}

//@{
/*!
 * Compute the specific kinetic energy, \f$v^2/2\f$,
 * where \f$v\f$ is the magnitude of the velocity.
 */
template <typename DataType, size_t Dim, typename Fr>
void specific_kinetic_energy(
    gsl::not_null<Scalar<DataType>*> result,
    const tnsr::I<DataType, Dim, Fr>& velocity) noexcept;

template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> specific_kinetic_energy(
    const tnsr::I<DataType, Dim, Fr>& velocity) noexcept;
//@}

namespace Tags {
/// Compute item for the internal energy density, \f$\rho \epsilon\f$.
/// \see NewtonianEuler::internal_energy_density
///
/// Can be retrieved using `NewtonianEuler::Tags::InternalEnergyDensity`
template <typename DataType>
struct InternalEnergyDensityCompute : InternalEnergyDensity<DataType>,
                                      db::ComputeTag {
  using base = InternalEnergyDensity<DataType>;

  using argument_tags =
      tmpl::list<MassDensity<DataType>, SpecificInternalEnergy<DataType>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function =
      static_cast<void (*)(const gsl::not_null<Scalar<DataType>*>,
                           const Scalar<DataType>&, const Scalar<DataType>&)>(
          &internal_energy_density<DataType>);
};

/// Compute item for the kinetic energy density, \f$\rho v^2/2\f$.
/// \see NewtonianEuler::kinetic_energy_density
///
/// Can be retrieved using `NewtonianEuler::Tags::KineticEnergyDensity`
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct KineticEnergyDensityCompute : KineticEnergyDensity<DataType>,
                                     db::ComputeTag {
  using base = KineticEnergyDensity<DataType>;

  using argument_tags =
      tmpl::list<MassDensity<DataType>, Velocity<DataType, Dim, Fr>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<Scalar<DataType>*>, const Scalar<DataType>&,
      const tnsr::I<DataType, Dim, Fr>&)>(
      &kinetic_energy_density<DataType, Dim, Fr>);
};

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

/// Compute item for the specific kinetic energy, \f$v^2/2\f$.
/// \see NewtonianEuler::specific_kinetic_energy
///
/// Can be retrieved using `NewtonianEuler::Tags::SpecificKineticEnergy`
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct SpecificKineticEnergyCompute : SpecificKineticEnergy<DataType>,
                                      db::ComputeTag {
  using base = SpecificKineticEnergy<DataType>;

  using argument_tags = tmpl::list<Velocity<DataType, Dim, Fr>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function =
      static_cast<void (*)(const gsl::not_null<Scalar<DataType>*>,
                           const tnsr::I<DataType, Dim, Fr>&)>(
          &specific_kinetic_energy<DataType, Dim, Fr>);
};
}  // namespace Tags
}  // namespace NewtonianEuler
