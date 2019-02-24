// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"               // for get
#include "DataStructures/Tensor/TypeAliases.hpp"          // IWYU pragma: keep
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::MassDensity
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::SoundSpeed
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::SpecificInternalEnergy
// IWYU pragma: no_forward_declare NewtonianEuler::Tags::Velocity
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare hydro::Tags::EquationOfState

/// \cond
class DataVector;
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace NewtonianEuler {

// @{
/*!
 * \brief Compute the characteristic speeds of NewtonianEuler system
 *
 * The principal symbol of the system is diagonalized so that the elements of
 * the diagonal matrix are the characteristic speeds
 *
 * \f{align*}
 * \lambda_1 &= v_n - c_s,\\
 * \lambda_{i + 1} &= v_n,\\
 * \lambda_{\text{Dim} + 2} &= v_n + c_s,
 * \f}
 *
 * where \f$i = 1,...,\text{Dim}\f$,
 * \f$v_n = n_i v^i\f$ is the velocity projected onto the normal,
 * and \f$c_s\f$ is the sound speed.
 */
template <size_t Dim>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, Dim + 2>*> char_speeds,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed,
    const tnsr::i<DataVector, Dim>& normal) noexcept;

template <size_t Dim>
std::array<DataVector, Dim + 2> characteristic_speeds(
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed,
    const tnsr::i<DataVector, Dim>& normal) noexcept;
// @}

namespace Tags {

template <size_t Dim>
struct CharacteristicSpeedsCompute : CharacteristicSpeeds<Dim>, db::ComputeTag {
  using argument_tags =
      tmpl::list<Velocity<DataVector, Dim>, SoundSpeed<DataVector>,
                 ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim>>>;

  using return_type = std::array<DataVector, Dim + 2>;

  static constexpr void function(
      const gsl::not_null<return_type*> result,
      const tnsr::I<DataVector, Dim>& velocity,
      const Scalar<DataVector>& sound_speed,
      const tnsr::i<DataVector, Dim>& normal) noexcept {
    characteristic_speeds<Dim>(result, velocity, sound_speed, normal);
  }
};
}  // namespace Tags

template <size_t Dim>
struct ComputeLargestCharacteristicSpeed {
  using argument_tags =
      tmpl::list<Tags::Velocity<DataVector, Dim>, Tags::SoundSpeed<DataVector>>;

  static double apply(const tnsr::I<DataVector, Dim>& velocity,
                      const Scalar<DataVector>& sound_speed) noexcept {
    return max(get(magnitude(velocity)) + get(sound_speed));
  }
};

}  // namespace NewtonianEuler
