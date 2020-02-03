// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace Tags {
template <typename Tag>
struct Normalized;
}  // namespace Tags
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace RelativisticEuler {
namespace Valencia {

// @{
/*!
 * \brief Compute the characteristic speeds for the Valencia formulation
 * of the relativistic Euler system.
 *
 * The principal symbol of the system is diagonalized so that the elements of
 * the diagonal matrix are the \f$(\text{Dim} + 2)\f$ characteristic speeds
 *
 * \f{align*}
 * \lambda_1 &= \alpha \Lambda^- - \beta_n,\\
 * \lambda_{i + 1} &= \alpha v_n - \beta_n,\quad i = 1,...,\text{Dim}\\
 * \lambda_{\text{Dim} + 2} &= \alpha \Lambda^+ - \beta_n,
 * \f}
 *
 * where \f$\alpha\f$ is the lapse, \f$\beta_n = n_i \beta^i\f$ and
 * \f$v_n = n_i v^i\f$ are the projections of the shift \f$\beta^i\f$ and the
 * spatial velocity \f$v^i\f$ onto the normal one-form \f$n_i\f$, respectively,
 * and
 *
 * \f{align*}
 * \Lambda^{\pm} &= \dfrac{1}{1 - v^2 c_s^2}\left[ v_n (1- c_s^2) \pm
 * c_s\sqrt{\left(1 - v^2\right)\left[1 - v^2 c_s^2 - v_n^2(1 - c_s^2)\right]}
 * \right],
 * \f}
 *
 * where \f$v^2 = \gamma_{ij}v^iv^j\f$ is the magnitude squared of the spatial
 * velocity, and \f$c_s\f$ is the sound speed.
 */
template <size_t Dim>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, Dim + 2>*> char_speeds,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::I<DataVector, Dim>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::i<DataVector, Dim>& normal) noexcept;

template <size_t Dim>
std::array<DataVector, Dim + 2> characteristic_speeds(
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::I<DataVector, Dim>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::i<DataVector, Dim>& normal) noexcept;
// @}

namespace Tags {

template <size_t Dim>
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds<Dim>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicSpeeds<Dim>;
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<>, gr::Tags::Shift<Dim>,
                 gr::Tags::SpatialMetric<Dim>,
                 hydro::Tags::SpatialVelocity<DataVector, Dim>,
                 hydro::Tags::SoundSpeedSquared<DataVector>,
                 ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim>>>;

  using volume_tags = tmpl::list<>;

  using return_type = std::array<DataVector, Dim + 2>;

  static constexpr void function(
      const gsl::not_null<return_type*> result, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim>& shift,
      const tnsr::ii<DataVector, Dim>& spatial_metric,
      const tnsr::I<DataVector, Dim>& spatial_velocity,
      const Scalar<DataVector>& sound_speed_squared,
      const tnsr::i<DataVector, Dim>& unit_normal) noexcept {
    characteristic_speeds(
        result, lapse, shift, spatial_velocity,
        dot_product(spatial_velocity, spatial_velocity, spatial_metric),
        sound_speed_squared, unit_normal);
  }
};

}  // namespace Tags

struct ComputeLargestCharacteristicSpeed {
  using argument_tags = tmpl::list<>;
  static double apply() noexcept { return 1.0; }
};

}  // namespace Valencia
}  // namespace RelativisticEuler
