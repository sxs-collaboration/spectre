// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ConstraintDamping/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
// IWYU pragma: no_include "PointwiseFunctions/Hydro/Tags.hpp"

/// \cond
class DataVector;
namespace Tags {
template <typename Tag>
struct Normalized;
}  // namespace Tags
/// \endcond

// IWYU pragma:  no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma:  no_forward_declare EquationsOfSTate::IdealFluid
// IWYU pragma: no_forward_declare Tensor

namespace grmhd {
namespace GhValenciaDivClean {

// @{
/*!
 * \brief Compute the characteristic speeds for the Valencia formulation of
 * GRMHD with divergence cleaning with a dynamical spacetime evolved in the
 * Generalized Harmonic formulation.
 */
template <size_t ThermodynamicDim>
std::array<DataVector, 13> characteristic_speeds(
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const Scalar<DataVector>& gh_constraint_gamma1) noexcept;

template <size_t ThermodynamicDim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 13>*> char_speeds,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const Scalar<DataVector>& gh_constraint_gamma1) noexcept;
// @}

namespace Tags {
/*!
 * \brief Compute the characteristic speeds for the Valencia formulation of
 * GRMHD with divergence cleaning with a dynamical spacetime evolved in the
 * Generalized Harmonic formulation.
 *
 * \details see grmhd::GhValenciaDivClean::characteristic_speeds
 */
template <typename EquationOfStateType>
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds,
                                     db::ComputeTag {
  using base = Tags::CharacteristicSpeeds;
  using argument_tags = tmpl::list<
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::SpecificInternalEnergy<DataVector>,
      hydro::Tags::SpecificEnthalpy<DataVector>,
      hydro::Tags::SpatialVelocity<DataVector, 3>,
      hydro::Tags::LorentzFactor<DataVector>,
      hydro::Tags::MagneticField<DataVector, 3>, gr::Tags::Lapse<>,
      gr::Tags::Shift<3>, gr::Tags::SpatialMetric<3>,
      ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<3>>,
      hydro::Tags::EquationOfState<EquationOfStateType>,
      GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1>;

  using volume_tags =
      tmpl::list<hydro::Tags::EquationOfState<EquationOfStateType>>;

  using return_type = std::array<DataVector, 13>;

  static constexpr void function(
      const gsl::not_null<return_type*> result,
      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& specific_internal_energy,
      const Scalar<DataVector>& specific_enthalpy,
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
      const Scalar<DataVector>& lorentz_factor,
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::i<DataVector, 3>& unit_normal,
      const EquationsOfState::EquationOfState<
          true, EquationOfStateType::thermodynamic_dim>& equation_of_state,
      const Scalar<DataVector>& gh_constraint_gamma1) noexcept {
    characteristic_speeds<EquationOfStateType::thermodynamic_dim>(
        result, rest_mass_density, specific_internal_energy, specific_enthalpy,
        spatial_velocity, lorentz_factor, magnetic_field, lapse, shift,
        spatial_metric, unit_normal, equation_of_state, gh_constraint_gamma1);
  }
};

struct ComputeLargestCharacteristicSpeed : LargestCharacteristicSpeed,
                                           db::ComputeTag {
  using base = LargestCharacteristicSpeed;
  using type = typename base::type;
  using argument_tags = tmpl::list<
      ::GeneralizedHarmonic::ConstraintDamping::Tags::ConstraintGamma1,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<3, Frame::Inertial, DataVector>,
      gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>;

  using return_type = double;

  static void function(
      const gsl::not_null<double*> speed,
      const Scalar<DataVector>& gh_constraint_gamma1,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric) noexcept;
};
}  // namespace Tags
}  // namespace GhValenciaDivClean
}  // namespace grmhd
