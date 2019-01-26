// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
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
namespace ValenciaDivClean {

// @{
/*!
 * \brief Compute the characteristic speeds for the Valencia formulation of
 * GRMHD with divergence cleaning.
 *
 * Obtaining the exact form of the characteristic speeds involves the solution
 * of a nontrivial quartic equation for the fast and slow modes. Here we make
 * use of a common approximation in the literature (e.g. \cite Gammie2003)
 * where the resulting characteristic speeds are analogous to those of the
 * Valencia formulation of the 3-D relativistic Euler system
 * (see RelativisticEuler::Valencia::characteristic_speeds),
 *
 * \f{align*}
 * \lambda_2 &= \alpha \Lambda^- - \beta_n,\\
 * \lambda_{3, 4, 5, 6, 7} &= \alpha v_n - \beta_n,\\
 * \lambda_{8} &= \alpha \Lambda^+ - \beta_n,
 * \f}
 *
 * with the substitution
 *
 * \f{align*}
 * c_s^2 \longrightarrow c_s^2 + v_A^2(1 - c_s^2)
 * \f}
 *
 * in the definition of \f$\Lambda^\pm\f$. Here \f$v_A\f$ is the Alfvén
 * speed. In addition, two more speeds corresponding to the divergence cleaning
 * mode and the longitudinal magnetic field are added,
 *
 * \f{align*}
 * \lambda_1 = -\alpha - \beta_n,\\
 * \lambda_9 = \alpha - \beta_n.
 * \f}
 *
 * \note The ordering assumed here is such that, in the Newtonian limit,
 * the exact expressions for \f$\lambda_{2, 8}\f$, \f$\lambda_{3, 7}\f$,
 * and \f$\lambda_{4, 6}\f$ should reduce to the
 * corresponding fast modes, Alfvén modes, and slow modes, respectively.
 * See \cite Dedner2002 for a detailed description of the hyperbolic
 * characterization of Newtonian MHD.  In terms of the primitive variables:
 *
 * \f{align*}
 * v^2 &= \gamma_{mn} v^m v^n \\
 * c_s^2 &= \frac{1}{h} \left[ \left( \frac{\partial p}{\partial \rho}
 * \right)_\epsilon +
 * \frac{p}{\rho^2} \left(\frac{\partial p}{\partial \epsilon}
 * \right)_\rho \right] \\
 * v_A^2 &= \frac{b^2}{b^2 + \rho h} \\
 * b^2 &= \frac{1}{W^2} \gamma_{mn} B^m B^n + \left( \gamma_{mn} B^m v^n
 * \right)^2
 * \f}
 *
 * where \f$\gamma_{mn}\f$ is the spatial metric, \f$\rho\f$ is the rest
 * mass density, \f$W = 1/\sqrt{1-v_i v^i}\f$ is the Lorentz factor, \f$h = 1 +
 * \epsilon + \frac{p}{\rho}\f$ is the specific enthalpy, \f$v^i\f$ is the
 * spatial velocity, \f$\epsilon\f$ is the specific internal energy, \f$p\f$ is
 * the pressure, and \f$B^i\f$ is the spatial magnetic field measured by an
 * Eulerian observer.
 */
template <size_t ThermodynamicDim>
std::array<DataVector, 9> characteristic_speeds(
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
        equation_of_state) noexcept;

template <size_t ThermodynamicDim>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, 9>*> char_speeds,
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
        equation_of_state) noexcept;
// @}

namespace Tags {
/// \brief Compute the characteristic speeds for the Valencia formulation of
/// GRMHD with divergence cleaning.
///
/// \details see grmhd::ValenciaDivClean::characteristic_speeds
template <typename EquationOfStateType>
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds,
                                     db::ComputeTag {
  using argument_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpecificEnthalpy<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::MagneticField<DataVector, 3>, gr::Tags::Lapse<>,
                 gr::Tags::Shift<3>, gr::Tags::SpatialMetric<3>,
                 ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<3>>,
                 hydro::Tags::EquationOfState<EquationOfStateType>>;

  using volume_tags =
      tmpl::list<hydro::Tags::EquationOfState<EquationOfStateType>>;

  using return_type = std::array<DataVector, 9>;

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
          true, EquationOfStateType::thermodynamic_dim>&
          equation_of_state) noexcept {
    characteristic_speeds<EquationOfStateType::thermodynamic_dim>(
        result, rest_mass_density, specific_internal_energy, specific_enthalpy,
        spatial_velocity, lorentz_factor, magnetic_field, lapse, shift,
        spatial_metric, unit_normal, equation_of_state);
  }
};
}  // namespace Tags

struct ComputeLargestCharacteristicSpeed {
  using argument_tags = tmpl::list<>;
  static double apply() noexcept { return 1.0; }
};

}  // namespace ValenciaDivClean
}  // namespace grmhd
