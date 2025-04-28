// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservativeOptions.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;
/// \endcond

namespace grmhd {
namespace ValenciaDivClean {
/*!
 * \brief Compute the primitive variables from the conservative variables
 *
 * For the Valencia formulation of the GRMHD system with divergence cleaning,
 * the conversion of the evolved conserved variables to the primitive variables
 * cannot be expressed in closed analytic form and requires a root find.
 *
 * [Siegel {\em et al}, The Astrophysical Journal 859:71(2018)]
 * (http://iopscience.iop.org/article/10.3847/1538-4357/aabcc5/meta)
 * compares several inversion methods.
 *
 * If `ErrorOnFailure` is `false` then the returned `bool` will be `false` if
 * recovery failed and `true` if it succeeded.
 *
 * If `EnforcePhysicality` is `false` then the hydrodynamic inversion will
 * return with an error if the input conservatives are unphysical, i.e., if no
 * solution exits. The exact behavior is governed by `ErrorOnFailure`.
 *
 * \note
 * Here, we outline the algebra for computing the spatial velocity
 * \f$ v^i \f$:
 *
 * We start from definition of the momentum density \f$ S_i \f$:
 * \f[
 * \begin{aligned}
 * S_i &= (\rho h)^* W^2 v_i - \alpha b^0 b_i \\
 *     &= \left(\rho h + \frac{B^2}{W^2} + (v \cdot B)^2\right) W^2 v_i
 *        - \alpha \left(\frac{W}{\alpha}(v \cdot B)\right)
 *        \left(\frac{B_i}{W} + (v \cdot B) W v_i\right) \\
 *     &= \left(\rho h + \frac{B^2}{W^2}\right) W^2 v_i - (v \cdot B) B_i
 * \end{aligned}
 * \f]
 *
 * From the above, we obtain:
 * \f[
 * \begin{aligned}
 * S^i &= \left(\rho h + \frac{B^2}{W^2}\right) W^2 v^i - (v \cdot B) B^i \\
 * S \cdot B &= \left(\rho h + \frac{B^2}{W^2}\right) W^2 (v \cdot B)
 *              - (v \cdot B) B^2 \\
 *          &= \rho h W^2 (v \cdot B) \\
 * v^i &= \frac{1}{\sqrt{\gamma} (\rho h W^2 + B^2)} \tilde{S}^i
 *       + \frac{S \cdot B}{\rho h W^2 (\rho h W^2 + B^2)} B^i
 * \end{aligned}
 * \f]
 */
template <typename OrderedListOfPrimitiveRecoverySchemes,
          bool ErrorOnFailure = true>
struct PrimitiveFromConservative {
  using return_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::Temperature<DataVector>>;

  using argument_tags = tmpl::list<
      grmhd::ValenciaDivClean::Tags::TildeD,
      grmhd::ValenciaDivClean::Tags::TildeYe,
      grmhd::ValenciaDivClean::Tags::TildeTau,
      grmhd::ValenciaDivClean::Tags::TildeS<>,
      grmhd::ValenciaDivClean::Tags::TildeB<>,
      grmhd::ValenciaDivClean::Tags::TildePhi,
      gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>,
      hydro::Tags::GrmhdEquationOfState,
      grmhd::ValenciaDivClean::Tags::PrimitiveFromConservativeOptions>;

  template <bool EnforcePhysicality = true>
  static bool apply(
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> electron_fraction,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> magnetic_field,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
      gsl::not_null<Scalar<DataVector>*> lorentz_factor,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<Scalar<DataVector>*> temperature,
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
      const Scalar<DataVector>& tilde_tau,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_phi,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
      const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&
          primitive_from_conservative_options);

 private:
  template <bool EnforcePhysicality, typename EosType>
  static bool impl(
      gsl::not_null<Scalar<DataVector>*> rest_mass_density,
      gsl::not_null<Scalar<DataVector>*> electron_fraction,
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> spatial_velocity,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> magnetic_field,
      gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
      gsl::not_null<Scalar<DataVector>*> lorentz_factor,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<Scalar<DataVector>*> temperature,
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
      const Scalar<DataVector>& tilde_tau,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_phi,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const EosType& equation_of_state,
      const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&
          primitive_from_conservative_options);

  // Use Kastaun hydro inversion if B is dynamically unimportant
  static constexpr bool use_hydro_optimization = true;
};
}  // namespace ValenciaDivClean
}  // namespace grmhd
