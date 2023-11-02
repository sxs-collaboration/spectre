// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/IrrotationalBns/Geometry.hpp"
#include "Elliptic/Systems/IrrotationalBns/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
namespace IrrotationalBns {
template <Geometry BackgroundGeometry>
struct Fluxes;
template <Geometry BackgroundGeometry>
struct Sources;
}  // namespace IrrotationalBns
/// \endcond

namespace IrrotationalBns {

/*!
 * \brief Compute the fluxes \f$F^i = U^i\f$ for the Irrotational BNS
 * equation on a flat spatial metric in Cartesian coordinates.
 */
void flat_potential_fluxes(
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::i<DataVector, 3>& auxiliary_velocity);
/*!
 * \brief Compute the generic_fluxes \f$F^i = U^i\f$ for the Irrotational BNS
 * equation.
 */
void curved_potential_fluxes(
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    const tnsr::i<DataVector, 3>& auxiliary_velocity);

/*!
 * \brief Compute the sources \f$S=U^i\pd_i\frac{\alpha}{h}\f$
 * for a flat-spatial metric in Cartesian coordinates for the
 * Irrotational BNS equation$.
 */
void add_flat_potential_sources(
    gsl::not_null<Scalar<DataVector>*> source_for_potential,
    const tnsr::i<DataVector, 3>& auxiliary_velocity,
    const tnsr::i<DataVector, 3>& log_deriv_lapse_over_specific_enthalpy);
/*!
 * \brief Add the sources \f$S=-\Gamma^i_{ij}v^j\f$
 * for the curved-space Irrotational BNS equation on a spatial metric
 * \f$\gamma_{ij}\f$.
 *
 * These sources arise from the non-principal part of the Laplacian on a
 * non-Euclidean background.
 */
void add_curved_potential_sources(
    gsl::not_null<Scalar<DataVector>*> source_for_potential,
    const tnsr::i<DataVector, 3>& log_deriv_of_lapse_over_specific_enthalpy,
    const tnsr::i<DataVector, 3>& christoffel_contracted,
    const tnsr::I<DataVector, 3>& flux_for_potential);

/*!
 * \brief Compute the fluxes \f$F^i_j=\delta^i_j u(x)\f$ for the auxiliary
 * field in the first-order formulation of the Irrotational BNS equation.
 *
 * \see IrrotationalBns::FirstOrderSystem
 */
void auxiliary_fluxes(
    gsl::not_null<tnsr::Ij<DataVector, 3>*> flux_for_auxiliary,
    const Scalar<DataVector>& velocity_potential,
    const tnsr::Ij<DataVector, 3>& rotational_shift_stress);

/*!
 * \brief Compute the sources \f$Sf$ for the auxiliary
 * field in the first-order formulation of the Irrotational BNS equation.
 *
 * \see IrrotationalBns::FirstOrderSystem
 */
void add_auxiliary_sources_without_flux_christoffels(
    gsl::not_null<tnsr::i<DataVector, 3>*> auxiliary_sources,
    const Scalar<DataVector>& velocity_potential,
    const tnsr::i<DataVector, 3>& auxiliary_velocity,
    const tnsr::i<DataVector, 3>& div_rotational_shift_stress,
    const tnsr::i<DataVector, 3>& fixed_sources);

/*!
 * \brief Compute the fluxes \f$F^i_j=\delta^i_j u(x)\f$ for the auxiliary
 * field in the first-order formulation of the Irrotational BNS equation .
 *
 * \see IrrotationalBns::FirstOrderSystem
 */
void add_auxiliary_source_flux_christoffels(
    gsl::not_null<tnsr::i<DataVector, 3>*> auxiliary_sources,
    const Scalar<DataVector>& velocity_potential,
    const tnsr::i<DataVector, 3>& christoffel_contracted,
    const tnsr::Ijk<DataVector, 3>& christoffel,
    const tnsr::Ij<DataVector, 3>& rotational_shift_stress);
/*!
 * \brief Compute the fluxes \f$F^i_A\f$ for the IrrotationalBns equation on
 * a flat metric in Cartesian coordinates.
 *
 * \see IrrotationalBns::FirstOrderSystem
 */
template <>
struct Fluxes<Geometry::FlatCartesian> {
  using argument_tags = tmpl::list<Tags::RotationalShiftStress>;
  using volume_tags = tmpl::list<>;
  // Order is prmal
  static void apply(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
                    const tnsr::Ij<DataVector, 3>& /*rotational_shift_stress*/,
                    const tnsr::i<DataVector, 3>& auxiliary_velocity);
  static void apply(gsl::not_null<tnsr::Ij<DataVector, 3>*> flux_for_auxiliary,
                    const tnsr::Ij<DataVector, 3>& rotational_shift_stress,
                    const Scalar<DataVector>& velocity_potential);
};

/*!
 * \brief Compute the fluxes \f$F^i_A\f$ for the curved-space Irrotatational BNS
 * equations on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see IrrotationalBns::FirstOrderSystem
 */
template <>
struct Fluxes<Geometry::Curved> {
  using argument_tags =
      tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, 3>,
                 Tags::RotationalShiftStress>;
  using volume_tags = tmpl::list<>;
  static void apply(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
                    const tnsr::II<DataVector, 3>& inv_spatial_metric,
                    const tnsr::Ij<DataVector, 3>& rotational_shift_stress,
                    const tnsr::i<DataVector, 3>& auxiliary_velocity);
  static void apply(gsl::not_null<tnsr::Ij<DataVector, 3>*> flux_for_auxiliary,
                    const tnsr::II<DataVector, 3>& inv_spatial_metric,
                    const tnsr::Ij<DataVector, 3>& rotational_shift_stress,
                    const Scalar<DataVector>& velocity_potential);
};

/*!
 * \brief Add the sources \f$S_A\f$ for the IrrotationalBns equation on a flat
 * metric in Cartesian coordinates.
 *
 * \see IrrotationalBns::FirstOrderSystem
 */
template <>
struct Sources<Geometry::FlatCartesian> {
  using argument_tags =
      tmpl::list<Tags::DerivLogLapseOverSpecificEnthalpy,
                 Tags::AuxiliaryVelocity, Tags::DivergenceRotationalShiftStress,
                 Tags::FixedSources>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> equation_for_potential,
      const tnsr::i<DataVector, 3>& log_deriv_of_lapse_over_specific_enthalpy,
      const tnsr::i<DataVector, 3>& auxiliary_velocity,
      const tnsr::i<DataVector, 3>& /*div_rotational_shift_stress*/,
      const tnsr::i<DataVector, 3>& /*fixed_sources*/,
      const Scalar<DataVector>& /*velocity_potential*/,
      const tnsr::I<DataVector, 3>& flux_for_potential);
  static void apply(
      gsl::not_null<tnsr::i<DataVector, 3>*> equation_for_auxiliary_velocity,
      const tnsr::i<DataVector,
                    3>& /*log_deriv_of_lapse_over_specific_enthalpy*/,
      const tnsr::i<DataVector, 3>& auxiliary_velocity,
      const tnsr::i<DataVector, 3>& div_rotational_shift_stress,
      const tnsr::i<DataVector, 3>& fixed_sources,
      const Scalar<DataVector>& velocity_potential);
};

/*!
 * \brief Add the sources \f$S_A\f$ for the curved-space Irrotatioanl BNS
 * equation on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see IrrotationalBns::FirstOrderSystem
 */
template <>
struct Sources<Geometry::Curved> {
  using argument_tags = tmpl::list<
      Tags::DerivLogLapseOverSpecificEnthalpy,
      gr::Tags::SpatialChristoffelSecondKind<DataVector, 3>,
      gr::Tags::SpatialChristoffelSecondKindContracted<DataVector, 3>,
      Tags::DivergenceRotationalShiftStress, Tags::AuxiliaryVelocity,
      Tags::RotationalShiftStress, Tags::FixedSources>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> equation_for_potential,
      const tnsr::i<DataVector, 3>& log_deriv_of_lapse_over_specific_enthalpy,
      const tnsr::i<DataVector, 3>& christoffel_contracted,
      const tnsr::Ijk<DataVector, 3>& /*christoffel*/,
      const tnsr::i<DataVector, 3>& /*div_rotational_shift_stress*/,
      const tnsr::i<DataVector, 3>& /*auxiliary_velocity*/,
      const tnsr::Ij<DataVector, 3>& /*rotational_shift_stress*/,
      const tnsr::i<DataVector, 3>& /*fixed_sources*/,
      const Scalar<DataVector>& /*velocity_potential*/,
      const tnsr::I<DataVector, 3>& flux_for_potential);
  static void apply(
      gsl::not_null<tnsr::i<DataVector, 3>*> equation_for_auxiliary_velocity,
      const tnsr::i<DataVector,
                    3>& /*log_deriv_of_lapse_over_specific_enthalpy*/,
      const tnsr::i<DataVector, 3>& christoffel_contracted,
      const tnsr::Ijk<DataVector, 3>& christoffel,
      const tnsr::i<DataVector, 3>& div_rotational_shift_stress,
      const tnsr::i<DataVector, 3>& auxiliary_velocity,
      const tnsr::Ij<DataVector, 3>& rotational_shift_stress,
      const tnsr::i<DataVector, 3>& fixed_sources,
      const Scalar<DataVector>& velocity_potential);
};

}  // namespace IrrotationalBns
