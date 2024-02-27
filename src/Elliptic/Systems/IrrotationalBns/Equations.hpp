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
 * \brief Compute the fluxes \f$F^i = U^i = U_i\f$ for the Irrotational BNS
 * equation on a flat spatial metric in Cartesian coordinates.
 */
void flat_potential_fluxes(
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::II<DataVector, 3>& rotational_shift_stress,
    const tnsr::i<DataVector, 3>& auxiliary_velocity);

/*!
 * \brief Compute the fluxes $F^i=\gamma^{ij} n_j \Phi  + n_i \Phi
 * \frac{B^iB^j}{\alpha^2}  $ where $n_j$ is the
 * `face_normal`.
 *
 * The `face_normal_vector` is $\gamma^{ij} n_j$.
 */
void fluxes_on_face(
    gsl::not_null<tnsr::I<DataVector, 3>*> face_flux_for_potential,
    const tnsr::I<DataVector, 3>& face_normal_vector,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::II<DataVector, 3>& rotational_shift_stress,
    const Scalar<DataVector>& velocity_potential);

/*!
 * \brief Compute the generic fluxes \f$F^i = U^i\f$ for the Irrotational BNS
 * equation for the velocity potential.
 */
void curved_potential_fluxes(
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::II<DataVector, 3>& rotational_shift_stress,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    const tnsr::i<DataVector, 3>& auxiliary_velocity);

/*!
 * \brief Compute the sources \f$S=U^i\pd_i\log\left(\frac{\alpha}{h}\right)\f$
 * for a flat-spatial metric in Cartesian coordinates for the
 * Irrotational BNS equation$.
 */
void add_flat_potential_sources(
    gsl::not_null<Scalar<DataVector>*> source_for_potential,
    const tnsr::i<DataVector, 3>& log_deriv_lapse_over_specific_enthalpy,
    const tnsr::I<DataVector, 3>& flux_for_potential);
/*!
 * \brief Add the sources \f$S=-\Gamma^i_{ij}U^j\f$
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
 * \brief Compute the fluxes \f$F^i_A\f$ for the IrrotationalBns equation on
 * a flat metric in Cartesian coordinates.
 *
 * \see IrrotationalBns::FirstOrderSystem
 */
template <>
struct Fluxes<Geometry::FlatCartesian> {
  using argument_tags = tmpl::list<Tags::RotationalShiftStress<DataVector>>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  // Order is prmal
  static void apply(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
                    const tnsr::II<DataVector, 3>& rotational_shift_stress,
                    const tnsr::i<DataVector, 3>& auxiliary_velocity);
  static void apply(gsl::not_null<tnsr::I<DataVector, 3>*> flux_on_face,
                    const tnsr::II<DataVector, 3>& rotational_shift_stress,
                    const tnsr::i<DataVector, 3>& face_normal,
                    const tnsr::I<DataVector, 3>& face_normal_vector,
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
                 Tags::RotationalShiftStress<DataVector>>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  static void apply(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
                    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
                    const tnsr::II<DataVector, 3>& rotational_shift_stress,
                    const tnsr::i<DataVector, 3>& auxiliary_velocity);
  static void apply(gsl::not_null<tnsr::I<DataVector, 3>*> flux_on_face,
                    const tnsr::II<DataVector, 3>& rotational_shift_stress,
                    const tnsr::i<DataVector, 3>& face_normal,
                    const tnsr::I<DataVector, 3>& face_normal_vector,
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
      tmpl::list<Tags::DerivLogLapseOverSpecificEnthalpy<DataVector>>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> equation_for_potential,
      const tnsr::i<DataVector, 3>& log_deriv_of_lapse_over_specific_enthalpy,
      const tnsr::I<DataVector, 3>& flux_for_potential);
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
      Tags::DerivLogLapseOverSpecificEnthalpy<DataVector>,
      gr::Tags::SpatialChristoffelSecondKindContracted<DataVector, 3>>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> equation_for_potential,
      const tnsr::i<DataVector, 3>& log_deriv_of_lapse_over_specific_enthalpy,
      const tnsr::i<DataVector, 3>& christoffel_contracted,
      const Scalar<DataVector>& velocity_potential,
      const tnsr::I<DataVector, 3>& flux_for_potential);
};

}  // namespace IrrotationalBns
