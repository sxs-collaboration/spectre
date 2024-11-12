// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/BnsInitialData/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
namespace BnsInitialData {
struct Fluxes;
struct Sources;
}  // namespace BnsInitialData
/// \endcond

namespace BnsInitialData {

/*!
 * \brief Compute the fluxes
 * \f[ F^i = \gamma^{ij} n_j \Phi  - n_j \Phi
 * \frac{B^iB^j}{\alpha^2}  \f] where \f$ n_j\f$ is the
 * `face_normal`.
 *
 * The `face_normal_vector` is \f$ \gamma^{ij} n_j\f$.
 */
void fluxes_on_face(
    gsl::not_null<tnsr::I<DataVector, 3>*> face_flux_for_potential,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::I<DataVector, 3>& face_normal_vector,
    const tnsr::II<DataVector, 3>& rotational_shift_stress,
    const Scalar<DataVector>& velocity_potential);

/*!
 * \brief Compute the generic fluxes \f$ F^i = D^i \Phi - B^jD_j\Phi /\alpha^2
 * B^i \f$ for the Irrotational BNS equation for the velocity potential.
 */
void potential_fluxes(
    gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
    const tnsr::II<DataVector, 3>& rotational_shift_stress,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
    const tnsr::i<DataVector, 3>& velocity_potential_gradient);

/*!
 * \brief Add the sources \f$S=-\Gamma^i_{ij}F^j - \D_j \left(\alpha \rho /
 * h\right)  F^j\f$ for the curved-space Irrotational BNS equation on a spatial
 * metric \f$\gamma_{ij}\f$.
 *
 * These sources arise from the non-principal part of the Laplacian on a
 * non-Euclidean background.
 */
void add_potential_sources(
    gsl::not_null<Scalar<DataVector>*> source_for_potential,
    const tnsr::i<DataVector, 3>&
        log_deriv_lapse_times_density_over_specific_enthalpy,
    const tnsr::i<DataVector, 3>& christoffel_contracted,
    const tnsr::I<DataVector, 3>& flux_for_potential);

/*!
 * \brief Compute the fluxes \f$F^i\f$ for the curved-space Irrotatational BNS
 * equations on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see IrrotationalBns::FirstOrderSystem
 */
struct Fluxes {
  using argument_tags =
      tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, 3>,
                 Tags::RotationalShiftStress<DataVector>>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  static void apply(gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_potential,
                    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
                    const tnsr::II<DataVector, 3>& rotational_shift_stress,
                    const Scalar<DataVector>& velocity_potential,
                    const tnsr::i<DataVector, 3>& velocity_potential_gradient);
  static void apply(gsl::not_null<tnsr::I<DataVector, 3>*> flux_on_face,
                    const tnsr::II<DataVector, 3>& inverse_spatial_metric,
                    const tnsr::II<DataVector, 3>& rotational_shift_stress,
                    const tnsr::i<DataVector, 3>& face_normal,
                    const tnsr::I<DataVector, 3>& face_normal_vector,
                    const Scalar<DataVector>& velocity_potential);
};

/*!
 * \brief Add the sources \f$S_A\f$ for the curved-space Irrotatioanl BNS
 * equation on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see IrrotationalBns::FirstOrderSystem
 */
struct Sources {
  using argument_tags = tmpl::list<
      Tags::DerivLogLapseTimesDensityOverSpecificEnthalpy<DataVector>,
      gr::Tags::SpatialChristoffelSecondKindContracted<DataVector, 3>>;
  using const_global_cache_tags = tmpl::list<>;
  static void apply(gsl::not_null<Scalar<DataVector>*> equation_for_potential,
                    const tnsr::i<DataVector, 3>&
                        log_deriv_lapse_times_density_over_specific_enthalpy,
                    const tnsr::i<DataVector, 3>& christoffel_contracted,
                    const Scalar<DataVector>& velocity_potential,
                    const tnsr::I<DataVector, 3>& flux_for_potential);
};

}  // namespace BnsInitialData
