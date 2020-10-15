// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Poisson {

/*!
 * \brief Compute the fluxes \f$F^i=\partial_i u(x)\f$ for the Poisson
 * equation on a flat spatial metric in Cartesian coordinates.
 */
template <size_t Dim>
void euclidean_fluxes(gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
                      const tnsr::i<DataVector, Dim>& field_gradient) noexcept;

/*!
 * \brief Compute the fluxes \f$F^i=\sqrt{\gamma}\gamma^{ij}\partial_j u(x)\f$
 * for the curved-space Poisson equation on a spatial metric \f$\gamma_{ij}\f$.
 */
template <size_t Dim>
void non_euclidean_fluxes(
    gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
    const Scalar<DataVector>& det_spatial_metric,
    const tnsr::i<DataVector, Dim>& field_gradient) noexcept;

/*!
 * \brief Compute the fluxes \f$F^i_j=\delta^i_j u(x)\f$ for the auxiliary
 * field in the first-order formulation of the Poisson equation.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim>
void auxiliary_fluxes(
    gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_gradient,
    const Scalar<DataVector>& field) noexcept;

/*!
 * \brief Compute the fluxes \f$F^i_A\f$ for the Poisson equation on a flat
 * metric in Cartesian coordinates.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim>
struct EuclideanFluxes {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
      const tnsr::i<DataVector, Dim>& field_gradient) noexcept {
    euclidean_fluxes(flux_for_field, field_gradient);
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_gradient,
      const Scalar<DataVector>& field) noexcept {
    auxiliary_fluxes(flux_for_gradient, field);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

/*!
 * \brief Compute the fluxes \f$F^i_A\f$ for the curved-space Poisson equation
 * on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim>
struct NonEuclideanFluxes {
  using argument_tags = tmpl::list<
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>,
      gr::Tags::DetSpatialMetric<DataVector>>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
      const tnsr::II<DataVector, Dim>& inv_spatial_metric,
      const Scalar<DataVector>& det_spatial_metric,
      const tnsr::i<DataVector, Dim>& field_gradient) noexcept {
    non_euclidean_fluxes(flux_for_field, inv_spatial_metric, det_spatial_metric,
                         field_gradient);
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_gradient,
      const tnsr::II<DataVector, Dim>& /*inv_spatial_metric*/,
      const Scalar<DataVector>& /*det_spatial_metric*/,
      const Scalar<DataVector>& field) noexcept {
    auxiliary_fluxes(flux_for_gradient, field);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

/*!
 * \brief Compute the sources \f$S_A\f$ for the Poisson equation.
 *
 * \see Poisson::FirstOrderSystem
 */
struct Sources {
  using argument_tags = tmpl::list<>;
  template <size_t Dim>
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_field,
      const gsl::not_null<
          tnsr::i<DataVector, Dim>*> /*source_for_field_gradient*/,
      const Scalar<DataVector>& /*field*/,
      const tnsr::I<DataVector, Dim>& /*field_flux*/) noexcept {
    get(*source_for_field) = 0.;
  }
};

}  // namespace Poisson
