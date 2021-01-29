// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Poisson/Geometry.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace PUP {
class er;
}  // namespace PUP
namespace Poisson {
template <size_t Dim, Geometry BackgroundGeometry>
struct Fluxes;
template <size_t Dim, Geometry BackgroundGeometry>
struct Sources;
}  // namespace Poisson
/// \endcond

namespace Poisson {

/*!
 * \brief Compute the fluxes \f$F^i=\partial_i u(x)\f$ for the Poisson
 * equation on a flat spatial metric in Cartesian coordinates.
 */
template <size_t Dim>
void flat_cartesian_fluxes(
    gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
    const tnsr::i<DataVector, Dim>& field_gradient) noexcept;

/*!
 * \brief Compute the fluxes \f$F^i=\gamma^{ij}\partial_j u(x)\f$
 * for the curved-space Poisson equation on a spatial metric \f$\gamma_{ij}\f$.
 */
template <size_t Dim>
void curved_fluxes(gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
                   const tnsr::II<DataVector, Dim>& inv_spatial_metric,
                   const tnsr::i<DataVector, Dim>& field_gradient) noexcept;

/*!
 * \brief Add the sources \f$S=-\Gamma^i_{ij}v^j\f$
 * for the curved-space Poisson equation on a spatial metric \f$\gamma_{ij}\f$.
 *
 * These sources arise from the non-principal part of the Laplacian on a
 * non-Euclidean background.
 */
template <size_t Dim>
void add_curved_sources(
    gsl::not_null<Scalar<DataVector>*> source_for_field,
    const tnsr::i<DataVector, Dim>& christoffel_contracted,
    const tnsr::I<DataVector, Dim>& flux_for_field) noexcept;

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
struct Fluxes<Dim, Geometry::FlatCartesian> {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
      const tnsr::i<DataVector, Dim>& field_gradient) noexcept {
    flat_cartesian_fluxes(flux_for_field, field_gradient);
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
struct Fluxes<Dim, Geometry::Curved> {
  using argument_tags = tmpl::list<
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataVector>>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
      const tnsr::II<DataVector, Dim>& inv_spatial_metric,
      const tnsr::i<DataVector, Dim>& field_gradient) noexcept {
    curved_fluxes(flux_for_field, inv_spatial_metric, field_gradient);
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_gradient,
      const tnsr::II<DataVector, Dim>& /*inv_spatial_metric*/,
      const Scalar<DataVector>& field) noexcept {
    auxiliary_fluxes(flux_for_gradient, field);
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

/*!
 * \brief Compute the sources \f$S_A\f$ for the Poisson equation on a flat
 * metric in Cartesian coordinates.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim>
struct Sources<Dim, Geometry::FlatCartesian> {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_field,
      const gsl::not_null<
          tnsr::i<DataVector, Dim>*> /*source_for_field_gradient*/,
      const Scalar<DataVector>& /*field*/,
      const tnsr::I<DataVector, Dim>& /*field_flux*/) noexcept {
    get(*source_for_field) = 0.;
  }
};

/*!
 * \brief Compute the sources \f$S_A\f$ for the curved-space Poisson equation
 * on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim>
struct Sources<Dim, Geometry::Curved> {
  using argument_tags =
      tmpl::list<gr::Tags::SpatialChristoffelSecondKindContracted<
          Dim, Frame::Inertial, DataVector>>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> source_for_field,
      const gsl::not_null<
          tnsr::i<DataVector, Dim>*> /*source_for_field_gradient*/,
      const tnsr::i<DataVector, Dim>& christoffel_contracted,
      const Scalar<DataVector>& /*field*/,
      const tnsr::I<DataVector, Dim>& field_flux) noexcept {
    get(*source_for_field) = 0.;
    add_curved_sources(source_for_field, christoffel_contracted, field_flux);
  }
};

}  // namespace Poisson
