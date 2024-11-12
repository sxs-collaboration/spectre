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
template <size_t Dim, Geometry BackgroundGeometry,
          typename DataType = DataVector>
struct Fluxes;
template <size_t Dim, Geometry BackgroundGeometry,
          typename DataType = DataVector>
struct Sources;
}  // namespace Poisson
/// \endcond

namespace Poisson {

/*!
 * \brief Compute the fluxes \f$F^i=\partial_i u(x)\f$ for the Poisson
 * equation on a flat spatial metric in Cartesian coordinates.
 */
template <typename DataType, size_t Dim>
void flat_cartesian_fluxes(
    gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
    const tnsr::i<DataType, Dim>& field_gradient);

/*!
 * \brief Compute the fluxes \f$F^i=\gamma^{ij}\partial_j u(x)\f$
 * for the curved-space Poisson equation on a spatial metric \f$\gamma_{ij}\f$.
 */
template <typename DataType, size_t Dim>
void curved_fluxes(gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
                   const tnsr::II<DataVector, Dim>& inv_spatial_metric,
                   const tnsr::i<DataType, Dim>& field_gradient);

/*!
 * \brief Compute the fluxes $F^i=\gamma^{ij} n_j u$ where $n_j$ is the
 * `face_normal`.
 *
 * The `face_normal_vector` is $\gamma^{ij} n_j$.
 */
template <typename DataType, size_t Dim>
void fluxes_on_face(gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
                    const tnsr::I<DataVector, Dim>& face_normal_vector,
                    const Scalar<DataType>& field);

/*!
 * \brief Add the sources \f$S=-\Gamma^i_{ij}v^j\f$
 * for the curved-space Poisson equation on a spatial metric \f$\gamma_{ij}\f$.
 *
 * These sources arise from the non-principal part of the Laplacian on a
 * non-Euclidean background.
 */
template <typename DataType, size_t Dim>
void add_curved_sources(gsl::not_null<Scalar<DataType>*> source_for_field,
                        const tnsr::i<DataVector, Dim>& christoffel_contracted,
                        const tnsr::I<DataType, Dim>& flux_for_field);

/*!
 * \brief Compute the fluxes \f$F^i\f$ for the Poisson equation on a flat
 * metric in Cartesian coordinates.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim, typename DataType>
struct Fluxes<Dim, Geometry::FlatCartesian, DataType> {
  using argument_tags = tmpl::list<>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  static constexpr bool is_trivial = true;
  static constexpr bool is_discontinuous = false;
  static void apply(gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
                    const Scalar<DataType>& field,
                    const tnsr::i<DataType, Dim>& field_gradient);
  static void apply(gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
                    const tnsr::i<DataVector, Dim>& face_normal,
                    const tnsr::I<DataVector, Dim>& face_normal_vector,
                    const Scalar<DataType>& field);
};

/*!
 * \brief Compute the fluxes \f$F^i\f$ for the curved-space Poisson equation
 * on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim, typename DataType>
struct Fluxes<Dim, Geometry::Curved, DataType> {
  using argument_tags =
      tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, Dim>>;
  using volume_tags = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  static constexpr bool is_trivial = true;
  static constexpr bool is_discontinuous = false;
  static void apply(gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
                    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
                    const Scalar<DataType>& field,
                    const tnsr::i<DataType, Dim>& field_gradient);
  static void apply(gsl::not_null<tnsr::I<DataType, Dim>*> flux_for_field,
                    const tnsr::II<DataVector, Dim>& inv_spatial_metric,
                    const tnsr::i<DataVector, Dim>& face_normal,
                    const tnsr::I<DataVector, Dim>& face_normal_vector,
                    const Scalar<DataType>& field);
};

/*!
 * \brief Add the sources \f$S\f$ for the curved-space Poisson equation
 * on a spatial metric \f$\gamma_{ij}\f$.
 *
 * \see Poisson::FirstOrderSystem
 */
template <size_t Dim, typename DataType>
struct Sources<Dim, Geometry::Curved, DataType> {
  using argument_tags = tmpl::list<
      gr::Tags::SpatialChristoffelSecondKindContracted<DataVector, Dim>>;
  using const_global_cache_tags = tmpl::list<>;
  static void apply(gsl::not_null<Scalar<DataType>*> equation_for_field,
                    const tnsr::i<DataVector, Dim>& christoffel_contracted,
                    const Scalar<DataType>& field,
                    const tnsr::I<DataType, Dim>& field_flux);
};

}  // namespace Poisson
