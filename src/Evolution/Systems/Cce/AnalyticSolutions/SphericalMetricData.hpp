// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/BoundaryDataTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Cce {
namespace Solutions {

/*!
 * \brief Abstract base class for analytic worldtube data most easily derived in
 * spherical coordinate form.
 *
 * \details This class provides the functions required by the `WorldtubeData`
 * interface that convert from a spherical coordinate spacetime metric to
 * Cartesian coordinates. Derived classes of `SphericalMetricData` need not
 * implement the `variables_impl`s for the Cartesian quantities. Instead, the
 * derived classes must override the protected functions:
 * - `SphericalMetricData::spherical_metric()`
 * - `SphericalMetricData::dr_spherical_metric()`
 * - `SphericalMetricData::dt_spherical_metric()`
 * Derived classes are still responsible for overriding
 * `WorldtubeData::get_clone()`, `WorldtubeData::variables_impl()` for tag
 * `Cce::Tags::News`, and `WorldtubeData::prepare_solution()`.
 */
struct SphericalMetricData : public WorldtubeData {
  /*!
   * Computes and returns by pointer the Jacobian
   * \f$\partial x_{\mathrm{spherical}}^j / \partial
   * x_{\mathrm{Cartesian}}^i\f$
   */
  void inverse_jacobian(gsl::not_null<CartesianiSphericalJ*> inverse_jacobian,
                        size_t l_max) const noexcept;

  /*!
   * Computes and returns by pointer the first radial derivative of the
   * Jacobian: \f$\partial_r (\partial x_{\mathrm{spherical}}^j / \partial
   * x_{\mathrm{Cartesian}}^i)\f$
   */
  void dr_inverse_jacobian(
      gsl::not_null<CartesianiSphericalJ*> dr_inverse_jacobian,
      size_t l_max) const noexcept;

 protected:
  using WorldtubeData::variables_impl;

  /*!
   * \brief Computes the Cartesian spacetime metric from the spherical solution
   * provided by the derived classes.
   *
   * \details The derived classes provide spherical metric data via the virtual
   * function `SphericalMetricData::spherical_metric()` at a resolution
   * determined by member variable `l_max_`. This function performs the
   * coordinate transformation using the Jacobian computed from
   * `SphericalMetricData::inverse_jacobian()`.
   */
  void variables_impl(gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric,
                      size_t l_max, double time,
                      tmpl::type_<gr::Tags::SpacetimeMetric<
                          3, ::Frame::Inertial, DataVector>> /*meta*/) const
      noexcept override;

  /*!
   * \brief Computes the time derivative of the Cartesian spacetime metric from
   * the spherical solution provided by the derived classes.
   *
   * \details The derived classes provide the time derivative of the spherical
   * metric data via the virtual function
   * `SphericalMetricData::dt_spherical_metric()` at a resolution determined by
   * member variable `l_max_`. This function performs the coordinate
   * transformation using the Jacobian computed from
   * `SphericalMetricData::inverse_jacobian()`.
   */
  void variables_impl(
      gsl::not_null<tnsr::aa<DataVector, 3>*> dt_spacetime_metric, size_t l_max,
      double time,
      tmpl::type_<::Tags::dt<gr::Tags::SpacetimeMetric<
          3, ::Frame::Inertial, DataVector>>> /*meta*/) const noexcept override;

  /*!
   * \brief Computes the spatial derivatives of the Cartesian spacetime metric
   * from the spherical solution provided by the derived classes.
   *
   * \details The derived classes provide the radial derivative of the spherical
   * metric data via the virtual function
   * `SphericalMetricData::dr_spherical_metric()` at a resolution determined by
   * member variable `l_max_`. This function performs the additional angular
   * derivatives necessary to assemble the full spatial derivative and performs
   * the coordinate transformation to Cartesian coordinates via the Jacobians
   * computed in `SphericalMetricData::inverse_jacobian()` and
   * `SphericalMetricData::inverse_jacobian()`.
   */
  void variables_impl(
      gsl::not_null<tnsr::iaa<DataVector, 3>*> d_spacetime_metric, size_t l_max,
      double time,
      tmpl::type_<
          GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>> /*meta*/) const
      noexcept override;

  /// Must be overriden in the derived class; should compute the spacetime
  /// metric of the analytic solution in spherical coordinates.
  virtual void spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          spherical_metric,
      double time) const noexcept = 0;

  /// Must be overriden in the derived class; should compute the first radial
  /// derivative of the spacetime metric of the analytic solution in spherical
  /// coordinates.
  virtual void dr_spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          dr_spherical_metric,
      double time) const noexcept = 0;

  /// Must be overriden in the derived class; should compute the first time
  /// derivative of the spacetime metric of the analytic solution in spherical
  /// coordinates.
  virtual void dt_spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          dt_spherical_metric,
      double time) const noexcept = 0;
};

}  // namespace Solutions
}  // namespace Cce
