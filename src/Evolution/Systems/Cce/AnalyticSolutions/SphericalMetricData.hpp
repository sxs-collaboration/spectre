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
 *
 * Derived classes are still responsible for overriding
 * `WorldtubeData::get_clone()`, `WorldtubeData::variables_impl()` for tag
 * `Cce::Tags::News`, and `WorldtubeData::prepare_solution()`.
 */
struct SphericalMetricData : public WorldtubeData {

  WRAPPED_PUPable_abstract(SphericalMetricData);  // NOLINT

  SphericalMetricData() = default;

  explicit SphericalMetricData(CkMigrateMessage* msg) : WorldtubeData(msg) {}

  explicit SphericalMetricData(const double extraction_radius)
      : WorldtubeData{extraction_radius} {}

  /*!
   * Computes the Jacobian
   * \f$\partial x_{\mathrm{Cartesian}}^j / \partial x_{\mathrm{spherical}}^i\f$
   *
   * \details The Jacobian (with \f$ \sin \theta \f$
   * scaled out of \f$\phi\f$ components) in question is
   *
   * \f{align*}{
   * \frac{\partial x_{\mathrm{Cartesian}}^j}{\partial x_{\mathrm{spherical}}^i}
   * = \left[
   * \begin{array}{ccc}
   * \frac{\partial x}{\partial r} & \frac{\partial y}{\partial r} &
   *  \frac{\partial z}{\partial r} \\
   * \frac{\partial x}{\partial \theta} & \frac{\partial y}{\partial \theta} &
   *  \frac{\partial z}{\partial \theta} \\
   * \frac{\partial x}{\sin \theta \partial \phi} &
   *  \frac{\partial y}{\sin \theta \partial \phi} &
   *  \frac{\partial y}{\sin \theta \partial \phi}
   * \end{array}
   * \right]
   * = \left[
   * \begin{array}{ccc}
   * \sin \theta \cos \phi & \sin \theta \sin \phi & \cos \theta \\
   * r \cos \theta \cos \phi & r \cos \theta \sin \phi & -r \sin \theta \\
   * -r \sin \phi & r \cos \phi & 0
   * \end{array}
   * \right]
   * \f}
   */
  void jacobian(gsl::not_null<SphericaliCartesianJ*> jacobian,
                size_t l_max) const;

  /*!
   * Computes the first radial derivative of the
   * Jacobian: \f$\partial_r (\partial x_{\mathrm{Cartesian}}^j /
   * \partial x_{\mathrm{Spherical}}^i)\f$
   *
   * \details The radial derivative of the Jacobian (with \f$ \sin \theta \f$
   * scaled out of \f$\phi\f$ components) in question is
   *
   * \f{align*}{
   * \frac{\partial}{\partial r}
   * \frac{\partial x_{\mathrm{Cartesian}}^j}{\partial x_{\mathrm{spherical}}^i}
   * = \left[
   * \begin{array}{ccc}
   * \frac{\partial^2 x}{(\partial r)^2} & \frac{\partial^2 y}{(\partial r)^2} &
   *  \frac{\partial^2 z}{(\partial r)^2} \\
   * \frac{\partial^2 x}{\partial r \partial \theta} &
   * \frac{\partial^2 y}{\partial r \partial \theta} &
   *  \frac{\partial^2 z}{\partial r \partial \theta} \\
   * \frac{\partial^2 x}{\sin \theta \partial r \partial \phi} &
   *  \frac{\partial^2 y}{\sin \theta \partial r \partial \phi} &
   *  \frac{\partial^2 y}{\sin \theta \partial r \partial \phi}
   * \end{array}
   * \right]
   * = \left[
   * \begin{array}{ccc}
   * 0 & 0 & 0 \\
   * \cos \theta \cos \phi & \cos \theta \sin \phi & - \sin \theta \\
   * - \sin \phi & \cos \phi & 0
   * \end{array}
   * \right]
   * \f}
   */
  static void dr_jacobian(gsl::not_null<SphericaliCartesianJ*> dr_jacobian,
                          size_t l_max);

  /*!
   * Computes the Jacobian
   * \f$\partial x_{\mathrm{spherical}}^j / \partial
   * x_{\mathrm{Cartesian}}^i\f$
   *
   * \details The Jacobian (with \f$ \sin \theta \f$
   * scaled out of \f$\phi\f$ components) in question is
   *
   * \f{align*}{
   * \frac{\partial x_{\mathrm{spherical}}^j}{\partial x_{\mathrm{Cartesian}}^i}
   * = \left[
   * \begin{array}{ccc}
   * \frac{\partial r}{\partial x} & \frac{\partial \theta}{\partial x} &
   *  \frac{\sin \theta \partial \phi}{\partial x} \\
   * \frac{\partial r}{\partial y} & \frac{\partial \theta}{\partial y} &
   *  \frac{\sin \theta \partial \phi}{\partial y} \\
   * \frac{\partial r}{\partial z} & \frac{\partial \theta}{\partial z} &
   *  \frac{\sin \theta \partial \phi}{\partial z}
   * \end{array}
   * \right]
   * = \left[
   * \begin{array}{ccc}
   * \cos \phi \sin \theta & \frac{\cos \phi \cos \theta}{r} &
   *  - \frac{\sin \phi}{r} \\
   * \sin \phi \sin \theta & \frac{\cos \theta \sin \phi}{r} &
   *  \frac{\cos \phi}{r} \\
   * \cos \theta & -\frac{\sin \theta}{r} & 0
   * \end{array}
   * \right]
   * \f}
   */
  void inverse_jacobian(gsl::not_null<CartesianiSphericalJ*> inverse_jacobian,
                        size_t l_max) const;

  /*!
   * Computes the first radial derivative of the
   * Jacobian: \f$\partial_r (\partial x_{\mathrm{spherical}}^j / \partial
   * x_{\mathrm{Cartesian}}^i)\f$
   *
   * \details The first radial derivative of the Jacobian (with
   * \f$ \sin \theta \f$  scaled out of \f$\phi\f$ components) in question is
   *
   * \f{align*}{
   * \frac{\partial}{\partial r}
   * \frac{\partial x_{\mathrm{spherical}}^j}{\partial x_{\mathrm{Cartesian}}^i}
   * = \left[
   * \begin{array}{ccc}
   * \frac{\partial}{\partial r} \frac{\partial r}{\partial x} &
   * \frac{\partial}{\partial r} \frac{\partial \theta}{\partial x} &
   * \frac{\partial}{\partial r} \frac{\sin \theta \partial \phi}{\partial x} \\
   * \frac{\partial}{\partial r} \frac{\partial r}{\partial y} &
   * \frac{\partial}{\partial r} \frac{\partial \theta}{\partial y} &
   * \frac{\partial}{\partial r} \frac{\sin \theta \partial \phi}{\partial y} \\
   * \frac{\partial}{\partial r} \frac{\partial r}{\partial z} &
   * \frac{\partial}{\partial r} \frac{\partial \theta}{\partial z} &
   * \frac{\partial}{\partial r} \frac{\sin \theta \partial \phi}{\partial z}
   * \end{array}
   * \right]
   * = \left[
   * \begin{array}{ccc}
   * 0 & - \frac{\cos \phi \cos \theta}{r^2} & \frac{\sin \phi}{r^2} \\
   * 0 & - \frac{\cos \theta \sin \phi}{r^2} & -\frac{\cos \phi}{r^2} \\
   * 0 & \frac{\sin \theta}{r^2} & 0
   * \end{array}
   * \right]
   * \f}
   */
  void dr_inverse_jacobian(
      gsl::not_null<CartesianiSphericalJ*> dr_inverse_jacobian,
      size_t l_max) const;

  void pup(PUP::er& p) override;

 protected:
  using WorldtubeData::variables_impl;

  /*!
   * \brief Computes the Cartesian spacetime metric from the spherical solution
   * provided by the derived classes.
   *
   * \details The derived classes provide spherical metric data via the virtual
   * function `SphericalMetricData::spherical_metric()` at a resolution
   * determined by the `l_max` argument. This function performs the
   * coordinate transformation using the Jacobian computed from
   * `SphericalMetricData::inverse_jacobian()`.
   */
  void variables_impl(
      gsl::not_null<tnsr::aa<DataVector, 3>*> spacetime_metric, size_t l_max,
      double time,
      tmpl::type_<
          gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>> /*meta*/)
      const override;

  /*!
   * \brief Computes the time derivative of the Cartesian spacetime metric from
   * the spherical solution provided by the derived classes.
   *
   * \details The derived classes provide the time derivative of the spherical
   * metric data via the virtual function
   * `SphericalMetricData::dt_spherical_metric()` at a resolution determined by
   * the `l_max` argument. This function performs the coordinate
   * transformation using the Jacobian computed from
   * `SphericalMetricData::inverse_jacobian()`.
   */
  void variables_impl(
      gsl::not_null<tnsr::aa<DataVector, 3>*> dt_spacetime_metric, size_t l_max,
      double time,
      tmpl::type_<::Tags::dt<gr::Tags::SpacetimeMetric<
          3, ::Frame::Inertial, DataVector>>> /*meta*/) const override;

  /*!
   * \brief Computes the spatial derivatives of the Cartesian spacetime metric
   * from the spherical solution provided by the derived classes.
   *
   * \details The derived classes provide the radial derivative of the spherical
   * metric data via the virtual function
   * `SphericalMetricData::dr_spherical_metric()` at a resolution determined by
   * the `l_max_` argument. This function performs the additional angular
   * derivatives necessary to assemble the full spatial derivative and performs
   * the coordinate transformation to Cartesian coordinates via the Jacobians
   * computed in `SphericalMetricData::inverse_jacobian()` and
   * `SphericalMetricData::inverse_jacobian()`.
   */
  void variables_impl(
      gsl::not_null<tnsr::iaa<DataVector, 3>*> d_spacetime_metric, size_t l_max,
      double time,
      tmpl::type_<
          GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>> /*meta*/)
      const override;

  /// Must be overriden in the derived class; should compute the spacetime
  /// metric of the analytic solution in spherical coordinates.
  virtual void spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          spherical_metric,
      size_t l_max, double time) const = 0;

  /// Must be overriden in the derived class; should compute the first radial
  /// derivative of the spacetime metric of the analytic solution in spherical
  /// coordinates.
  virtual void dr_spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          dr_spherical_metric,
      size_t l_max, double time) const = 0;

  /// Must be overriden in the derived class; should compute the first time
  /// derivative of the spacetime metric of the analytic solution in spherical
  /// coordinates.
  virtual void dt_spherical_metric(
      gsl::not_null<
          tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
          dt_spherical_metric,
      size_t l_max, double time) const = 0;
};

}  // namespace Solutions
}  // namespace Cce
