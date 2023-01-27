// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::AnalyticData {

/// Tags for variables that analytic-data classes can share
template <typename DataType>
using common_tags = tmpl::list<
    // Background quantities
    Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
    gr::Tags::TraceExtrinsicCurvature<DataType>,
    ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>>,
    Tags::ShiftBackground<DataType, 3, Frame::Inertial>,
    Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataType, 3,
                                                            Frame::Inertial>,
    // Analytic derivatives
    ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial>,
    // Background quantities derived from the above
    Tags::InverseConformalMetric<DataType, 3, Frame::Inertial>,
    Tags::ConformalChristoffelFirstKind<DataType, 3, Frame::Inertial>,
    Tags::ConformalChristoffelSecondKind<DataType, 3, Frame::Inertial>,
    Tags::ConformalChristoffelContracted<DataType, 3, Frame::Inertial>,
    // Fixed sources
    ::Tags::FixedSource<Tags::ConformalFactor<DataType>>,
    ::Tags::FixedSource<Tags::LapseTimesConformalFactor<DataType>>,
    ::Tags::FixedSource<Tags::ShiftExcess<DataType, 3, Frame::Inertial>>,
    // These tags require numerical differentiation
    ::Tags::deriv<
        Tags::ConformalChristoffelSecondKind<DataType, 3, Frame::Inertial>,
        tmpl::size_t<3>, Frame::Inertial>,
    Tags::ConformalRicciTensor<DataType, 3, Frame::Inertial>,
    Tags::ConformalRicciScalar<DataType>,
    ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>, tmpl::size_t<3>,
                  Frame::Inertial>,
    ::Tags::div<Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial>>>;

/*!
 * \brief Implementations for variables that analytic-data classes can share
 *
 * Analytic-data classes can derive their variable computers from this class to
 * inherit implementations for the `common_tags`. Note that some variables
 * require a numeric differentiation. To compute those variables, a `mesh` and
 * an `inv_jacobian` must be passed to the constructor. The `mesh` and the
 * `inv_jacobian` can be set to `std::nullopt` if no variables with numeric
 * derivatives are requested.
 *
 * \tparam DataType `double` or `DataVector`. Must be `DataVector` if variables
 * with numeric derivatives are requested.
 * \tparam Cache The `CachedTempBuffer` used by the analytic-data class.
 */
template <typename DataType, typename Cache>
struct CommonVariables {
  static constexpr size_t Dim = 3;

  CommonVariables(const CommonVariables&) = default;
  CommonVariables& operator=(const CommonVariables&) = default;
  CommonVariables(CommonVariables&&) = default;
  CommonVariables& operator=(CommonVariables&&) = default;
  virtual ~CommonVariables() = default;

  CommonVariables(
      std::optional<std::reference_wrapper<const Mesh<Dim>>> local_mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataVector, Dim, Frame::ElementLogical, Frame::Inertial>>>
          local_inv_jacobian)
      : mesh(std::move(local_mesh)),
        inv_jacobian(std::move(local_inv_jacobian)) {}

  virtual void operator()(
      gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
      gsl::not_null<Cache*> cache,
      Tags::ConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/) const = 0;
  virtual void operator()(
      gsl::not_null<Scalar<DataType>*> extrinsic_curvature_trace,
      gsl::not_null<Cache*> cache,
      gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const = 0;
  virtual void operator()(
      gsl::not_null<Scalar<DataType>*> dt_extrinsic_curvature_trace,
      gsl::not_null<Cache*> cache,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/)
      const = 0;
  virtual void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
      gsl::not_null<Cache*> cache,
      Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/) const = 0;
  virtual void operator()(
      gsl::not_null<tnsr::II<DataType, Dim>*> longitudinal_shift_background,
      gsl::not_null<Cache*> cache,
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataType, Dim, Frame::Inertial> /*meta*/) const = 0;
  virtual void operator()(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const = 0;
  virtual void operator()(
      gsl::not_null<tnsr::II<DataType, Dim>*> inv_conformal_metric,
      gsl::not_null<Cache*> cache,
      Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/)
      const;
  virtual void operator()(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> conformal_christoffel_first_kind,
      gsl::not_null<Cache*> cache,
      Tags::ConformalChristoffelFirstKind<DataType, Dim,
                                          Frame::Inertial> /*meta*/) const;
  virtual void operator()(gsl::not_null<tnsr::Ijj<DataType, Dim>*>
                              conformal_christoffel_second_kind,
                          gsl::not_null<Cache*> cache,
                          Tags::ConformalChristoffelSecondKind<
                              DataType, Dim, Frame::Inertial> /*meta*/) const;
  virtual void operator()(
      gsl::not_null<tnsr::i<DataType, Dim>*> conformal_christoffel_contracted,
      gsl::not_null<Cache*> cache,
      Tags::ConformalChristoffelContracted<DataType, Dim,
                                           Frame::Inertial> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> fixed_source_for_hamiltonian_constraint,
      gsl::not_null<Cache*> cache,
      ::Tags::FixedSource<Tags::ConformalFactor<DataType>> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> fixed_source_for_lapse_equation,
      gsl::not_null<Cache*> cache,
      ::Tags::FixedSource<Tags::LapseTimesConformalFactor<DataType>> /*meta*/)
      const;
  void operator()(
      gsl::not_null<tnsr::I<DataType, 3>*> fixed_source_momentum_constraint,
      gsl::not_null<Cache*> cache,
      ::Tags::FixedSource<
          Tags::ShiftExcess<DataType, 3, Frame::Inertial>> /*meta*/) const;
  virtual void operator()(
      gsl::not_null<tnsr::iJkk<DataType, Dim>*>
          deriv_conformal_christoffel_second_kind,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<
          Tags::ConformalChristoffelSecondKind<DataType, Dim, Frame::Inertial>,
          tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const;
  virtual void operator()(
      gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_ricci_tensor,
      gsl::not_null<Cache*> cache,
      Tags::ConformalRicciTensor<DataType, Dim, Frame::Inertial> /*meta*/)
      const;
  virtual void operator()(
      gsl::not_null<Scalar<DataType>*> conformal_ricci_scalar,
      gsl::not_null<Cache*> cache,
      Tags::ConformalRicciScalar<DataType> /*meta*/) const;
  virtual void operator()(
      gsl::not_null<tnsr::i<DataType, Dim>*> deriv_extrinsic_curvature_trace,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const;
  virtual void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> div_longitudinal_shift_background,
      gsl::not_null<Cache*> cache,
      ::Tags::div<Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataType, Dim, Frame::Inertial>> /*meta*/) const;

  std::optional<std::reference_wrapper<const Mesh<Dim>>> mesh;
  std::optional<std::reference_wrapper<const InverseJacobian<
      DataVector, Dim, Frame::ElementLogical, Frame::Inertial>>>
      inv_jacobian;
};

}  // namespace Xcts::AnalyticData
