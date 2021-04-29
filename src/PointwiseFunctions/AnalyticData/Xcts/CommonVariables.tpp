// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace Xcts::AnalyticData {

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::II<DataType, Dim>*> inv_conformal_metric,
    const gsl::not_null<Cache*> cache,
    Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/)
    const noexcept {
  const auto& conformal_metric =
      cache->get_var(Tags::ConformalMetric<DataType, Dim, Frame::Inertial>{});
  Scalar<DataType> unused_det{get_size(*conformal_metric.begin())};
  determinant_and_inverse(make_not_null(&unused_det), inv_conformal_metric,
                          conformal_metric);
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, Dim>*>
        conformal_christoffel_first_kind,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalChristoffelFirstKind<
        DataType, Dim, Frame::Inertial> /*meta*/) const noexcept {
  const auto& deriv_conformal_metric = cache->get_var(
      ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>{});
  gr::christoffel_first_kind(conformal_christoffel_first_kind,
                             deriv_conformal_metric);
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::Ijj<DataType, Dim>*>
        conformal_christoffel_second_kind,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalChristoffelSecondKind<
        DataType, Dim, Frame::Inertial> /*meta*/) const noexcept {
  const auto& conformal_christoffel_first_kind = cache->get_var(
      Tags::ConformalChristoffelFirstKind<DataType, Dim, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      Tags::InverseConformalMetric<DataType, 3, Frame::Inertial>{});
  raise_or_lower_first_index(conformal_christoffel_second_kind,
                             conformal_christoffel_first_kind,
                             inv_conformal_metric);
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::i<DataType, Dim>*>
        conformal_christoffel_contracted,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalChristoffelContracted<
        DataType, Dim, Frame::Inertial> /*meta*/) const noexcept {
  const auto& conformal_christoffel_second_kind = cache->get_var(
      Tags::ConformalChristoffelSecondKind<DataType, Dim, Frame::Inertial>{});
  for (size_t i = 0; i < Dim; ++i) {
    conformal_christoffel_contracted->get(i) =
        conformal_christoffel_second_kind.get(0, i, 0);
    for (size_t j = 1; j < Dim; ++j) {
      conformal_christoffel_contracted->get(i) +=
          conformal_christoffel_second_kind.get(j, i, j);
    }
  }
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        fixed_source_for_hamiltonian_constraint,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<Tags::ConformalFactor<DataType>> /*meta*/)
    const noexcept {
  std::fill(fixed_source_for_hamiltonian_constraint->begin(),
            fixed_source_for_hamiltonian_constraint->end(), 0.);
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<Scalar<DataType>*> fixed_source_for_lapse_equation,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<Tags::LapseTimesConformalFactor<DataType>> /*meta*/)
    const noexcept {
  std::fill(fixed_source_for_lapse_equation->begin(),
            fixed_source_for_lapse_equation->end(), 0.);
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> fixed_source_momentum_constraint,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::FixedSource<
        Tags::ShiftExcess<DataType, 3, Frame::Inertial>> /*meta*/)
    const noexcept {
  std::fill(fixed_source_momentum_constraint->begin(),
            fixed_source_momentum_constraint->end(), 0.);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif  // defined(__GNUC__) && !defined(__clang__)
template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::iJkk<DataType, Dim>*>
        deriv_conformal_christoffel_second_kind,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<
        Tags::ConformalChristoffelSecondKind<DataType, Dim, Frame::Inertial>,
        tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const noexcept {
  ASSERT(mesh.has_value() and inv_jacobian.has_value(),
         "Need a mesh and a Jacobian for numeric differentiation.");
  if constexpr (std::is_same_v<DataType, DataVector>) {
    // Copy into a Variables to take partial derivatives because at this time
    // the `partial_derivatives` function only works with Variables. This won't
    // be used for anything performance-critical, but adding a
    // `partial_derivatives` overload that takes a Tensor is an obvious
    // optimization here.
    using tag =
        Tags::ConformalChristoffelSecondKind<DataType, Dim, Frame::Inertial>;
    Variables<tmpl::list<tag>> vars{mesh->get().number_of_grid_points()};
    get<tag>(vars) = cache->get_var(tag{});
    const auto derivs = partial_derivatives<tmpl::list<tag>>(
        vars, mesh->get(), inv_jacobian->get());
    *deriv_conformal_christoffel_second_kind =
        get<::Tags::deriv<tag, tmpl::size_t<Dim>, Frame::Inertial>>(derivs);
  } else {
    (void)deriv_conformal_christoffel_second_kind;
    (void)cache;
    ERROR(
        "Numeric differentiation only works with DataVectors because it needs "
        "a grid.");
  }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__)

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_ricci_tensor,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalRicciTensor<DataType, Dim, Frame::Inertial> /*meta*/)
    const noexcept {
  const auto& conformal_christoffel_second_kind = cache->get_var(
      Tags::ConformalChristoffelSecondKind<DataType, Dim, Frame::Inertial>{});
  const auto& deriv_conformal_christoffel_second_kind = cache->get_var(
      ::Tags::deriv<
          Tags::ConformalChristoffelSecondKind<DataType, Dim, Frame::Inertial>,
          tmpl::size_t<Dim>, Frame::Inertial>{});
  gr::ricci_tensor(conformal_ricci_tensor, conformal_christoffel_second_kind,
                   deriv_conformal_christoffel_second_kind);
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_ricci_scalar,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalRicciScalar<DataType> /*meta*/) const noexcept {
  const auto& conformal_ricci_tensor = cache->get_var(
      Tags::ConformalRicciTensor<DataType, Dim, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>{});
  trace(conformal_ricci_scalar, conformal_ricci_tensor, inv_conformal_metric);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif  // defined(__GNUC__) && !defined(__clang__)
template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::i<DataType, Dim>*>
        deriv_extrinsic_curvature_trace,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const noexcept {
  ASSERT(mesh.has_value() and inv_jacobian.has_value(),
         "Need a mesh and a Jacobian for numeric differentiation.");
  if constexpr (std::is_same_v<DataType, DataVector>) {
    // Copy into a Variables to take partial derivatives because at this time
    // the `partial_derivatives` function only works with Variables. This won't
    // be used for anything performance-critical, but adding a
    // `partial_derivatives` overload that takes a Tensor is an obvious
    // optimization here.
    using tag = gr::Tags::TraceExtrinsicCurvature<DataType>;
    Variables<tmpl::list<tag>> vars{mesh->get().number_of_grid_points()};
    get<tag>(vars) = cache->get_var(tag{});
    const auto derivs = partial_derivatives<tmpl::list<tag>>(
        vars, mesh->get(), inv_jacobian->get());
    *deriv_extrinsic_curvature_trace =
        get<::Tags::deriv<tag, tmpl::size_t<Dim>, Frame::Inertial>>(derivs);
  } else {
    (void)deriv_extrinsic_curvature_trace;
    (void)cache;
    ERROR(
        "Numeric differentiation only works with DataVectors because it needs "
        "a grid.");
  }
}

template <typename DataType, typename Cache>
void CommonVariables<DataType, Cache>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*>
        div_longitudinal_shift_background,
    const gsl::not_null<Cache*> cache,
    ::Tags::div<Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, Dim, Frame::Inertial>> /*meta*/) const noexcept {
  ASSERT(mesh.has_value() and inv_jacobian.has_value(),
         "Need a mesh and a Jacobian for numeric differentiation.");
  if constexpr (std::is_same_v<DataType, DataVector>) {
    // Copy into a Variables to take the divergence because at this time the
    // `divergence` function only works with Variables. This won't be used for
    // anything performance-critical, but adding a `divergence` overload that
    // takes a Tensor is an obvious optimization here.
    using tag = Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, Dim, Frame::Inertial>;
    Variables<tmpl::list<tag>> vars{mesh->get().number_of_grid_points()};
    get<tag>(vars) = cache->get_var(tag{});
    const auto derivs = divergence(vars, mesh->get(), inv_jacobian->get());
    *div_longitudinal_shift_background = get<::Tags::div<tag>>(derivs);
  } else {
    (void)div_longitudinal_shift_background;
    (void)cache;
    ERROR(
        "Numeric differentiation only works with DataVectors because it needs "
        "a grid.");
  }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__)

}  // namespace Xcts::AnalyticData

/// \endcond
