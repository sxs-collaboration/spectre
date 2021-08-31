// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Xcts/SpacetimeQuantities.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts {

namespace detail {
template <typename T>
struct TempTag {
  using type = T;
};
template <typename U, typename T, size_t Dim, typename DerivativeFrame>
auto deriv_tensor(
    const gsl::not_null<U*> deriv, const T& tensor, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, DerivativeFrame>&
        inv_jacobian) noexcept {
  Variables<tmpl::list<TempTag<T>>> vars{tensor.begin()->size()};
  get<TempTag<T>>(vars) = tensor;
  const auto deriv_vars =
      partial_derivatives<tmpl::list<TempTag<T>>>(vars, mesh, inv_jacobian);
  *deriv = get<::Tags::deriv<TempTag<T>, tmpl::size_t<Dim>, DerivativeFrame>>(
      deriv_vars);
}
}  // namespace detail

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> spatial_metric,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector> /*meta*/)
    const noexcept {
  *spatial_metric = conformal_metric;
  for (auto& spatial_metric_component : *spatial_metric) {
    spatial_metric_component *= pow<4>(get(conformal_factor));
  }
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::II<DataVector, 3>*> inv_spatial_metric,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector> /*meta*/)
    const noexcept {
  *inv_spatial_metric = inv_conformal_metric;
  for (auto& inv_spatial_metric_component : *inv_spatial_metric) {
    inv_spatial_metric_component /= pow<4>(get(conformal_factor));
  }
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::ijj<DataVector, 3>*> deriv_spatial_metric,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const noexcept {
  const auto& spatial_metric =
      cache->get_var(gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>{});
  detail::deriv_tensor(deriv_spatial_metric, spatial_metric, mesh,
                       inv_jacobian);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::Ijj<DataVector, 3>*> christoffel_second_kind,
    const gsl::not_null<Cache*> cache,
    gr::Tags::SpatialChristoffelSecondKind<
        3, Frame::Inertial, DataVector> /*meta*/) const noexcept {
  const auto& deriv_spatial_metric = cache->get_var(
      ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>{});
  const auto& inv_spatial_metric = cache->get_var(
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>{});
  gr::christoffel_second_kind(christoffel_second_kind, deriv_spatial_metric,
                              inv_spatial_metric);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::iJkk<DataVector, 3>*>
        deriv_christoffel_second_kind,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<
        gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>,
        tmpl::size_t<3>, Frame::Inertial> /*meta*/) const noexcept {
  const auto& christoffel_second_kind = cache->get_var(
      gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>{});
  detail::deriv_tensor(deriv_christoffel_second_kind, christoffel_second_kind,
                       mesh, inv_jacobian);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> ricci_tensor,
    const gsl::not_null<Cache*> cache,
    gr::Tags::SpatialRicci<3, Frame::Inertial, DataVector> /*meta*/)
    const noexcept {
  const auto& christoffel_second_kind = cache->get_var(
      gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>{});
  const auto& deriv_christoffel_second_kind = cache->get_var(
      ::Tags::deriv<gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial,
                                                           DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>{});
  gr::ricci_tensor(ricci_tensor, christoffel_second_kind,
                   deriv_christoffel_second_kind);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Lapse<DataVector> /*meta*/) const noexcept {
  get(*lapse) = get(lapse_times_conformal_factor) / get(conformal_factor);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Shift<3, Frame::Inertial, DataVector> /*meta*/) const noexcept {
  for (size_t i = 0; i < 3; ++i) {
    shift->get(i) = shift_excess.get(i) + shift_background.get(i);
  }
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::iJ<DataVector, 3>*> deriv_shift,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const noexcept {
  const auto& shift =
      cache->get_var(gr::Tags::Shift<3, Frame::Inertial, DataVector>{});
  detail::deriv_tensor(deriv_shift, shift, mesh, inv_jacobian);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> dt_spatial_metric,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial,
                                       DataVector>> /*meta*/) const noexcept {
  for (auto& dt_spatial_metric_component : *dt_spatial_metric) {
    dt_spatial_metric_component = 0.;
  }
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> extrinsic_curvature,
    const gsl::not_null<Cache*> cache,
    gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector> /*meta*/)
    const noexcept {
  const auto& lapse = cache->get_var(gr::Tags::Lapse<DataVector>{});
  const auto& shift =
      cache->get_var(gr::Tags::Shift<3, Frame::Inertial, DataVector>{});
  const auto& deriv_shift = cache->get_var(
      ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>{});
  const auto& spatial_metric =
      cache->get_var(gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>{});
  const auto& dt_spatial_metric = cache->get_var(
      ::Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>{});
  const auto& deriv_spatial_metric = cache->get_var(
      ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>{});
  gr::extrinsic_curvature(extrinsic_curvature, lapse, shift, deriv_shift,
                          spatial_metric, dt_spatial_metric,
                          deriv_spatial_metric);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<Scalar<DataVector>*> extrinsic_curvature_square,
    const gsl::not_null<Cache*> cache,
    detail::ExtrinsicCurvatureSquare<DataVector> /*meta*/) const noexcept {
  const auto& extrinsic_curvature = cache->get_var(
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>{});
  const auto& inv_spatial_metric = cache->get_var(
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>{});
  get(*extrinsic_curvature_square) = 0.;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          get(*extrinsic_curvature_square) +=
              inv_spatial_metric.get(i, k) * inv_spatial_metric.get(j, l) *
              extrinsic_curvature.get(i, j) * extrinsic_curvature.get(k, l);
        }
      }
    }
  }
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::ijj<DataVector, 3>*> deriv_extrinsic_curvature,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const noexcept {
  const auto& extrinsic_curvature = cache->get_var(
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>{});
  const auto& christoffel = cache->get_var(
      gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>{});
  detail::deriv_tensor(deriv_extrinsic_curvature, extrinsic_curvature, mesh,
                       inv_jacobian);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k <= j; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          deriv_extrinsic_curvature->get(i, j, k) -=
              christoffel.get(l, i, j) * extrinsic_curvature.get(l, k);
          deriv_extrinsic_curvature->get(i, j, k) -=
              christoffel.get(l, i, k) * extrinsic_curvature.get(j, l);
        }
      }
    }
  }
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
    const gsl::not_null<Cache*> cache,
    gr::Tags::HamiltonianConstraint<DataVector> /*meta*/) const noexcept {
  const auto& ricci_tensor =
      cache->get_var(gr::Tags::SpatialRicci<3, Frame::Inertial, DataVector>{});
  const auto& extrinsic_curvature = cache->get_var(
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>{});
  const auto& inv_spatial_metric = cache->get_var(
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>{});
  const auto& extrinsic_curvature_square =
      cache->get_var(detail::ExtrinsicCurvatureSquare<DataVector>{});
  get(*hamiltonian_constraint) =
      get(trace(ricci_tensor, inv_spatial_metric)) +
             square(get(trace(extrinsic_curvature, inv_spatial_metric))) -
             get(extrinsic_curvature_square);
}

void SpacetimeQuantitiesComputer::operator()(
    const gsl::not_null<tnsr::i<DataVector, 3>*> momentum_constraint,
    const gsl::not_null<Cache*> cache,
    gr::Tags::MomentumConstraint<3, Frame::Inertial, DataVector> /*meta*/)
    const noexcept {
  const auto& deriv_extrinsic_curvature = cache->get_var(
      ::Tags::deriv<
          gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>,
          tmpl::size_t<3>, Frame::Inertial>{});
  const auto& inv_spatial_metric = cache->get_var(
      gr::Tags::InverseSpatialMetric<3, Frame::Inertial, DataVector>{});
  for (size_t i = 0; i < 3; ++i) {
    momentum_constraint->get(i) = 0.;
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        momentum_constraint->get(i) += inv_spatial_metric.get(j, k) *
                                       (deriv_extrinsic_curvature.get(j, k, i) -
                                        deriv_extrinsic_curvature.get(i, j, k));
      }
    }
  }
}

}  // namespace Xcts
