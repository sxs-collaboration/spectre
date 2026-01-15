// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/Systems/Ccz4/ATilde.hpp"
#include "Evolution/Systems/Ccz4/Christoffel.hpp"
#include "Evolution/Systems/Ccz4/DerivChristoffel.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/Ccz4/System.hpp"
#include "Evolution/Systems/Ccz4/Tags.hpp"
#include "Evolution/Systems/Ccz4/TimeDerivative.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugePlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::Ccz4::fd::detail {
using GhostData = evolution::dg::subcell::GhostData;
template <typename Fr = Frame::ElementLogical, typename F, typename... Args>
DirectionalIdMap<3, GhostData> compute_ghost_data(
    const Mesh<3>& subcell_mesh,
    const tnsr::I<DataVector, 3, Fr>& volume_coords,
    const DirectionMap<3, Neighbors<3>>& neighbors,
    const size_t ghost_zone_size, const F& compute_variables_of_neighbor_data,
    const std::array<double, 3>& coords_range, const Args&... args) {
  DirectionalIdMap<3, GhostData> ghost_data{};
  for (const auto& [direction, neighbors_in_direction] : neighbors) {
    REQUIRE(neighbors_in_direction.size() == 1);
    const ElementId<3>& neighbor_id = *neighbors_in_direction.begin();
    auto neighbor_coords = volume_coords;
    neighbor_coords.get(direction.dimension()) +=
        direction.sign() * gsl::at(coords_range, direction.dimension());
    const auto neighbor_vars_for_reconstruction =
        compute_variables_of_neighbor_data(neighbor_coords, args...);

    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars_for_reconstruction.data(),
                       neighbor_vars_for_reconstruction.size()),
        subcell_mesh.extents(), ghost_zone_size,
        std::unordered_set{direction.opposite()}, 0, {});
    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));
    ghost_data[DirectionalId<3>{direction, neighbor_id}] = GhostData{1};
    ghost_data.at(DirectionalId<3>{direction, neighbor_id})
        .neighbor_ghost_data_for_reconstruction() =
        sliced_data.at(direction.opposite());
  }
  return ghost_data;
}

template <typename Fr = Frame::ElementLogical, typename F, typename... Args>
DirectionalIdMap<3, GhostData> compute_ghost_data(
    const Mesh<3>& subcell_mesh,
    const tnsr::I<DataVector, 3, Fr>& volume_coords,
    const DirectionMap<3, Neighbors<3>>& neighbors,
    const size_t ghost_zone_size, const F& compute_variables_of_neighbor_data,
    Args&&... args) {
  const std::array<double, 3> coords_range{2., 2., 2.};
  return compute_ghost_data(subcell_mesh, volume_coords, neighbors,
                            ghost_zone_size, compute_variables_of_neighbor_data,
                            coords_range, std::forward<Args>(args)...);
}

inline Variables<::Ccz4::fd::Tags::spacetime_reconstruction_tags>
compute_prim_solution(
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& coords) {
  using ConformalMetric = ::Ccz4::Tags::ConformalMetric<DataVector, 3>;
  using ATilde = ::Ccz4::Tags::ATilde<DataVector, 3>;
  using ConformalFactor = ::Ccz4::Tags::ConformalFactor<DataVector>;
  using TraceExtrinsicCurvature = gr::Tags::TraceExtrinsicCurvature<DataVector>;
  using Theta = ::Ccz4::Tags::Theta<DataVector>;
  using GammaHat = ::Ccz4::Tags::GammaHat<DataVector, 3>;
  using Lapse = gr::Tags::Lapse<DataVector>;
  using Shift = gr::Tags::Shift<DataVector, 3>;
  using AuxiliaryShiftB = ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>;

  Variables<::Ccz4::fd::Tags::spacetime_reconstruction_tags> vars{
      get<0>(coords).size(), 0.0};
  for (size_t i = 0; i < 3; ++i) {
    get(get<ConformalFactor>(vars)) += coords.get(i);
    get(get<TraceExtrinsicCurvature>(vars)) += coords.get(i);
    get(get<Theta>(vars)) += coords.get(i);
    get(get<Lapse>(vars)) += coords.get(i);
    for (size_t j = 0; j < 3; ++j) {
      get<GammaHat>(vars).get(j) += coords.get(i);
      get<Shift>(vars).get(j) += coords.get(i);
      get<AuxiliaryShiftB>(vars).get(j) += coords.get(i);
    }
  }
  get(get<ConformalFactor>(vars)) += 2.0;
  get(get<TraceExtrinsicCurvature>(vars)) += 15.0;
  get(get<Theta>(vars)) += 30.0;
  get(get<Lapse>(vars)) += 50.0;
  for (size_t j = 0; j < 3; ++j) {
    get<GammaHat>(vars).get(j) += 1.0e-2 * static_cast<double>((j + 2) + 10);
    get<Shift>(vars).get(j) += 1.0e-2 * static_cast<double>((j + 2) + 60);
    get<AuxiliaryShiftB>(vars).get(j) +=
        1.0e-2 * static_cast<double>((j + 2) + 110);
  }

  auto& conformal_metric = get<ConformalMetric>(vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      conformal_metric.get(i, j) = (10 * i + 50 * j + 1) * coords.get(i);
    }
  }
  auto& atilde = get<ATilde>(vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      atilde.get(i, j) = (1000 * i + 5000 * j + 1) * coords.get(i);
    }
  }
  return vars;
}

inline Variables<::Ccz4::fd::Tags::spacetime_reconstruction_tags>
compute_prim_solution_for_second_deriv(
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& coords) {
  using ConformalMetric = ::Ccz4::Tags::ConformalMetric<DataVector, 3>;
  using ATilde = ::Ccz4::Tags::ATilde<DataVector, 3>;
  using ConformalFactor = ::Ccz4::Tags::ConformalFactor<DataVector>;
  using TraceExtrinsicCurvature = gr::Tags::TraceExtrinsicCurvature<DataVector>;
  using Theta = ::Ccz4::Tags::Theta<DataVector>;
  using GammaHat = ::Ccz4::Tags::GammaHat<DataVector, 3>;
  using Lapse = gr::Tags::Lapse<DataVector>;
  using Shift = gr::Tags::Shift<DataVector, 3>;
  using AuxiliaryShiftB = ::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>;

  Variables<::Ccz4::fd::Tags::spacetime_reconstruction_tags> vars{
      get<0>(coords).size(), 0.0};
  for (size_t i = 0; i < 3; ++i) {
    get(get<ConformalFactor>(vars)) += square(coords.get(i)) + coords.get(i);
    get(get<TraceExtrinsicCurvature>(vars)) +=
        square(coords.get(i)) + coords.get(i);
    get(get<Theta>(vars)) += square(coords.get(i)) + coords.get(i);
    get(get<Lapse>(vars)) += square(coords.get(i)) + coords.get(i);
    for (size_t j = 0; j < 3; ++j) {
      get<GammaHat>(vars).get(j) += square(coords.get(i)) + coords.get(i);
      get<Shift>(vars).get(j) += square(coords.get(i)) + coords.get(i);
      get<AuxiliaryShiftB>(vars).get(j) +=
          square(coords.get(i)) + coords.get(i);
    }
  }
  get(get<ConformalFactor>(vars)) += 2.0;
  get(get<TraceExtrinsicCurvature>(vars)) += 15.0;
  get(get<Theta>(vars)) += 30.0;
  get(get<Lapse>(vars)) += 50.0;
  for (size_t j = 0; j < 3; ++j) {
    get<GammaHat>(vars).get(j) += 1.0e-2 * static_cast<double>((j + 2) + 10);
    get<Shift>(vars).get(j) += 1.0e-2 * static_cast<double>((j + 2) + 60);
    get<AuxiliaryShiftB>(vars).get(j) +=
        1.0e-2 * static_cast<double>((j + 2) + 110);
  }

  auto& conformal_metric = get<ConformalMetric>(vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      conformal_metric.get(i, j) =
          (10 * i + 50 * j + 1) * square(coords.get(i)) +
          (10 * i + 50 * j + 1) * coords.get(i);
    }
  }
  auto& atilde = get<ATilde>(vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      atilde.get(i, j) = (1000 * i + 5000 * j + 1) * square(coords.get(i)) +
                         (1000 * i + 5000 * j + 1) * coords.get(i);
    }
  }
  return vars;
}

inline Element<3> set_element(const bool skip_last = false) {
  DirectionMap<3, Neighbors<3>> neighbors{};
  for (size_t i = 0; i < 6; ++i) {
    if (skip_last and i == 5) {
      ASSERT(gsl::at(Direction<3>::all_directions(), i) ==
                 Direction<3>::upper_zeta(),
             "Last direction is not upper_zeta");
      break;
    }
    neighbors[gsl::at(Direction<3>::all_directions(), i)] = Neighbors<3>{
        {ElementId<3>{i + 1, {}}}, OrientationMap<3>::create_aligned()};
  }
  return Element<3>{ElementId<3>{0, {}}, neighbors};
}

inline tnsr::I<DataVector, 3, Frame::ElementLogical> set_logical_coordinates(
    const Mesh<3>& subcell_mesh) {
  auto logical_coords = logical_coordinates(subcell_mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 1; i < 3; ++i) {
    logical_coords.get(i) += static_cast<double>(4 * i);
  }
  return logical_coords;
}

namespace Minkowski {
template <bool UsedForSommerfeldTest = false>
inline Variables<::Ccz4::fd::Tags::spacetime_reconstruction_tags>
compute_prim_solution_for_Minkowski(
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords) {
  using FrameType = Frame::Inertial;
  const size_t SpatialDim = 3;

  Variables<typename ::Ccz4::fd::System::variables_tag::tags_list> evolved_vars{
      get<0>(coords).size()};
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();
  // Setup solution
  const gr::Solutions::Minkowski<SpatialDim> solution{};
  // Evaluate solution
  const auto minkowski_vars = solution.variables(
      coords, t,
      typename gr::Solutions::Minkowski<SpatialDim>::tags<DataVector>{});
  const DataVector used_for_size = DataVector(
      get<0>(coords).size(), std::numeric_limits<double>::signaling_NaN());

  get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(evolved_vars) =
      get<gr::Tags::SpatialMetric<DataVector, SpatialDim, FrameType>>(
          minkowski_vars);  // conformal factor is 1

  const auto det_spatial_metric = determinant(
      get<gr::Tags::SpatialMetric<DataVector, SpatialDim, FrameType>>(
          minkowski_vars));
  const auto conformal_factor = pow(get(det_spatial_metric), -1. / 6.);
  Scalar<DataVector> conformal_factor_squared{};
  get(conformal_factor_squared) = square(conformal_factor);

  get(get<::Ccz4::Tags::ConformalFactor<DataVector>>(evolved_vars)) =
      conformal_factor;

  const auto extrinsic_curvature =
      make_with_value<tnsr::ii<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);
  get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars) =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);

  get<::Ccz4::Tags::ATilde<DataVector, 3>>(evolved_vars) = ::Ccz4::a_tilde(
      conformal_factor_squared,
      get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(evolved_vars),
      extrinsic_curvature,
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars));

  get<::Ccz4::Tags::Theta<DataVector>>(evolved_vars) =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);

  const auto field_d =
      make_with_value<tnsr::ijj<DataVector, SpatialDim, FrameType>>(
          used_for_size, 0.0);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, SpatialDim, FrameType>>(
          minkowski_vars);
  const auto& inverse_conformal_spatial_metric = inverse_spatial_metric;
  const auto conformal_christoffel_second_kind =
      ::Ccz4::conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d);
  const auto contracted_conformal_christoffel_second_kind =
      ::Ccz4::contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, conformal_christoffel_second_kind);
  // If Z4 constraint is 0, then \hat{\gamma} == \tilde{\gamma}
  get<::Ccz4::Tags::GammaHat<DataVector, 3>>(evolved_vars) =
      contracted_conformal_christoffel_second_kind;

  Scalar<DataVector> lapse{};
  if constexpr (UsedForSommerfeldTest) {
    get(lapse) = get<0>(coords);
  } else {
    lapse = get<gr::Tags::Lapse<DataVector>>(minkowski_vars);
  }

  get<gr::Tags::Lapse<DataVector>>(evolved_vars) = lapse;

  get<gr::Tags::Shift<DataVector, 3>>(evolved_vars) =
      get<gr::Tags::Shift<DataVector, SpatialDim, FrameType>>(minkowski_vars);

  get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(evolved_vars) =
      make_with_value<tnsr::I<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);
  return evolved_vars;
}
}  // namespace Minkowski

namespace KerrSchild {
// Compute the spatial derivative of the conformal spatial metric
//
// If \tilde{\gamma}_{ij} is the conformal metric and \phi is the
// conformal factor, \tilde{\gamma}_{ij} = \phi^2 \gamma_{ij}.
// Therefore, the derivative of the conformal metric is:
//   \partial_k \tilde{\gamma}_{ij} =
//       \phi^2 \partial_k \gamma_{ij} +
//       \partial_k \phi^2 \gamma_{ij}
//
// Since \phi = (det(\gamma_{ij}))^{-1/6}:
//   \partial_k \phi^2
//        = \partial_k ((det(\gamma_{ij}))^{-1/6})^2
//        = \partial_k (det(\gamma_{ij}))^{-1/3}
//        = -(det(\gamma_{ij}))^{-4/3}  \partial_k (det(\gamma_{ij}))/ 3
//        = -\phi^4 \partial_k (det(\gamma_{ij})) / 3
//
// Therefore:
//   \partial_k \tilde{\gamma}_{ij} =
//       \phi^2 \partial_k \gamma_{ij} -
//       \phi^4 \partial_k (det(\gamma_{ij})) \gamma_{ij} / 3
template <size_t SpatialDim, typename FrameType>
tnsr::ijj<DataVector, SpatialDim, FrameType> get_d_conformal_spatial_metric(
    const Scalar<DataVector>& conformal_factor_squared,
    const tnsr::ii<DataVector, SpatialDim, FrameType>& spatial_metric,
    const tnsr::ijj<DataVector, SpatialDim, FrameType>& d_spatial_metric,
    const tnsr::i<DataVector, SpatialDim, FrameType>& d_det_spatial_metric) {
  tnsr::ijj<DataVector, SpatialDim, FrameType> d_conformal_spatial_metric(
      get(conformal_factor_squared));
  for (size_t k = 0; k < SpatialDim; k++) {
    for (size_t i = 0; i < SpatialDim; i++) {
      for (size_t j = i; j < SpatialDim; j++) {
        d_conformal_spatial_metric.get(k, i, j) =
            get(conformal_factor_squared) * d_spatial_metric.get(k, i, j) -
            pow<4>(get(conformal_factor_squared)) *
                d_det_spatial_metric.get(k) * spatial_metric.get(i, j) / 3.;
      }
    }
  }
  return d_conformal_spatial_metric;
}

// Compute D_{kij} from eq 6
template <size_t SpatialDim, typename FrameType>
tnsr::ijj<DataVector, SpatialDim, FrameType> get_field_d(
    const tnsr::ijj<DataVector, SpatialDim, FrameType>&
        d_conformal_spatial_metric) {
  tnsr::ijj<DataVector, SpatialDim, FrameType> field_d(
      get<0, 0, 0>(d_conformal_spatial_metric));
  for (size_t i = 0; i < field_d.size(); i++) {
    field_d[i] = 0.5 * d_conformal_spatial_metric[i];
  }
  return field_d;
}

// Compute b^i for KerrSchild
//
// Solve eq 12c for b^i, where we assume \partial_t \beta^i = 0:
//   0 = s f b + s \beta^k \partial_k \beta^i
template <size_t SpatialDim, typename FrameType>
tnsr::I<DataVector, SpatialDim, FrameType> get_b_kerr(
    const bool evolve_shift,
    const tnsr::I<DataVector, SpatialDim, FrameType>& shift,
    const tnsr::iJ<DataVector, SpatialDim, FrameType>& d_shift,
    const double f) {
  tnsr::I<DataVector, SpatialDim, FrameType> b(get<0>(shift));
  if (not evolve_shift or
      not ::Ccz4::fd::System::shifting_shift) {
    // s == 0
    // pick b = 0
    for (auto& component : b) {
      component = 0.0;
    }
  } else {
    // s == 1
    // b = -(\beta^k \partial_k \beta^i) / f
    for (size_t i = 0; i < SpatialDim; i++) {
      b.get(i) = -shift.get(0) * d_shift.get(0, i);
      for (size_t k = 1; k < SpatialDim; k++) {
        b.get(i) -= shift.get(k) * d_shift.get(k, i);
      }
      b.get(i) /= f;
    }
  }
  return b;
}

// Compute the conformal spatial metric
//
// \tilde{\gamma}_{ij} = \phi^2 \gamma_{ij}
template <size_t SpatialDim, typename FrameType>
tnsr::ii<DataVector, SpatialDim, FrameType> get_conformal_spatial_metric(
    const Scalar<DataVector>& conformal_factor_squared,
    const tnsr::ii<DataVector, SpatialDim, FrameType>& spatial_metric) {
  tnsr::ii<DataVector, SpatialDim, FrameType> conformal_spatial_metric(
      get(conformal_factor_squared));
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = i; j < SpatialDim; j++) {
      conformal_spatial_metric.get(i, j) =
          get(conformal_factor_squared) * spatial_metric.get(i, j);
    }
  }
  return conformal_spatial_metric;
}

// Compute the trace of the extrinsic curvature K = K_{ij} * \gamma^ij
template <size_t SpatialDim, typename FrameType>
Scalar<DataVector> get_trace_extrinsic_curvature(
    const tnsr::ii<DataVector, SpatialDim, FrameType>& extrinsic_curvature,
    const tnsr::II<DataVector, SpatialDim, FrameType>& inverse_spatial_metric) {
  Scalar<DataVector> trace_extrinsic_curvature(get<0, 0>(extrinsic_curvature));
  get(trace_extrinsic_curvature) = 0.0;
  for (size_t i = 0; i < SpatialDim; i++) {
    for (size_t j = 0; j < SpatialDim; j++) {
      get(trace_extrinsic_curvature) +=
          extrinsic_curvature.get(i, j) * inverse_spatial_metric.get(i, j);
    }
  }
  return trace_extrinsic_curvature;
}

// Compute g(\alpha)
inline Scalar<DataVector> get_slicing_condition(
    const ::Ccz4::SlicingConditionType slicing_condition_type,
    const Scalar<DataVector>& lapse) {
  Scalar<DataVector> slicing_condition(get(lapse));
  if (slicing_condition_type == ::Ccz4::SlicingConditionType::Harmonic) {
    // harmonic slicing condition: g(\alpha) = 1.0
    get(slicing_condition) = 1.0;
  } else if (slicing_condition_type == ::Ccz4::SlicingConditionType::Log) {
    // 1 + log slicing condition: g(\alpha) = 2 / \alpha
    get(slicing_condition) = 2.0 / get(lapse);
  } else {
    ERROR("Unknown Ccz4::SlicingConditionType");
  }
  return slicing_condition;
}

// Compute K_0 for KerrSchild
//
// Solve eq 4g for K_0, where we assume \partial_t \alpha = 0:
//    0 = -\alpha^2 g(\alpha) (K - K_0 - 2 \Theta) +
//       \beta^k \partial_k \alpha
//   K_0 = -((\beta^k \partial_k \alpha) / (\alpha^2 * g(\alpha)) -
//           K + 2 \Theta);
template <size_t SpatialDim, typename FrameType>
Scalar<DataVector> get_k_0_kerr(
    const tnsr::I<DataVector, SpatialDim, FrameType>& shift,
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, SpatialDim, FrameType>& d_lapse,
    const Scalar<DataVector>& slicing_condition,
    const Scalar<DataVector>& theta,
    const Scalar<DataVector>& trace_extrinsic_curvature) {
  Scalar<DataVector> k_0(get(lapse));
  get(k_0) = get<0>(shift) * get<0>(d_lapse);
  for (size_t k = 1; k < SpatialDim; k++) {
    get(k_0) += shift.get(k) * d_lapse.get(k);
  }
  get(k_0) = -((get(k_0) / (square(get(lapse)) * get(slicing_condition))) -
               get(trace_extrinsic_curvature) + 2.0 * get(theta));
  return k_0;
}

// Compute expected value for LHS of eq 12i
//
// \partial_t b will not be 0 for KerrSchild if evolve_shift == true
template <size_t SpatialDim, typename FrameType>
tnsr::I<DataVector, SpatialDim, FrameType> get_dt_b_kerr_expected(
    const bool evolve_shift, const Scalar<DataVector>& eta,
    const tnsr::I<DataVector, SpatialDim, FrameType>& shift,
    const tnsr::iJ<DataVector, SpatialDim, FrameType>& d_gamma_hat,
    const tnsr::I<DataVector, SpatialDim, FrameType>& b,
    const tnsr::iJ<DataVector, SpatialDim, FrameType>& d_b) {
  tnsr::I<DataVector, SpatialDim, FrameType> dt_b_kerr_expected(get(eta));
  if (evolve_shift) {
    // s == 1
    for (size_t i = 0; i < SpatialDim; i++) {
      dt_b_kerr_expected.get(i) = -get(eta) * b.get(i);

      if (::Ccz4::fd::System::shifting_shift) {
        for (size_t k = 0; k < SpatialDim; k++) {
          dt_b_kerr_expected.get(i) += shift.get(k) * d_b.get(k, i) -
                                       shift.get(k) * d_gamma_hat.get(k, i);
        }
      }
    }
  } else {
    // s == 0
    for (auto& component : dt_b_kerr_expected) {
      component = 0.0;
    }
  }
  return dt_b_kerr_expected;
}

inline Variables<
    ::Ccz4::fd::Tags::spacetime_reconstruction_tags>
compute_prim_solution_for_KerrSchild(
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords, const double t,
    const double f, const bool evolve_shift,
    const gr::Solutions::KerrSchild& solution) {
  // Evaluate solution
  const auto kerrschild_vars = solution.variables(
      coords, t, typename gr::Solutions::KerrSchild::tags<DataVector>{});

  // get system evolved vars
  Variables<typename ::Ccz4::fd::System::variables_tag::tags_list> evolved_vars{
      get<0>(coords).size()};

  const DataVector used_for_size = DataVector(
      get<0>(coords).size(), std::numeric_limits<double>::signaling_NaN());

  const size_t SpatialDim = 3;
  using FrameType = Frame::Inertial;
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, SpatialDim, FrameType>>(
          kerrschild_vars);
  const auto det_spatial_metric = determinant(spatial_metric);
  const auto conformal_factor = pow(get(det_spatial_metric), -1. / 6.);
  Scalar<DataVector> conformal_factor_squared{};
  get(conformal_factor_squared) = square(conformal_factor);
  get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(evolved_vars) =
      get_conformal_spatial_metric(conformal_factor_squared, spatial_metric);

  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(kerrschild_vars);

  get<gr::Tags::Shift<DataVector, 3>>(evolved_vars) =
      get<gr::Tags::Shift<DataVector, SpatialDim, FrameType>>(kerrschild_vars);
  const auto& d_shift =
      get<Tags::deriv<gr::Tags::Shift<DataVector, SpatialDim, FrameType>,
                      tmpl::size_t<SpatialDim>, FrameType>>(kerrschild_vars);

  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<DataVector, SpatialDim>>>(
          kerrschild_vars);
  const auto& d_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<DataVector, SpatialDim>,
                      tmpl::size_t<SpatialDim>, FrameType>>(kerrschild_vars);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, SpatialDim, FrameType>>(
          kerrschild_vars);
  const auto extrinsic_curvature = gr::extrinsic_curvature(
      lapse, get<gr::Tags::Shift<DataVector, 3>>(evolved_vars), d_shift,
      spatial_metric, dt_spatial_metric, d_spatial_metric);
  get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars) =
      get_trace_extrinsic_curvature(extrinsic_curvature,
                                    inverse_spatial_metric);

  get<::Ccz4::Tags::ATilde<DataVector, 3>>(evolved_vars) = ::Ccz4::a_tilde(
      conformal_factor_squared, spatial_metric, extrinsic_curvature,
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars));

  get<::Ccz4::Tags::Theta<DataVector>>(evolved_vars) =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);

  const auto inverse_conformal_spatial_metric =
      determinant_and_inverse(
          get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(evolved_vars))
          .second;
  const auto d_det_spatial_metric =
      get<gr::Tags::DerivDetSpatialMetric<DataVector, SpatialDim, FrameType>>(
          solution.variables(
              coords, t,
              tmpl::list<gr::Tags::DerivDetSpatialMetric<DataVector, SpatialDim,
                                                         FrameType>>{}));
  const tnsr::ijj<DataVector, SpatialDim, FrameType>
      d_conformal_spatial_metric = get_d_conformal_spatial_metric(
          conformal_factor_squared, spatial_metric, d_spatial_metric,
          d_det_spatial_metric);
  const tnsr::ijj<DataVector, SpatialDim, FrameType> field_d =
      get_field_d(d_conformal_spatial_metric);
  const auto conformal_christoffel_second_kind =
      ::Ccz4::conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d);
  const auto contracted_conformal_christoffel_second_kind =
      ::Ccz4::contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, conformal_christoffel_second_kind);
  get<::Ccz4::Tags::GammaHat<DataVector, 3>>(evolved_vars) =
      contracted_conformal_christoffel_second_kind;  // Z4 constraint is 0

  get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(evolved_vars) =
      get_b_kerr(evolve_shift,
                 get<gr::Tags::Shift<DataVector, 3>>(evolved_vars), d_shift, f);

  get<gr::Tags::Lapse<DataVector>>(evolved_vars) = lapse;

  get(get<::Ccz4::Tags::ConformalFactor<DataVector>>(evolved_vars)) =
      conformal_factor;

  return evolved_vars;
}
}  // namespace KerrSchild

namespace GaugePlaneWave {
// calculate the spatial derivative of the determinant of the spatial metric
template <size_t SpatialDim, typename FrameType>
tnsr::i<DataVector, SpatialDim, FrameType>
get_d_det_spatial_metric_gauge_plane_wave(
    const Scalar<DataVector>& det_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, FrameType>& inverse_spatial_metric,
    const tnsr::ijj<DataVector, SpatialDim, FrameType>& d_spatial_metric) {
  tnsr::i<DataVector, SpatialDim, FrameType> d_det_spatial_metric;
  ::tenex::evaluate<ti::i>(make_not_null(&d_det_spatial_metric),
                           det_spatial_metric() *
                               inverse_spatial_metric(ti::J, ti::K) *
                               d_spatial_metric(ti::i, ti::j, ti::k));
  return d_det_spatial_metric;
}

// calculate the exact time derivative of the gauge plane wave conformal spatial
// metric
template <size_t SpatialDim, typename FrameType>
tnsr::ii<DataVector, SpatialDim, FrameType>
get_dt_conformal_spatial_metric_gauge_plane_wave(
    const Scalar<DataVector>& conformal_factor_squared,
    const tnsr::ii<DataVector, SpatialDim, FrameType>& spatial_metric,
    const tnsr::II<DataVector, SpatialDim, FrameType>& inverse_spatial_metric,
    const tnsr::ii<DataVector, SpatialDim, FrameType>& dt_spatial_metric) {
  tnsr::ii<DataVector, SpatialDim, FrameType> dt_conformal_spatial_metric;
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&dt_conformal_spatial_metric),
      conformal_factor_squared() * dt_spatial_metric(ti::i, ti::j) -
          1.0 / 3.0 * conformal_factor_squared() *
              spatial_metric(ti::i, ti::j) *
              inverse_spatial_metric(ti::L, ti::K) *
              dt_spatial_metric(ti::l, ti::k));
  return dt_conformal_spatial_metric;
}

// calculate the exact time derivative of the gauge plane wave conforma factor
template <size_t SpatialDim, typename FrameType>
Scalar<DataVector> get_dt_conformal_factor_gauge_plane_wave(
    const tnsr::II<DataVector, SpatialDim, FrameType>& inverse_spatial_metric,
    const tnsr::ii<DataVector, SpatialDim, FrameType>& dt_spatial_metric,
    const Scalar<DataVector>& conformal_factor) {
  Scalar<DataVector> dt_conformal_factor;
  ::tenex::evaluate(make_not_null(&dt_conformal_factor),
                    -1.0 / 6.0 * conformal_factor() *
                        inverse_spatial_metric(ti::J, ti::K) *
                        dt_spatial_metric(ti::j, ti::k));
  return dt_conformal_factor;
}

// calculate the exact spatial derivative of the gauge plane wave conformal
// factor
template <size_t SpatialDim, typename FrameType>
tnsr::i<DataVector, SpatialDim, FrameType>
get_d_conformal_factor_gauge_plane_wave(
    const tnsr::II<DataVector, SpatialDim, FrameType>& inverse_spatial_metric,
    const tnsr::ijj<DataVector, SpatialDim, FrameType>& d_spatial_metric,
    const Scalar<DataVector>& conformal_factor) {
  tnsr::i<DataVector, SpatialDim, FrameType> d_conformal_factor;
  ::tenex::evaluate<ti::i>(make_not_null(&d_conformal_factor),
                           -1.0 / 6.0 * conformal_factor() *
                               inverse_spatial_metric(ti::J, ti::K) *
                               d_spatial_metric(ti::i, ti::j, ti::k));
  return d_conformal_factor;
}

// calculate the exact time derivative of the gauge plane wave trace-free
// part of the extrinsic curvature
template <size_t SpatialDim, typename FrameType>
tnsr::ii<DataVector, SpatialDim, FrameType> get_dt_a_tilde_gauge_plane_wave(
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& conformal_factor_squared,
    const Scalar<DataVector>& dt_conformal_factor,
    const tnsr::ii<DataVector, SpatialDim, FrameType>& spatial_metric,
    const tnsr::ii<DataVector, SpatialDim, FrameType>& dt_spatial_metric,
    const tnsr::ii<DataVector, SpatialDim, FrameType>& extrinsic_curvature,
    const tnsr::ii<DataVector, SpatialDim, FrameType>& dt_extrinsic_curvature,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& dt_trace_extrinsic_curvature) {
  tnsr::ii<DataVector, SpatialDim, FrameType> dt_a_tilde;
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&dt_a_tilde),
      2.0 * conformal_factor() * dt_conformal_factor() *
              (extrinsic_curvature(ti::i, ti::j) -
               1.0 / 3.0 * trace_extrinsic_curvature() *
                   spatial_metric(ti::i, ti::j)) +
          conformal_factor_squared() *
              (dt_extrinsic_curvature(ti::i, ti::j) -
               1.0 / 3.0 * dt_trace_extrinsic_curvature() *
                   spatial_metric(ti::i, ti::j) -
               1.0 / 3.0 * trace_extrinsic_curvature() *
                   dt_spatial_metric(ti::i, ti::j)));
  return dt_a_tilde;
}

// calculate the time derivative of the inverse spatial metric
template <size_t SpatialDim, typename FrameType>
tnsr::II<DataVector, SpatialDim, FrameType> get_dt_inverse_spatial_metric(
    const tnsr::II<DataVector, SpatialDim, FrameType>& inverse_spatial_metric,
    const tnsr::ii<DataVector, SpatialDim, FrameType>& dt_spatial_metric) {
  tnsr::II<DataVector, SpatialDim, FrameType> dt_inverse_spatial_metric;
  ::tenex::evaluate<ti::L, ti::K>(make_not_null(&dt_inverse_spatial_metric),
                                  -1.0 * inverse_spatial_metric(ti::I, ti::L) *
                                      inverse_spatial_metric(ti::J, ti::K) *
                                      dt_spatial_metric(ti::i, ti::j));
  return dt_inverse_spatial_metric;
}

// calculate the time derivative of the inverse conformal spatial metric
template <size_t SpatialDim, typename FrameType>
tnsr::II<DataVector, SpatialDim, FrameType>
get_dt_inverse_conformal_spatial_metric(
    const tnsr::II<DataVector, SpatialDim, FrameType>&
        inverse_conformal_spatial_metric,
    const tnsr::ii<DataVector, SpatialDim, FrameType>&
        dt_conformal_spatial_metric) {
  tnsr::II<DataVector, SpatialDim, FrameType>
      dt_inverse_conformal_spatial_metric;
  ::tenex::evaluate<ti::L, ti::K>(
      make_not_null(&dt_inverse_conformal_spatial_metric),
      -1.0 * inverse_conformal_spatial_metric(ti::I, ti::L) *
          inverse_conformal_spatial_metric(ti::J, ti::K) *
          dt_conformal_spatial_metric(ti::i, ti::j));
  return dt_inverse_conformal_spatial_metric;
}

// calculate the exact time derivative of the gauge plane wave trace of
// extrinsic curvature
template <size_t SpatialDim, typename FrameType>
Scalar<DataVector> get_dt_trace_extrinsic_curvature_gauge_plane_wave(
    const tnsr::ii<DataVector, SpatialDim, FrameType>& extrinsic_curvature,
    const tnsr::ii<DataVector, SpatialDim, FrameType>& dt_extrinsic_curvature,
    const tnsr::II<DataVector, SpatialDim, FrameType>& inverse_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, FrameType>&
        dt_inverse_spatial_metric) {
  Scalar<DataVector> dt_trace_extrinsic_curvature;
  ::tenex::evaluate(make_not_null(&dt_trace_extrinsic_curvature),
                    inverse_spatial_metric(ti::I, ti::J) *
                            dt_extrinsic_curvature(ti::i, ti::j) +
                        dt_inverse_spatial_metric(ti::I, ti::J) *
                            extrinsic_curvature(ti::i, ti::j));
  return dt_trace_extrinsic_curvature;
}

// calculate 1 + H*\omega^2 in the gauge plane wave
inline Scalar<DataVector> get_one_plus_h_times_omega_squared(
    const Scalar<DataVector>& h, const double omega) {
  Scalar<DataVector> result;
  ::tenex::evaluate(make_not_null(&result), 1.0 + h() * square(omega));
  return result;
}

// calculate the exact time derivative of the gauge plane wave extrinsic
// curvature
template <size_t SpatialDim, typename FrameType>
tnsr::ii<DataVector, SpatialDim, FrameType>
get_dt_extrinsic_curvature_gauge_plane_wave(
    const tnsr::i<DataVector, SpatialDim, FrameType>& k,
    const Scalar<DataVector>& du_h, const Scalar<DataVector>& du_du_h,
    const Scalar<DataVector>& one_plus_h_times_omega_squared,
    const double omega) {
  tnsr::ii<DataVector, SpatialDim, FrameType> dt_extrinsic_curvature;
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&dt_extrinsic_curvature),
      1.0 / 2.0 * square(omega) * k(ti::i) * k(ti::j) * du_du_h() *
              Scalar<DataVector>(
                  pow(get(one_plus_h_times_omega_squared), -1.0 / 2.0))() -
          1.0 / 4.0 * pow(omega, 4) * k(ti::i) * k(ti::j) * du_h() * du_h() *
              Scalar<DataVector>(
                  pow(get(one_plus_h_times_omega_squared), -3.0 / 2.0))());
  return dt_extrinsic_curvature;
}

// calculate the exact time derivative of the spatial derivative of the
// conformal factor
template <size_t SpatialDim, typename FrameType>
tnsr::i<DataVector, SpatialDim, FrameType>
get_dt_d_conformal_factor_gauge_plane_wave(
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& dt_conformal_factor,
    const tnsr::II<DataVector, SpatialDim, FrameType>& inverse_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, FrameType>&
        dt_inverse_spatial_metric,
    const tnsr::ijj<DataVector, SpatialDim, FrameType>& d_spatial_metric,
    const tnsr::ijj<DataVector, SpatialDim, FrameType>& dt_d_spatial_metric) {
  tnsr::i<DataVector, SpatialDim, FrameType> dt_d_conformal_factor;
  ::tenex::evaluate<ti::i>(
      make_not_null(&dt_d_conformal_factor),
      -1.0 / 6.0 *
          (dt_conformal_factor() * inverse_spatial_metric(ti::J, ti::K) *
               d_spatial_metric(ti::i, ti::j, ti::k) +
           conformal_factor() * dt_inverse_spatial_metric(ti::J, ti::K) *
               d_spatial_metric(ti::i, ti::j, ti::k) +
           conformal_factor() * inverse_spatial_metric(ti::J, ti::K) *
               dt_d_spatial_metric(ti::i, ti::j, ti::k)));
  return dt_d_conformal_factor;
}

// calculate the exact time derivative of the spatial derivative of the
// spatial metric
template <size_t SpatialDim, typename FrameType>
tnsr::ijj<DataVector, SpatialDim, FrameType>
get_dt_d_spatial_metric_gauge_plane_wave(
    const tnsr::i<DataVector, SpatialDim, FrameType>& k,
    const Scalar<DataVector>& du_du_h, const double omega) {
  tnsr::ijj<DataVector, SpatialDim, FrameType> dt_d_spatial_metric;
  ::tenex::evaluate<ti::i, ti::j, ti::k>(
      make_not_null(&dt_d_spatial_metric),
      -1.0 * omega * k(ti::i) * k(ti::j) * k(ti::k) * du_du_h());
  return dt_d_spatial_metric;
}

// calculate the exact time derivative of the spatial derivative of the
// conformal spatial metric
template <size_t SpatialDim, typename FrameType>
tnsr::ijj<DataVector, SpatialDim, FrameType>
get_dt_d_conformal_spatial_metric_gauge_plane_wave(
    const tnsr::ii<DataVector, SpatialDim, FrameType>& spatial_metric,
    const tnsr::ii<DataVector, SpatialDim, FrameType>& dt_spatial_metric,
    const tnsr::ijj<DataVector, SpatialDim, FrameType>& d_spatial_metric,
    const tnsr::ijj<DataVector, SpatialDim, FrameType>& dt_d_spatial_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& dt_conformal_factor,
    const tnsr::i<DataVector, SpatialDim, FrameType>& d_conformal_factor,
    const tnsr::i<DataVector, SpatialDim, FrameType>& dt_d_conformal_factor) {
  tnsr::ijj<DataVector, SpatialDim, FrameType> dt_d_conformal_spatial_metric;
  ::tenex::evaluate<ti::l, ti::j, ti::k>(
      make_not_null(&dt_d_conformal_spatial_metric),
      2.0 * dt_spatial_metric(ti::j, ti::k) * conformal_factor() *
              d_conformal_factor(ti::l) +
          2.0 * spatial_metric(ti::j, ti::k) * dt_conformal_factor() *
              d_conformal_factor(ti::l) +
          2.0 * spatial_metric(ti::j, ti::k) * conformal_factor() *
              dt_d_conformal_factor(ti::l) +
          2.0 * conformal_factor() * dt_conformal_factor() *
              d_spatial_metric(ti::l, ti::j, ti::k) +
          conformal_factor() * conformal_factor() *
              dt_d_spatial_metric(ti::l, ti::j, ti::k));
  return dt_d_conformal_spatial_metric;
}

// calculate the exact time derivative of the gauge plane wave \hat{\Gamma}^i
// the Z4 constraint is assumed to be zero
template <size_t SpatialDim, typename FrameType>
tnsr::I<DataVector, SpatialDim, FrameType> get_dt_gamma_hat_gauge_plane_wave(
    const tnsr::II<DataVector, SpatialDim, FrameType>&
        inverse_conformal_spatial_metric,
    const tnsr::II<DataVector, SpatialDim, FrameType>&
        dt_inverse_conformal_spatial_metric,
    const tnsr::ijj<DataVector, SpatialDim, FrameType>&
        d_conformal_spatial_metric,
    const tnsr::ijj<DataVector, SpatialDim, FrameType>&
        dt_d_conformal_spatial_metric) {
  tnsr::I<DataVector, SpatialDim, FrameType> dt_gamma_hat;
  ::tenex::evaluate<ti::I>(
      make_not_null(&dt_gamma_hat),
      dt_inverse_conformal_spatial_metric(ti::I, ti::J) *
              inverse_conformal_spatial_metric(ti::K, ti::L) *
              d_conformal_spatial_metric(ti::l, ti::j, ti::k) +
          inverse_conformal_spatial_metric(ti::I, ti::J) *
              dt_inverse_conformal_spatial_metric(ti::K, ti::L) *
              d_conformal_spatial_metric(ti::l, ti::j, ti::k) +
          inverse_conformal_spatial_metric(ti::I, ti::J) *
              inverse_conformal_spatial_metric(ti::K, ti::L) *
              dt_d_conformal_spatial_metric(ti::l, ti::j, ti::k));
  return dt_gamma_hat;
}

// calculate the exact time derivative of the shift using Gamma driver
// b^i is assumed to be zero.
template <size_t SpatialDim, typename FrameType>
tnsr::I<DataVector, SpatialDim, FrameType> get_dt_shift_gauge_plane_wave(
    const tnsr::I<DataVector, SpatialDim, FrameType>& shift,
    const tnsr::iJ<DataVector, SpatialDim, FrameType>& d_shift) {
  auto dt_shift = make_with_value<tnsr::I<DataVector, SpatialDim, FrameType>>(
      get<0>(shift), 0.0);
  if (::Ccz4::fd::System::shifting_shift) {
    ::tenex::evaluate<ti::I>(make_not_null(&dt_shift),
                             shift(ti::K) * d_shift(ti::k, ti::I));
  }
  return dt_shift;
}

// calculate the exact time derivative of the b^i using Gamma driver
// b^i is assumed to be zero.
template <size_t SpatialDim, typename FrameType>
tnsr::I<DataVector, SpatialDim, FrameType> get_dt_b_gauge_plane_wave_expected(
    const tnsr::I<DataVector, SpatialDim, FrameType>& dt_gamma_hat,
    const tnsr::iJ<DataVector, SpatialDim, FrameType>& d_gamma_hat,
    const tnsr::I<DataVector, SpatialDim, FrameType>& shift) {
  tnsr::I<DataVector, SpatialDim, FrameType> dt_b;
  ::tenex::evaluate<ti::I>(make_not_null(&dt_b), dt_gamma_hat(ti::I));
  if (::Ccz4::fd::System::shifting_shift) {
    ::tenex::update<ti::I>(
        make_not_null(&dt_b),
        dt_b(ti::I) - shift(ti::K) * d_gamma_hat(ti::k, ti::I));
  }
  return dt_b;
}

// calculate the exact time derivative of the lapse using 1+log slicing
// theta is assumed to be zero. K0 is the trace extrinsic curvature on the
// initial time slice
template <size_t SpatialDim, typename FrameType>
Scalar<DataVector> get_dt_lapse_gauge_plane_wave(
    const tnsr::i<DataVector, SpatialDim, FrameType>& d_lapse,
    const tnsr::I<DataVector, SpatialDim, FrameType>& shift) {
  Scalar<DataVector> dt_lapse;
  ::tenex::evaluate(make_not_null(&dt_lapse), shift(ti::I) * d_lapse(ti::i));
  return dt_lapse;
}

// Compute D_{k}^{ij} from eq 14
template <size_t SpatialDim, typename FrameType>
tnsr::iJJ<DataVector, SpatialDim, FrameType> get_field_d_up(
    const tnsr::II<DataVector, SpatialDim, FrameType>&
        inverse_conformal_spatial_metric,
    const tnsr::ijj<DataVector, SpatialDim, FrameType>& field_d) {
  tnsr::iJJ<DataVector, SpatialDim, FrameType> field_d_up =
      gr::deriv_inverse_spatial_metric(inverse_conformal_spatial_metric,
                                       field_d);
  for (size_t i = 0; i < field_d.size(); i++) {
    field_d_up[i] *= -1.0;
  }
  return field_d_up;
}

inline Variables<
    ::Ccz4::fd::Tags::spacetime_reconstruction_tags>
compute_prim_solution_for_GaugePlaneWave(
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords, const double t,
    const gr::Solutions::GaugePlaneWave<3>& solution,
    const gr::Solutions::GaugePlaneWave<3>::IntermediateVars<DataVector>&
        intermediate_sol) {
  const Scalar<DataVector> h{intermediate_sol.h};
  const Scalar<DataVector> du_h{intermediate_sol.du_h};
  const Scalar<DataVector> du_du_h{intermediate_sol.du_du_h};

  const size_t SpatialDim = 3;

  // Setup solutions
  const auto gauge_plane_wave_vars = solution.variables(
      coords, t,
      typename gr::Solutions::GaugePlaneWave<SpatialDim>::tags<DataVector>{});

  // get system evolved vars
  Variables<typename ::Ccz4::fd::System::variables_tag::tags_list> evolved_vars{
      get<0>(coords).size()};

  using FrameType = Frame::Inertial;
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, SpatialDim, FrameType>>(
          gauge_plane_wave_vars);

  const auto det_spatial_metric = determinant(spatial_metric);
  const auto conformal_factor = pow(get(det_spatial_metric), -1. / 6.);
  Scalar<DataVector> conformal_factor_squared{};
  get(conformal_factor_squared) = square(conformal_factor);
  get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(evolved_vars) =
      KerrSchild::get_conformal_spatial_metric(conformal_factor_squared,
                                               spatial_metric);

  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(gauge_plane_wave_vars);

  get<gr::Tags::Shift<DataVector, 3>>(evolved_vars) =
      get<gr::Tags::Shift<DataVector, SpatialDim, FrameType>>(
          gauge_plane_wave_vars);

  get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars) =
      KerrSchild::get_trace_extrinsic_curvature(
          get<gr::Tags::ExtrinsicCurvature<DataVector, SpatialDim, FrameType>>(
              gauge_plane_wave_vars),
          get<gr::Tags::InverseSpatialMetric<DataVector, SpatialDim,
                                             FrameType>>(
              gauge_plane_wave_vars));

  get<::Ccz4::Tags::ATilde<DataVector, 3>>(evolved_vars) = ::Ccz4::a_tilde(
      conformal_factor_squared, spatial_metric,
      get<gr::Tags::ExtrinsicCurvature<DataVector, SpatialDim, FrameType>>(
          gauge_plane_wave_vars),
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(evolved_vars));

  const DataVector used_for_size = DataVector(
      get<0>(coords).size(), std::numeric_limits<double>::signaling_NaN());
  get<::Ccz4::Tags::Theta<DataVector>>(evolved_vars) =
      make_with_value<Scalar<DataVector>>(used_for_size, 0.0);

  const auto inverse_conformal_spatial_metric =
      determinant_and_inverse(
          get<::Ccz4::Tags::ConformalMetric<DataVector, 3>>(evolved_vars))
          .second;
  const auto d_det_spatial_metric = get_d_det_spatial_metric_gauge_plane_wave(
      det_spatial_metric,
      get<gr::Tags::InverseSpatialMetric<DataVector, SpatialDim, FrameType>>(
          gauge_plane_wave_vars),
      get<Tags::deriv<gr::Tags::SpatialMetric<DataVector, SpatialDim>,
                      tmpl::size_t<SpatialDim>, FrameType>>(
          gauge_plane_wave_vars));
  const auto& d_spatial_metric =
      get<Tags::deriv<gr::Tags::SpatialMetric<DataVector, SpatialDim>,
                      tmpl::size_t<SpatialDim>, FrameType>>(
          gauge_plane_wave_vars);
  const tnsr::ijj<DataVector, SpatialDim, FrameType>
      d_conformal_spatial_metric = KerrSchild::get_d_conformal_spatial_metric(
          conformal_factor_squared, spatial_metric, d_spatial_metric,
          d_det_spatial_metric);
  const tnsr::ijj<DataVector, SpatialDim, FrameType> field_d =
      KerrSchild::get_field_d(d_conformal_spatial_metric);
  const auto conformal_christoffel_second_kind =
      ::Ccz4::conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, field_d);
  const auto contracted_conformal_christoffel_second_kind =
      ::Ccz4::contracted_conformal_christoffel_second_kind(
          inverse_conformal_spatial_metric, conformal_christoffel_second_kind);
  get<::Ccz4::Tags::GammaHat<DataVector, 3>>(evolved_vars) =
      contracted_conformal_christoffel_second_kind;  // Z4 constraint is 0

  get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, 3>>(evolved_vars) =
      make_with_value<tnsr::I<DataVector, SpatialDim, FrameType>>(used_for_size,
                                                                  0.0);

  get<gr::Tags::Lapse<DataVector>>(evolved_vars) = lapse;

  get(get<::Ccz4::Tags::ConformalFactor<DataVector>>(evolved_vars)) =
      conformal_factor;

  return evolved_vars;
}
}  // namespace GaugePlaneWave
}  // namespace TestHelpers::Ccz4::fd::detail
