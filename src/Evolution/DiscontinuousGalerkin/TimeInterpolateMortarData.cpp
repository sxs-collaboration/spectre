// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/TimeInterpolateMortarData.hpp"

#include <boost/container/static_vector.hpp>
#include <cmath>
#include <cstddef>
#include <iterator>

#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "NumericalAlgorithms/Interpolation/LagrangePolynomial.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg {
namespace {
template <size_t Dim>
std::optional<gsl::not_null<const MortarData<Dim>*>>
time_interpolate_mortar_data_impl(
    const gsl::not_null<MortarData<Dim>*> interpolated,
    const TimeSteppers::ConstBoundaryHistorySide<MortarData<Dim>>& history,
    const double time, const bool include_geometric_data) {
  ASSERT(not history.empty(), "Cannot interpolate from an empty history.");

  for (const auto& step : history) {
    if (step.substep_time() == time) {
      return {&history.data(step)};
    }
  }

  const size_t interpolation_points =
      history.integration_order(history.size() - 1);
  ASSERT(history.size() >= interpolation_points,
         "Insufficient entries to interpolate: " << history.size() << " need "
                                                 << interpolation_points);

  // Center the stencil as well as possible on the goal time.  Usually
  // we will want to use the latest available data, but for large step
  // ratios we may want an earlier window.
  auto interpolation_begin = history.rbegin();
  auto interpolation_end = interpolation_begin +
                           static_cast<typename std::iterator_traits<
                               decltype(interpolation_begin)>::difference_type>(
                               interpolation_points);
  while (interpolation_end != history.rend() and
         std::abs(interpolation_end->substep_time() - time) <
             std::abs(interpolation_begin->substep_time() - time)) {
    ++interpolation_begin;
    ++interpolation_end;
  }

  boost::container::static_vector<double, 8> times{};
  for (auto step = interpolation_begin; step != interpolation_end; ++step) {
    times.push_back(step->substep_time());
  }
  *interpolated = MortarData<Dim>{};
  {
    const double coef =
        lagrange_polynomial(0, time, times.begin(), times.end());
    const auto& entry = history.data(*interpolation_begin);
    if (entry.mortar_data.has_value()) {
      interpolated->mortar_data.emplace(coef * *entry.mortar_data);
    }
    interpolated->mortar_mesh = entry.mortar_mesh;
    if (include_geometric_data) {
      if (entry.face_normal_magnitude.has_value()) {
        interpolated->face_normal_magnitude.emplace(
            coef * get(*entry.face_normal_magnitude));
      }
      if (entry.face_det_jacobian.has_value()) {
        interpolated->face_det_jacobian.emplace(coef *
                                                get(*entry.face_det_jacobian));
      }
      if (entry.volume_det_inv_jacobian.has_value()) {
        interpolated->volume_det_inv_jacobian.emplace(
            coef * get(*entry.volume_det_inv_jacobian));
      }
      interpolated->face_mesh = entry.face_mesh;
      interpolated->volume_mesh = entry.volume_mesh;
    }
  }
  for (size_t i = 1; i < interpolation_points; ++i) {
    const double coef =
        lagrange_polynomial(i, time, times.begin(), times.end());
    const auto& entry = history.data(
        *(interpolation_begin +
          static_cast<typename std::iterator_traits<
              decltype(interpolation_begin)>::difference_type>(i)));
    ASSERT(
        interpolated->mortar_data.has_value() == entry.mortar_data.has_value(),
        "Some entries have mortar_data but not others.");
    if (interpolated->mortar_data.has_value()) {
      *interpolated->mortar_data += coef * *entry.mortar_data;
    }
    ASSERT(interpolated->mortar_mesh == entry.mortar_mesh,
           "mortar_mesh does not match: " << interpolated->mortar_mesh
                                          << " and " << entry.mortar_mesh);
    if (include_geometric_data) {
      ASSERT(interpolated->face_normal_magnitude.has_value() ==
                 entry.face_normal_magnitude.has_value(),
             "Some entries have face_normal_magnitude but not others.");
      if (interpolated->face_normal_magnitude.has_value()) {
        get(*interpolated->face_normal_magnitude) +=
            coef * get(*entry.face_normal_magnitude);
      }
      ASSERT(interpolated->face_det_jacobian.has_value() ==
                 entry.face_det_jacobian.has_value(),
             "Some entries have face_det_jacobian but not others.");
      if (interpolated->face_det_jacobian.has_value()) {
        get(*interpolated->face_det_jacobian) +=
            coef * get(*entry.face_det_jacobian);
      }
      ASSERT(interpolated->volume_det_inv_jacobian.has_value() ==
                 entry.volume_det_inv_jacobian.has_value(),
             "Some entries have volume_det_inv_jacobian but not others.");
      if (interpolated->volume_det_inv_jacobian.has_value()) {
        get(*interpolated->volume_det_inv_jacobian) +=
            coef * get(*entry.volume_det_inv_jacobian);
      }
      ASSERT(interpolated->face_mesh == entry.face_mesh,
             "face_mesh does not match: " << interpolated->face_mesh << " and "
                                          << entry.face_mesh);
      ASSERT(interpolated->volume_mesh == entry.volume_mesh,
             "volume_mesh does not match: " << interpolated->volume_mesh
                                            << " and " << entry.volume_mesh);
    }
  }
  return {};
}
}  // namespace

template <size_t Dim>
void time_interpolate_mortar_data(
    const gsl::not_null<MortarData<Dim>*> interpolated,
    const TimeSteppers::ConstBoundaryHistorySide<MortarData<Dim>>& history,
    const double time, const bool include_geometric_data) {
  const auto existing_record = time_interpolate_mortar_data_impl(
      interpolated, history, time, include_geometric_data);
  if (existing_record.has_value()) {
    *interpolated = **existing_record;
  }
}

template <size_t Dim>
MaybeOwned<const MortarData<Dim>> time_interpolate_mortar_data(
    const TimeSteppers::ConstBoundaryHistorySide<MortarData<Dim>>& history,
    const double time, const bool include_geometric_data) {
  MortarData<Dim> new_record{};
  const auto existing_record = time_interpolate_mortar_data_impl(
      make_not_null(&new_record), history, time, include_geometric_data);
  if (existing_record.has_value()) {
    return {*existing_record};
  } else {
    return {std::move(new_record)};
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                               \
  template void time_interpolate_mortar_data(                              \
      gsl::not_null<MortarData<DIM(data)>*> interpolated,                  \
      const TimeSteppers::ConstBoundaryHistorySide<MortarData<DIM(data)>>& \
          history,                                                         \
      double time, bool include_geometric_data);                           \
  template MaybeOwned<const MortarData<DIM(data)>>                         \
  time_interpolate_mortar_data(                                            \
      const TimeSteppers::ConstBoundaryHistorySide<MortarData<DIM(data)>>& \
          history,                                                         \
      double time, bool include_geometric_data);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace evolution::dg
