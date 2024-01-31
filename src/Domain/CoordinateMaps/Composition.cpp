// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Composition.hpp"

#include <tuple>
#include <type_traits>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace domain::CoordinateMaps {

template <typename Frames, size_t Dim, size_t... Is>
Composition<Frames, Dim, std::index_sequence<Is...>>::Composition(
    std::unique_ptr<
        CoordinateMapBase<tmpl::at<frames, tmpl::size_t<Is>>,
                          tmpl::at<frames, tmpl::size_t<Is + 1>>, Dim>>... maps)
    : maps_{std::move(maps)...},
      function_of_time_names_(CoordinateMap_detail::initialize_names(
          maps_, std::index_sequence<Is...>{})) {}

template <typename Frames, size_t Dim, size_t... Is>
Composition<Frames, Dim, std::index_sequence<Is...>>&
Composition<Frames, Dim, std::index_sequence<Is...>>::operator=(
    const Composition& rhs) {
  expand_pack((get<Is>(maps_) = get<Is>(rhs.maps_)->get_clone())...);
  return *this;
}

template <typename Frames, size_t Dim, size_t... Is>
bool Composition<Frames, Dim, std::index_sequence<Is...>>::is_identity() const {
  bool result = true;
  EXPAND_PACK_LEFT_TO_RIGHT(
      (result = result and get<Is>(maps_)->is_identity()));
  return result;
}

template <typename Frames, size_t Dim, size_t... Is>
bool Composition<Frames, Dim,
                 std::index_sequence<Is...>>::inv_jacobian_is_time_dependent()
    const {
  bool result = false;
  EXPAND_PACK_LEFT_TO_RIGHT(
      (result = result or get<Is>(maps_)->inv_jacobian_is_time_dependent()));
  return result;
}

template <typename Frames, size_t Dim, size_t... Is>
bool Composition<Frames, Dim,
                 std::index_sequence<Is...>>::jacobian_is_time_dependent()
    const {
  bool result = false;
  EXPAND_PACK_LEFT_TO_RIGHT(
      (result = result or get<Is>(maps_)->jacobian_is_time_dependent()));
  return result;
}

template <typename Frames, size_t Dim, size_t... Is>
tnsr::I<double, Dim, tmpl::back<Frames>>
Composition<Frames, Dim, std::index_sequence<Is...>>::operator()(
    tnsr::I<double, Dim, SourceFrame> source_point, const double time,
    const FuncOfTimeMap& functions_of_time) const {
  return call_impl(std::move(source_point), time, functions_of_time);
}

template <typename Frames, size_t Dim, size_t... Is>
tnsr::I<DataVector, Dim, tmpl::back<Frames>>
Composition<Frames, Dim, std::index_sequence<Is...>>::operator()(
    tnsr::I<DataVector, Dim, SourceFrame> source_point, const double time,
    const FuncOfTimeMap& functions_of_time) const {
  return call_impl(std::move(source_point), time, functions_of_time);
}

template <typename Frames, size_t Dim, size_t... Is>
std::optional<tnsr::I<double, Dim, tmpl::front<Frames>>>
Composition<Frames, Dim, std::index_sequence<Is...>>::inverse(
    tnsr::I<double, Dim, tmpl::back<Frames>> target_point, const double time,
    const FuncOfTimeMap& functions_of_time) const {
  return inverse_impl(std::move(target_point), time, functions_of_time);
}

template <typename Frames, size_t Dim, size_t... Is>
InverseJacobian<double, Dim, tmpl::front<Frames>, tmpl::back<Frames>>
Composition<Frames, Dim, std::index_sequence<Is...>>::inv_jacobian(
    tnsr::I<double, Dim, SourceFrame> source_point, const double time,
    const FuncOfTimeMap& functions_of_time) const {
  return inv_jacobian_impl(std::move(source_point), time, functions_of_time);
}

template <typename Frames, size_t Dim, size_t... Is>
InverseJacobian<DataVector, Dim, tmpl::front<Frames>, tmpl::back<Frames>>
Composition<Frames, Dim, std::index_sequence<Is...>>::inv_jacobian(
    tnsr::I<DataVector, Dim, SourceFrame> source_point, const double time,
    const FuncOfTimeMap& functions_of_time) const {
  return inv_jacobian_impl(std::move(source_point), time, functions_of_time);
}

template <typename Frames, size_t Dim, size_t... Is>
Jacobian<double, Dim, tmpl::front<Frames>, tmpl::back<Frames>>
Composition<Frames, Dim, std::index_sequence<Is...>>::jacobian(
    tnsr::I<double, Dim, SourceFrame> source_point, const double time,
    const FuncOfTimeMap& functions_of_time) const {
  return jacobian_impl(std::move(source_point), time, functions_of_time);
}

template <typename Frames, size_t Dim, size_t... Is>
Jacobian<DataVector, Dim, tmpl::front<Frames>, tmpl::back<Frames>>
Composition<Frames, Dim, std::index_sequence<Is...>>::jacobian(
    tnsr::I<DataVector, Dim, SourceFrame> source_point, const double time,
    const FuncOfTimeMap& functions_of_time) const {
  return jacobian_impl(std::move(source_point), time, functions_of_time);
}

template <typename Frames, size_t Dim, size_t... Is>
[[noreturn]] std::tuple<
    tnsr::I<double, Dim, tmpl::back<Frames>>,
    InverseJacobian<double, Dim, tmpl::front<Frames>, tmpl::back<Frames>>,
    Jacobian<double, Dim, tmpl::front<Frames>, tmpl::back<Frames>>,
    tnsr::I<double, Dim, tmpl::back<Frames>>>
Composition<Frames, Dim, std::index_sequence<Is...>>::
    coords_frame_velocity_jacobians(
        tnsr::I<double, Dim, SourceFrame> /*source_point*/,
        const double /*time*/,
        const FuncOfTimeMap& /*functions_of_time*/) const {
  ERROR(
      "coords_frame_velocity_jacobians is not yet implemented in "
      "'Composition'. Please implement this function if you need it.");
}

template <typename Frames, size_t Dim, size_t... Is>
[[noreturn]] std::tuple<
    tnsr::I<DataVector, Dim, tmpl::back<Frames>>,
    InverseJacobian<DataVector, Dim, tmpl::front<Frames>, tmpl::back<Frames>>,
    Jacobian<DataVector, Dim, tmpl::front<Frames>, tmpl::back<Frames>>,
    tnsr::I<DataVector, Dim, tmpl::back<Frames>>>
Composition<Frames, Dim, std::index_sequence<Is...>>::
    coords_frame_velocity_jacobians(
        tnsr::I<DataVector, Dim, SourceFrame> /*source_point*/,
        const double /*time*/,
        const FuncOfTimeMap& /*functions_of_time*/) const {
  ERROR(
      "coords_frame_velocity_jacobians is not yet implemented in "
      "'Composition'. Please implement this function if you need it.");
}

template <typename Frames, size_t Dim, size_t... Is>
[[noreturn]] std::unique_ptr<
    CoordinateMapBase<tmpl::front<Frames>, Frame::Grid, Dim>>
Composition<Frames, Dim, std::index_sequence<Is...>>::get_to_grid_frame()
    const {
  // We probably don't need this.
  ERROR(
      "get_to_grid_frame is not implemented in 'Composition'. "
      "Just create a composition to the grid frame, or implement this "
      "function if you need it.");
}

// NOLINTNEXTLINE(google-runtime-references)
template <typename Frames, size_t Dim, size_t... Is>
void Composition<Frames, Dim, std::index_sequence<Is...>>::pup(PUP::er& p) {
  Base::pup(p);
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | maps_;
  }

  // No need to pup this because it is uniquely determined by the maps
  if (p.isUnpacking()) {
    function_of_time_names_ = CoordinateMap_detail::initialize_names(
        maps_, std::index_sequence<Is...>{});
  }
}

template <typename Frames, size_t Dim, size_t... Is>
template <typename DataType>
tnsr::I<DataType, Dim, tmpl::back<Frames>>
Composition<Frames, Dim, std::index_sequence<Is...>>::call_impl(
    tnsr::I<DataType, Dim, SourceFrame> source_point, const double time,
    const FuncOfTimeMap& functions_of_time) const {
  std::tuple<tnsr::I<DataType, Dim, SourceFrame>,
             tnsr::I<DataType, Dim, tmpl::at<frames, tmpl::size_t<Is + 1>>>...>
      points{};
  get<0>(points) = std::move(source_point);
  const auto apply = [&points, &time, &functions_of_time,
                      this](const auto index_v) {
    constexpr size_t index = decltype(index_v)::value;
    const auto& map = *get<index>(maps_);
    if (UNLIKELY(map.is_identity())) {
      for (size_t d = 0; d < Dim; ++d) {
        get<index + 1>(points).get(d) = std::move(get<index>(points).get(d));
      }
    } else {
      get<index + 1>(points) =
          map(std::move(get<index>(points)), time, functions_of_time);
    }
    return '0';
  };
  EXPAND_PACK_LEFT_TO_RIGHT(apply(tmpl::size_t<Is>{}));
  return get<tnsr::I<DataType, Dim, TargetFrame>>(points);
}

template <typename Frames, size_t Dim, size_t... Is>
template <typename DataType>
std::optional<tnsr::I<DataType, Dim, tmpl::front<Frames>>>
Composition<Frames, Dim, std::index_sequence<Is...>>::inverse_impl(
    tnsr::I<DataType, Dim, TargetFrame> target_point, const double time,
    const FuncOfTimeMap& functions_of_time) const {
  std::tuple<std::optional<tnsr::I<DataType, Dim, SourceFrame>>,
             std::optional<tnsr::I<DataType, Dim,
                                   tmpl::at<frames, tmpl::size_t<Is + 1>>>>...>
      points{};
  get<num_frames - 1>(points) = std::move(target_point);
  const auto apply_inverse = [&points, &time, &functions_of_time,
                              this](const auto index_v) {
    constexpr size_t index = decltype(index_v)::value;
    // index runs from 0 to num_frames - 2. We evaluate maps in reverse order.
    auto& local_target_point = get<num_frames - index - 1>(points);
    if (local_target_point.has_value()) {
      auto& local_source_point = get<num_frames - index - 2>(points);
      const auto& map = *get<num_frames - index - 2>(maps_);
      if (UNLIKELY(map.is_identity())) {
        local_source_point =
            tnsr::I<DataType, Dim,
                    tmpl::at<frames, tmpl::size_t<num_frames - index - 2>>>{};
        for (size_t d = 0; d < Dim; ++d) {
          local_source_point->get(d) = std::move(local_target_point->get(d));
        }
      } else {
        local_source_point = map.inverse(std::move(local_target_point.value()),
                                         time, functions_of_time);
      }
    }
    return '0';
  };
  EXPAND_PACK_LEFT_TO_RIGHT(apply_inverse(tmpl::size_t<Is>{}));
  return get<0>(points);
}

template <typename Frames, size_t Dim, size_t... Is>
template <typename DataType>
InverseJacobian<DataType, Dim, tmpl::front<Frames>, tmpl::back<Frames>>
Composition<Frames, Dim, std::index_sequence<Is...>>::inv_jacobian_impl(
    tnsr::I<DataType, Dim, SourceFrame> source_point, const double time,
    const FuncOfTimeMap& functions_of_time) const {
  std::tuple<tnsr::I<DataType, Dim, tmpl::at<frames, tmpl::size_t<Is>>>...>
      source_points{};
  std::tuple<InverseJacobian<DataType, Dim, SourceFrame,
                             tmpl::at<frames, tmpl::size_t<Is + 1>>>...>
      inv_jacobians{};
  get<0>(source_points) = std::move(source_point);
  const auto apply_inv_jacobian = [&source_points, &inv_jacobians, &time,
                                   &functions_of_time,
                                   this](const auto index_v) {
    constexpr size_t index = decltype(index_v)::value;
    const auto& map = *get<index>(maps_);
    auto& local_source_point = get<index>(source_points);
    if constexpr (index == 0) {
      get<0>(inv_jacobians) =
          map.inv_jacobian(local_source_point, time, functions_of_time);
    } else {
      auto& prev_inv_jacobian = get<index - 1>(inv_jacobians);
      if (UNLIKELY(map.is_identity())) {
        for (size_t i = 0; i < Dim; ++i) {
          for (size_t j = 0; j < Dim; ++j) {
            get<index>(inv_jacobians).get(i, j) =
                std::move(prev_inv_jacobian.get(i, j));
          }
        }
      } else {
        // Compose inverse Jacobians
        const auto next_inv_jacobian =
            map.inv_jacobian(local_source_point, time, functions_of_time);
        get<index>(inv_jacobians) = tenex::evaluate<ti::I, ti::j>(
            prev_inv_jacobian(ti::I, ti::k) * next_inv_jacobian(ti::K, ti::j));
      }
    }
    // Map next source point
    if constexpr (index < num_frames - 2) {
      if (UNLIKELY(map.is_identity())) {
        for (size_t d = 0; d < Dim; ++d) {
          get<index + 1>(source_points).get(d) =
              std::move(local_source_point.get(d));
        }
      } else {
        get<index + 1>(source_points) =
            map(std::move(local_source_point), time, functions_of_time);
      }
    }
    return '0';
  };
  EXPAND_PACK_LEFT_TO_RIGHT(apply_inv_jacobian(tmpl::size_t<Is>{}));
  return get<InverseJacobian<DataType, Dim, SourceFrame, TargetFrame>>(
      inv_jacobians);
}

template <typename Frames, size_t Dim, size_t... Is>
template <typename DataType>
Jacobian<DataType, Dim, tmpl::front<Frames>, tmpl::back<Frames>>
Composition<Frames, Dim, std::index_sequence<Is...>>::jacobian_impl(
    tnsr::I<DataType, Dim, SourceFrame> source_point, const double time,
    const FuncOfTimeMap& functions_of_time) const {
  std::tuple<tnsr::I<DataType, Dim, tmpl::at<frames, tmpl::size_t<Is>>>...>
      source_points{};
  std::tuple<Jacobian<DataType, Dim, SourceFrame,
                      tmpl::at<frames, tmpl::size_t<Is + 1>>>...>
      jacobians{};
  get<0>(source_points) = std::move(source_point);
  const auto apply_jacobian = [&source_points, &jacobians, &time,
                               &functions_of_time, this](const auto index_v) {
    constexpr size_t index = decltype(index_v)::value;
    const auto& map = *get<index>(maps_);
    auto& local_source_point = get<index>(source_points);
    if constexpr (index == 0) {
      get<0>(jacobians) =
          map.jacobian(local_source_point, time, functions_of_time);
    } else {
      auto& prev_jacobian = get<index - 1>(jacobians);
      if (UNLIKELY(map.is_identity())) {
        for (size_t i = 0; i < Dim; ++i) {
          for (size_t j = 0; j < Dim; ++j) {
            get<index>(jacobians).get(i, j) =
                std::move(prev_jacobian.get(i, j));
          }
        }
      } else {
        // Compose Jacobians
        const auto next_jacobian =
            map.jacobian(local_source_point, time, functions_of_time);
        get<index>(jacobians) = tenex::evaluate<ti::I, ti::j>(
            next_jacobian(ti::I, ti::k) * prev_jacobian(ti::K, ti::j));
      }
    }
    // Map next source point
    if constexpr (index < num_frames - 2) {
      if (UNLIKELY(map.is_identity())) {
        for (size_t d = 0; d < Dim; ++d) {
          get<index + 1>(source_points).get(d) =
              std::move(local_source_point.get(d));
        }
      } else {
        get<index + 1>(source_points) =
            map(std::move(local_source_point), time, functions_of_time);
      }
    }
    return '0';
  };
  EXPAND_PACK_LEFT_TO_RIGHT(apply_jacobian(tmpl::size_t<Is>{}));
  return get<Jacobian<DataType, Dim, SourceFrame, TargetFrame>>(jacobians);
}

template <typename Frames, size_t Dim, size_t... Is>
bool Composition<Frames, Dim, std::index_sequence<Is...>>::is_equal_to(
    const CoordinateMapBase<SourceFrame, TargetFrame, Dim>& other) const {
  const auto& cast_of_other = dynamic_cast<const Composition&>(other);

  return (... and (*get<Is>(maps_) == *get<Is>(cast_of_other.maps_)));
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template class Composition<                                                  \
      tmpl::list<Frame::BlockLogical, Frame::Grid, Frame::Inertial>,           \
      DIM(data)>;                                                              \
  template class Composition<                                                  \
      tmpl::list<Frame::BlockLogical, Frame::Grid, Frame::Distorted>,          \
      DIM(data)>;                                                              \
  template class Composition<tmpl::list<Frame::BlockLogical, Frame::Grid,      \
                                        Frame::Distorted, Frame::Inertial>,    \
                             DIM(data)>;                                       \
  template class Composition<                                                  \
      tmpl::list<Frame::ElementLogical, Frame::BlockLogical, Frame::Inertial>, \
      DIM(data)>;                                                              \
  template class Composition<                                                  \
      tmpl::list<Frame::ElementLogical, Frame::BlockLogical, Frame::Grid>,     \
      DIM(data)>;                                                              \
  template class Composition<                                                  \
      tmpl::list<Frame::ElementLogical, Frame::BlockLogical, Frame::Grid,      \
                 Frame::Inertial>,                                             \
      DIM(data)>;                                                              \
  template class Composition<                                                  \
      tmpl::list<Frame::ElementLogical, Frame::BlockLogical, Frame::Grid,      \
                 Frame::Distorted>,                                            \
      DIM(data)>;                                                              \
  template class Composition<                                                  \
      tmpl::list<Frame::ElementLogical, Frame::BlockLogical, Frame::Grid,      \
                 Frame::Distorted, Frame::Inertial>,                           \
      DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE

#if defined(__clang__) && __clang_major__ >= 15 && __clang_major__ < 17
#define INSTANTIATE2(_, data)                                                  \
  template domain::CoordinateMaps::Composition<                                \
      brigand::list<Frame::BlockLogical, Frame::Grid, Frame::Distorted>,       \
      DIM(data), std::integer_sequence<unsigned long, 0ul, 1ul>>::             \
      Composition(                                                             \
          std::unique_ptr<domain::CoordinateMapBase<Frame::BlockLogical,       \
                                                    Frame::Grid, DIM(data)>,   \
                          std::default_delete<domain::CoordinateMapBase<       \
                              Frame::BlockLogical, Frame::Grid, DIM(data)>>>,  \
          std::unique_ptr<domain::CoordinateMapBase<                           \
                              Frame::Grid, Frame::Distorted, DIM(data)>,       \
                          std::default_delete<domain::CoordinateMapBase<       \
                              Frame::Grid, Frame::Distorted, DIM(data)>>>);    \
  template domain::CoordinateMaps::Composition<                                \
      brigand::list<Frame::BlockLogical, Frame::Grid, Frame::Inertial>,        \
      DIM(data), std::integer_sequence<unsigned long, 0ul, 1ul>>::             \
      Composition(                                                             \
          std::unique_ptr<domain::CoordinateMapBase<Frame::BlockLogical,       \
                                                    Frame::Grid, DIM(data)>,   \
                          std::default_delete<domain::CoordinateMapBase<       \
                              Frame::BlockLogical, Frame::Grid, DIM(data)>>>,  \
          std::unique_ptr<domain::CoordinateMapBase<                           \
                              Frame::Grid, Frame::Inertial, DIM(data)>,        \
                          std::default_delete<domain::CoordinateMapBase<       \
                              Frame::Grid, Frame::Inertial, DIM(data)>>>);     \
  template domain::CoordinateMaps::Composition<                                \
      brigand::list<Frame::BlockLogical, Frame::Grid, Frame::Distorted,        \
                    Frame::Inertial>,                                          \
      DIM(data), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul>>::        \
      Composition(                                                             \
          std::unique_ptr<domain::CoordinateMapBase<Frame::BlockLogical,       \
                                                    Frame::Grid, DIM(data)>,   \
                          std::default_delete<domain::CoordinateMapBase<       \
                              Frame::BlockLogical, Frame::Grid, DIM(data)>>>,  \
          std::unique_ptr<domain::CoordinateMapBase<                           \
                              Frame::Grid, Frame::Distorted, DIM(data)>,       \
                          std::default_delete<domain::CoordinateMapBase<       \
                              Frame::Grid, Frame::Distorted, DIM(data)>>>,     \
          std::unique_ptr<                                                     \
              domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial,     \
                                        DIM(data)>,                            \
              std::default_delete<domain::CoordinateMapBase<                   \
                  Frame::Distorted, Frame::Inertial, DIM(data)>>>);            \
  template domain::CoordinateMaps::Composition<                                \
      brigand::list<Frame::ElementLogical, Frame::BlockLogical,                \
                    Frame::Inertial>,                                          \
      DIM(data), std::integer_sequence<unsigned long, 0ul, 1ul>>::             \
      Composition(                                                             \
          std::unique_ptr<                                                     \
              domain::CoordinateMapBase<Frame::ElementLogical,                 \
                                        Frame::BlockLogical, DIM(data)>,       \
              std::default_delete<domain::CoordinateMapBase<                   \
                  Frame::ElementLogical, Frame::BlockLogical, DIM(data)>>>,    \
          std::unique_ptr<                                                     \
              domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial,  \
                                        DIM(data)>,                            \
              std::default_delete<domain::CoordinateMapBase<                   \
                  Frame::BlockLogical, Frame::Inertial, DIM(data)>>>);         \
  template domain::CoordinateMaps::Composition<                                \
      brigand::list<Frame::ElementLogical, Frame::BlockLogical, Frame::Grid>,  \
      DIM(data), std::integer_sequence<unsigned long, 0ul, 1ul>>::             \
      Composition(                                                             \
          std::unique_ptr<                                                     \
              domain::CoordinateMapBase<Frame::ElementLogical,                 \
                                        Frame::BlockLogical, DIM(data)>,       \
              std::default_delete<domain::CoordinateMapBase<                   \
                  Frame::ElementLogical, Frame::BlockLogical, DIM(data)>>>,    \
          std::unique_ptr<domain::CoordinateMapBase<Frame::BlockLogical,       \
                                                    Frame::Grid, DIM(data)>,   \
                          std::default_delete<domain::CoordinateMapBase<       \
                              Frame::BlockLogical, Frame::Grid, DIM(data)>>>); \
  template domain::CoordinateMaps::Composition<                                \
      brigand::list<Frame::ElementLogical, Frame::BlockLogical, Frame::Grid,   \
                    Frame::Distorted>,                                         \
      DIM(data), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul>>::        \
      Composition(                                                             \
          std::unique_ptr<                                                     \
              domain::CoordinateMapBase<Frame::ElementLogical,                 \
                                        Frame::BlockLogical, DIM(data)>,       \
              std::default_delete<domain::CoordinateMapBase<                   \
                  Frame::ElementLogical, Frame::BlockLogical, DIM(data)>>>,    \
          std::unique_ptr<domain::CoordinateMapBase<Frame::BlockLogical,       \
                                                    Frame::Grid, DIM(data)>,   \
                          std::default_delete<domain::CoordinateMapBase<       \
                              Frame::BlockLogical, Frame::Grid, DIM(data)>>>,  \
          std::unique_ptr<domain::CoordinateMapBase<                           \
                              Frame::Grid, Frame::Distorted, DIM(data)>,       \
                          std::default_delete<domain::CoordinateMapBase<       \
                              Frame::Grid, Frame::Distorted, DIM(data)>>>);    \
  template domain::CoordinateMaps::Composition<                                \
      brigand::list<Frame::ElementLogical, Frame::BlockLogical, Frame::Grid,   \
                    Frame::Inertial>,                                          \
      DIM(data), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul>>::        \
      Composition(                                                             \
          std::unique_ptr<                                                     \
              domain::CoordinateMapBase<Frame::ElementLogical,                 \
                                        Frame::BlockLogical, DIM(data)>,       \
              std::default_delete<domain::CoordinateMapBase<                   \
                  Frame::ElementLogical, Frame::BlockLogical, DIM(data)>>>,    \
          std::unique_ptr<domain::CoordinateMapBase<Frame::BlockLogical,       \
                                                    Frame::Grid, DIM(data)>,   \
                          std::default_delete<domain::CoordinateMapBase<       \
                              Frame::BlockLogical, Frame::Grid, DIM(data)>>>,  \
          std::unique_ptr<domain::CoordinateMapBase<                           \
                              Frame::Grid, Frame::Inertial, DIM(data)>,        \
                          std::default_delete<domain::CoordinateMapBase<       \
                              Frame::Grid, Frame::Inertial, DIM(data)>>>);     \
  template domain::CoordinateMaps::Composition<                                \
      brigand::list<Frame::ElementLogical, Frame::BlockLogical, Frame::Grid,   \
                    Frame::Distorted, Frame::Inertial>,                        \
      DIM(data), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul>>::   \
      Composition(                                                             \
          std::unique_ptr<                                                     \
              domain::CoordinateMapBase<Frame::ElementLogical,                 \
                                        Frame::BlockLogical, DIM(data)>,       \
              std::default_delete<domain::CoordinateMapBase<                   \
                  Frame::ElementLogical, Frame::BlockLogical, DIM(data)>>>,    \
          std::unique_ptr<domain::CoordinateMapBase<Frame::BlockLogical,       \
                                                    Frame::Grid, DIM(data)>,   \
                          std::default_delete<domain::CoordinateMapBase<       \
                              Frame::BlockLogical, Frame::Grid, DIM(data)>>>,  \
          std::unique_ptr<domain::CoordinateMapBase<                           \
                              Frame::Grid, Frame::Distorted, DIM(data)>,       \
                          std::default_delete<domain::CoordinateMapBase<       \
                              Frame::Grid, Frame::Distorted, DIM(data)>>>,     \
          std::unique_ptr<                                                     \
              domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial,     \
                                        DIM(data)>,                            \
              std::default_delete<domain::CoordinateMapBase<                   \
                  Frame::Distorted, Frame::Inertial, DIM(data)>>>);

GENERATE_INSTANTIATIONS(INSTANTIATE2, (1, 2, 3))

#undef INSTANTIATE2
#endif /* defined(__clang__) && __clang_major__ >= 16 */

#undef DIM

}  // namespace domain::CoordinateMaps
