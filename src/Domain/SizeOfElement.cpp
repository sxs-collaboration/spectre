// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/SizeOfElement.hpp"

#include <memory>
#include <string>
#include <unordered_map>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/ElementMap.hpp"  // IWYU pragma: keep
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdArrayHelpers.hpp"  // IWYU pragma: keep

namespace {
// The face-center coordinates for the unit cube's face in direction `dir`, so,
// * in 1D, returns (-1), (1)
// * in 2D, returns (-1, 0), (1, 0), (0, -1), (0, 1)
// * in 3D, returns (-1, 0, 0), ..., (0, 0, 1)
template <size_t VolumeDim>
inline tnsr::I<double, VolumeDim, Frame::Logical> logical_face_center(
    const Direction<VolumeDim>& dir) noexcept {
  tnsr::I<double, VolumeDim, Frame::Logical> result{{{0.0}}};
  result.get(dir.dimension()) = (dir.side() == Side::Lower ? -1.0 : 1.0);
  return result;
}

template <size_t VolumeDim>
inline double distance_between_face_centers(
    const tnsr::I<double, VolumeDim, Frame::Inertial>& lower_center,
    const tnsr::I<double, VolumeDim, Frame::Inertial>& upper_center) noexcept {
  auto center_to_center = make_array<VolumeDim>(0.0);
  alg::transform(upper_center, lower_center, center_to_center.begin(),
                 std::minus<>{});
  return magnitude(center_to_center);
}
}  // namespace

template <size_t VolumeDim>
std::array<double, VolumeDim> size_of_element(
    const ElementMap<VolumeDim, Frame::Inertial>&
        logical_to_inertial_map) noexcept {
  auto result = make_array<VolumeDim>(0.0);
  for (size_t logical_index = 0; logical_index < VolumeDim; ++logical_index) {
    const auto inertial_face_center =
        [&logical_index, &logical_to_inertial_map](const Side& side) noexcept {
          const Direction<VolumeDim> dir(logical_index, side);
          return logical_to_inertial_map(logical_face_center(dir));
        };
    const auto lower_center = inertial_face_center(Side::Lower);
    const auto upper_center = inertial_face_center(Side::Upper);
    result.at(logical_index) =
        distance_between_face_centers(lower_center, upper_center);
  }
  return result;
}

template <size_t VolumeDim>
std::array<double, VolumeDim> size_of_element(
    const ElementMap<VolumeDim, Frame::Grid>& logical_to_grid_map,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>&
        grid_to_inertial_map,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) noexcept {
  auto result = make_array<VolumeDim>(0.0);
  for (size_t logical_index = 0; logical_index < VolumeDim; ++logical_index) {
    const auto inertial_face_center = [&functions_of_time,
                                       &grid_to_inertial_map, &logical_index,
                                       &logical_to_grid_map,
                                       time](const Side& side) noexcept {
      const Direction<VolumeDim> dir(logical_index, side);
      return grid_to_inertial_map(logical_to_grid_map(logical_face_center(dir)),
                                  time, functions_of_time);
    };
    const auto lower_center = inertial_face_center(Side::Lower);
    const auto upper_center = inertial_face_center(Side::Upper);
    result.at(logical_index) =
        distance_between_face_centers(lower_center, upper_center);
  }
  return result;
}

// Explicit instantiations
#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template std::array<double, GET_DIM(data)> size_of_element(               \
      const ElementMap<GET_DIM(data), Frame::Inertial>&                     \
          logical_to_inertial_map) noexcept;                                \
  template std::array<double, GET_DIM(data)> size_of_element(               \
      const ElementMap<GET_DIM(data), Frame::Grid>& logical_to_grid_map,    \
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial,         \
                                      GET_DIM(data)>& grid_to_inertial_map, \
      const double time,                                                    \
      const std::unordered_map<                                             \
          std::string,                                                      \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&        \
          functions_of_time) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
