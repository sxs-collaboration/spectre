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
#include "Domain/Side.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdArrayHelpers.hpp"  // IWYU pragma: keep

template <size_t VolumeDim>
void size_of_element(
    const gsl::not_null<std::array<double, VolumeDim>*> result,
    const ElementMap<VolumeDim, Frame::Grid>& logical_to_grid_map,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, VolumeDim>&
        grid_to_inertial_map,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) noexcept {
  *result = make_array<VolumeDim>(0.0);
  for (size_t logical_dim = 0; logical_dim < VolumeDim; ++logical_dim) {
    const auto face_center = [&functions_of_time, &grid_to_inertial_map,
                              &logical_dim, &logical_to_grid_map,
                              time](const Side& side) noexcept {
      tnsr::I<double, VolumeDim, Frame::Logical> logical_center{{{0.0}}};
      logical_center.get(logical_dim) = (side == Side::Lower ? -1.0 : 1.0);
      const tnsr::I<double, VolumeDim, Frame::Inertial> inertial_center =
          grid_to_inertial_map(logical_to_grid_map(logical_center), time,
                               functions_of_time);
      return inertial_center;
    };
    const auto lower_center = face_center(Side::Lower);
    const auto upper_center = face_center(Side::Upper);

    // inertial-coord distance from lower face center to upper face center
    auto center_to_center = make_array<VolumeDim>(0.0);
    for (size_t inertial_dim = 0; inertial_dim < VolumeDim; ++inertial_dim) {
      center_to_center.at(inertial_dim) =
          upper_center.get(inertial_dim) - lower_center.get(inertial_dim);
    }

    result->at(logical_dim) = magnitude(center_to_center);
  }
}

// Explicit instantiations
#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template void size_of_element(                                            \
      gsl::not_null<std::array<double, GET_DIM(data)>*> result,             \
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
