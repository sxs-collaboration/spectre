// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/SizeOfElement.hpp"

#include "Domain/ElementMap.hpp"  // IWYU pragma: keep
#include "Domain/Side.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdArrayHelpers.hpp"  // IWYU pragma: keep

template <size_t VolumeDim>
std::array<double, VolumeDim> size_of_element(
    const ElementMap<VolumeDim, Frame::Inertial>& element_map) noexcept {
  auto result = make_array<VolumeDim>(0.0);
  for (size_t logical_dim = 0; logical_dim < VolumeDim; ++logical_dim) {
    const auto face_center =
        [&logical_dim, &element_map ](const Side& side) noexcept {
      tnsr::I<double, VolumeDim, Frame::Logical> logical_center{{{0.0}}};
      logical_center.get(logical_dim) = (side == Side::Lower ? -1.0 : 1.0);
      const tnsr::I<double, VolumeDim, Frame::Inertial> inertial_center =
          element_map(logical_center);
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

    result.at(logical_dim) = magnitude(center_to_center);
  }
  return result;
}

// Explicit instantiations
#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                \
  template std::array<double, GET_DIM(data)> size_of_element( \
      const ElementMap<GET_DIM(data), Frame::Inertial>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
