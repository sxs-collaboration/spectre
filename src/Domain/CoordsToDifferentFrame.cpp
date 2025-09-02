// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordsToDifferentFrame.hpp"

#include <cmath>
#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// Transforms cartesian coordinates from one frame to another, by calling
// block_logical_coordinates and calling the correct map functions.
template <typename SrcFrame, typename DestFrame>
void coords_to_different_frame(
    const gsl::not_null<tnsr::I<DataVector, 3, DestFrame>*>
        dest_cartesian_coords,
    const tnsr::I<DataVector, 3, SrcFrame>& src_cartesian_coords,
    const Domain<3>& domain,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const double time) {
  // If ever additional cases besides Grid->Inertial, Grid->Distorted,
  // and Inertial->Distorted are needed, add if constexprs below
  static_assert(std::is_same_v<SrcFrame, ::Frame::Grid> or
                    std::is_same_v<SrcFrame, ::Frame::Inertial>,
                "Source frame must currently be Grid frame or Inertial frame");
  const auto block_logical_coords = block_logical_coordinates(
      domain, src_cartesian_coords, time, functions_of_time);

  tnsr::I<double, 3, DestFrame> x_dest{};
  tnsr::I<double, 3, SrcFrame> x_src{};
  for (size_t s = 0; s < get<0>(src_cartesian_coords).size(); ++s) {
    get<0>(x_src) = get<0>(src_cartesian_coords)[s];
    get<1>(x_src) = get<1>(src_cartesian_coords)[s];
    get<2>(x_src) = get<2>(src_cartesian_coords)[s];

    // If this doesn't have a value, then the point isn't in the domain which is
    // really bad.
    if (UNLIKELY(not block_logical_coords[s].has_value())) {
      ERROR("A point in the " << SrcFrame{}
                              << " could not be mapped to a block: " << x_src);
    }

    const auto& block_id_and_coords = block_logical_coords[s].value();
    const auto& block = domain.blocks()[block_id_and_coords.id.get_index()];

    if constexpr (std::is_same_v<DestFrame, ::Frame::Distorted> and
                  std::is_same_v<SrcFrame, ::Frame::Grid>) {
      if (not block.has_distorted_frame()) {
        ERROR("Point lies outside of distorted-frame region");
      }
      const auto& grid_to_distorted_map =
          block.moving_mesh_grid_to_distorted_map();
      x_dest = grid_to_distorted_map(x_src, time, functions_of_time);
    } else if constexpr (std::is_same_v<DestFrame, ::Frame::Distorted> and
                         std::is_same_v<SrcFrame, ::Frame::Inertial>) {
      if (not block.has_distorted_frame()) {
        ERROR("Point lies outside of distorted-frame region");
      }
      const auto& distorted_to_inertial_map =
          block.moving_mesh_distorted_to_inertial_map();
      const auto& inv_point =
          distorted_to_inertial_map.inverse(x_src, time, functions_of_time);
      if (inv_point.has_value()) {
        x_dest = *inv_point;
      } else {
        ERROR("Map from Frame::Distorted to Frame::Inertial is not invertible");
      }
    } else if constexpr (std::is_same_v<DestFrame, ::Frame::Inertial> and
                         std::is_same_v<SrcFrame, ::Frame::Grid>) {
      const auto& grid_to_inertial_map =
          block.moving_mesh_grid_to_inertial_map();
      x_dest = grid_to_inertial_map(x_src, time, functions_of_time);
    } else {
      static_assert(std::is_same_v<DestFrame, ::Frame::Grid> and
                        std::is_same_v<SrcFrame, ::Frame::Inertial>,
                    "Source frame -> destination frame must be Grid -> "
                    "Distorted, Grid -> Inertial, Inertial -> Distorted, or "
                    "Inertial -> Grid");
      const auto& grid_to_inertial_map =
          block.moving_mesh_grid_to_inertial_map();
      const auto inv_point =
          grid_to_inertial_map.inverse(x_src, time, functions_of_time);
      if (inv_point.has_value()) {
        x_dest = inv_point.value();
      } else {
        ERROR("Map from Frame::Inertial to Frame::Grid is not invertible");
      }
    }

    get<0>(*dest_cartesian_coords)[s] = get<0>(x_dest);
    get<1>(*dest_cartesian_coords)[s] = get<1>(x_dest);
    get<2>(*dest_cartesian_coords)[s] = get<2>(x_dest);
  }
}

#define SRCFRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DESTFRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_GRID(_, data)                                         \
  template void coords_to_different_frame(                                \
      const gsl::not_null<tnsr::I<DataVector, 3, DESTFRAME(data)>*>       \
          dest_cartesian_coords,                                          \
      const tnsr::I<DataVector, 3, SRCFRAME(data)>& src_cartesian_coords, \
      const Domain<3>& domain,                                            \
      const std::unordered_map<                                           \
          std::string,                                                    \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&      \
          functions_of_time,                                              \
      const double time);

GENERATE_INSTANTIATIONS(INSTANTIATE_GRID, (::Frame::Grid),
                        (::Frame::Inertial, ::Frame::Distorted))

template void coords_to_different_frame(
    gsl::not_null<tnsr::I<DataVector, 3, ::Frame::Distorted>*>
        dest_cartesian_coords,
    const tnsr::I<DataVector, 3, ::Frame::Inertial>& src_cartesian_coords,
    const Domain<3>& domain,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    double time);

template void coords_to_different_frame(
    gsl::not_null<tnsr::I<DataVector, 3, ::Frame::Grid>*> dest_cartesian_coords,
    const tnsr::I<DataVector, 3, ::Frame::Inertial>& src_cartesian_coords,
    const Domain<3>& domain,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    double time);

#undef INSTANTIATE
#undef INSTANTIATEGENERAL
#undef INSTANTIATEALIGNED
#undef DESTFRAME
#undef SRCFRAME
