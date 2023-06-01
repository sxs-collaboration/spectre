// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/StrahlkorperCoordsInDifferentFrame.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

template <typename SrcFrame, typename DestFrame>
void strahlkorper_coords_in_different_frame(
    const gsl::not_null<tnsr::I<DataVector, 3, DestFrame>*>
        dest_cartesian_coords,
    const Strahlkorper<SrcFrame>& src_strahlkorper, const Domain<3>& domain,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const double time) {
  static_assert(std::is_same_v<DestFrame, ::Frame::Inertial>,
                "Destination frame must currently be Inertial frame");
  destructive_resize_components(
      dest_cartesian_coords, src_strahlkorper.ylm_spherepack().physical_size());

  // Temporary storage; reduce the number of allocations.
  Variables<tmpl::list<::Tags::Tempi<0, 2, ::Frame::Spherical<SrcFrame>>,
                       ::Tags::Tempi<1, 3, SrcFrame>,
                       ::Tags::TempI<2, 3, SrcFrame>, ::Tags::TempScalar<3>>>
      temp_buffer(src_strahlkorper.ylm_spherepack().physical_size());
  auto& src_theta_phi =
      get<::Tags::Tempi<0, 2, ::Frame::Spherical<SrcFrame>>>(temp_buffer);
  auto& r_hat = get<::Tags::Tempi<1, 3, SrcFrame>>(temp_buffer);
  auto& src_cartesian_coords = get<Tags::TempI<2, 3, SrcFrame>>(temp_buffer);
  auto& src_radius = get<::Tags::TempScalar<3>>(temp_buffer);

  StrahlkorperTags::ThetaPhiCompute<SrcFrame>::function(
      make_not_null(&src_theta_phi), src_strahlkorper);
  StrahlkorperTags::RhatCompute<SrcFrame>::function(make_not_null(&r_hat),
                                                    src_theta_phi);
  StrahlkorperTags::RadiusCompute<SrcFrame>::function(
      make_not_null(&src_radius), src_strahlkorper);
  StrahlkorperTags::CartesianCoordsCompute<SrcFrame>::function(
      make_not_null(&src_cartesian_coords), src_strahlkorper, src_radius,
      r_hat);

  // We now wish to map src_cartesian_coords to the destination frame.
  // Each Block will have a different map, so the mapping must be done
  // Block by Block.  Since each point in general has a different
  // block (given by block_logical_coordinates), this means we must
  // map point by point.
  const auto block_logical_coords = block_logical_coordinates(
      domain, src_cartesian_coords, time, functions_of_time);
  for (size_t s = 0; s < block_logical_coords.size(); ++s) {
    const tnsr::I<double, 3, SrcFrame> x_src{{get<0>(src_cartesian_coords)[s],
                                              get<1>(src_cartesian_coords)[s],
                                              get<2>(src_cartesian_coords)[s]}};
    ASSERT(block_logical_coords[s].has_value(),
           "Found a point (source coords " << x_src
                                           << ") that is not in any Block.");
    const auto& block =
        domain.blocks()[block_logical_coords[s].value().id.get_index()];
    if (block.is_time_dependent()) {
      if constexpr (std::is_same_v<SrcFrame, ::Frame::Grid>) {
        const auto x_dest = block.moving_mesh_grid_to_inertial_map()(
            x_src, time, functions_of_time);
        get<0>(*dest_cartesian_coords)[s] = get<0>(x_dest);
        get<1>(*dest_cartesian_coords)[s] = get<1>(x_dest);
        get<2>(*dest_cartesian_coords)[s] = get<2>(x_dest);
      } else {
        static_assert(std::is_same_v<SrcFrame, ::Frame::Distorted>,
                      "Src frame must be Distorted if it is not Grid");
        const auto x_dest = block.moving_mesh_distorted_to_inertial_map()(
            x_src, time, functions_of_time);
        get<0>(*dest_cartesian_coords)[s] = get<0>(x_dest);
        get<1>(*dest_cartesian_coords)[s] = get<1>(x_dest);
        get<2>(*dest_cartesian_coords)[s] = get<2>(x_dest);
      }
    } else {
      // If we get here, then the frames are actually the same, but
      // they have different frame tags.  So we just copy.
      get<0>(*dest_cartesian_coords)[s] = get<0>(x_src);
      get<1>(*dest_cartesian_coords)[s] = get<1>(x_src);
      get<2>(*dest_cartesian_coords)[s] = get<2>(x_src);
    }
  }
}

#define SRCFRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DESTFRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                         \
  template void strahlkorper_coords_in_different_frame(              \
      const gsl::not_null<tnsr::I<DataVector, 3, DESTFRAME(data)>*>  \
          dest_cartesian_coords,                                     \
      const Strahlkorper<SRCFRAME(data)>& src_strahlkorper,          \
      const Domain<3>& domain,                                       \
      const std::unordered_map<                                      \
          std::string,                                               \
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>& \
          functions_of_time,                                         \
      const double time);

GENERATE_INSTANTIATIONS(INSTANTIATE, (::Frame::Grid, ::Frame::Distorted),
                        (::Frame::Inertial))

#undef INSTANTIATE
#undef DESTFRAME
#undef SRCFRAME
