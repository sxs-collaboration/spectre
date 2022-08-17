// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ApparentHorizons/StrahlkorperCoordsInDifferentFrame.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/YlmSpherepack.hpp"
#include "Utilities/ContainerHelpers.hpp"
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
  // Block by Block.
  for (const auto& block : domain.blocks()) {
    // Once there are more possible source frames than the grid frame
    // (i.e. the distorted frame), then this static_assert will change,
    // and there will be an `if constexpr` below to treat the different
    // possible source frames.
    static_assert(std::is_same_v<SrcFrame, ::Frame::Grid>,
                  "Source frame must currently be Grid frame");
    static_assert(std::is_same_v<DestFrame, ::Frame::Inertial>,
                  "Destination frame must currently be Inertial frame");
    const auto& grid_to_inertial_map = block.moving_mesh_grid_to_inertial_map();
    const auto& logical_to_grid_map = block.moving_mesh_logical_to_grid_map();
    // Fill only those dest_cartesian_coords that are in this Block.
    // Determine which coords are in this Block by checking if the
    // inverse grid-to-logical map yields logical coords between -1 and 1.
    for (size_t s = 0; s < get<0>(src_cartesian_coords).size(); ++s) {
      const tnsr::I<double, 3, SrcFrame> x_src{
          {get<0>(src_cartesian_coords)[s], get<1>(src_cartesian_coords)[s],
           get<2>(src_cartesian_coords)[s]}};
      const auto x_logical = logical_to_grid_map.inverse(x_src);
      // x_logical might be an empty std::optional.
      if (x_logical.has_value() and get<0>(x_logical.value()) <= 1.0 and
          get<0>(x_logical.value()) >= -1.0 and
          get<1>(x_logical.value()) <= 1.0 and
          get<1>(x_logical.value()) >= -1.0 and
          get<2>(x_logical.value()) <= 1.0 and
          get<2>(x_logical.value()) >= -1.0) {
        const auto x_dest =
            grid_to_inertial_map(x_src, time, functions_of_time);
        get<0>(*dest_cartesian_coords)[s] = get<0>(x_dest);
        get<1>(*dest_cartesian_coords)[s] = get<1>(x_dest);
        get<2>(*dest_cartesian_coords)[s] = get<2>(x_dest);
        // Note that if a point is on the boundary of two or more
        // Blocks, it might get filled twice, but that's ok.
      }
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

GENERATE_INSTANTIATIONS(INSTANTIATE, (::Frame::Grid), (::Frame::Inertial))

#undef INSTANTIATE
#undef DESTFRAME
#undef SRCFRAME
