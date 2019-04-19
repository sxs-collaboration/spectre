// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/MinmodHelpers.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "Domain/Mesh.hpp"                // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Numeric.hpp"

namespace SlopeLimiters {
namespace Minmod_detail {

template <size_t VolumeDim>
void allocate_buffers(
    const gsl::not_null<std::unique_ptr<double[], decltype(&free)>*>
        contiguous_buffer,
    const gsl::not_null<DataVector*> u_lin_buffer,
    const gsl::not_null<std::array<DataVector, VolumeDim>*> boundary_buffer,
    const Mesh<VolumeDim>& mesh) noexcept {
  const size_t half_number_boundary_points = alg::accumulate(
      alg::iota(std::array<size_t, VolumeDim>{{}}, 0_st),
      0_st, [&mesh](const size_t state, const size_t d) noexcept {
        return state + mesh.slice_away(d).number_of_grid_points();
      });
  contiguous_buffer->reset(static_cast<double*>(
      malloc(sizeof(double) *
             (mesh.number_of_grid_points() + half_number_boundary_points))));
  size_t alloc_offset = 0;
  u_lin_buffer->set_data_ref(contiguous_buffer->get() + alloc_offset,
                             mesh.number_of_grid_points());
  alloc_offset += mesh.number_of_grid_points();
  for (size_t d = 0; d < VolumeDim; ++d) {
    const size_t num_points = mesh.slice_away(d).number_of_grid_points();
    gsl::at(*boundary_buffer, d)
        .set_data_ref(contiguous_buffer->get() + alloc_offset, num_points);
    alloc_offset += num_points;
  }
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                            \
  template void allocate_buffers<DIM(data)>(                            \
      const gsl::not_null<std::unique_ptr<double[], decltype(&free)>*>, \
      const gsl::not_null<DataVector*>,                                 \
      const gsl::not_null<std::array<DataVector, DIM(data)>*>,          \
      const Mesh<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace Minmod_detail
}  // namespace SlopeLimiters
