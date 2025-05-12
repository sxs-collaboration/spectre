// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/CombineVolumeGhostData.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
namespace {
template <size_t Dim>
void fill_combined_data(const gsl::not_null<DataVector*> extended_data,
                        const DataVector& volume_data,
                        const DataVector& ghost_data, const size_t ext_offset,
                        const size_t vol_offset, const size_t ghost_offset,
                        const Index<Dim>& ext_extents,
                        const Index<Dim>& vol_extents,
                        const Index<Dim>& ghost_extents,
                        const Direction<Dim>& direction_to_extend) {
  const bool is_lower_side = direction_to_extend.side() == Side::Lower;
  const size_t extended_dim = direction_to_extend.dimension();

  const size_t num_vol_pts = vol_extents.product();
  const size_t num_ghost_pts = ghost_extents.product();

  for (size_t i_vol = 0; i_vol < num_vol_pts; ++i_vol) {
    Index<Dim> multi_idx = expanded_index(i_vol, vol_extents);
    if (is_lower_side) {
      multi_idx[extended_dim] += ghost_extents[extended_dim];
    }
    const size_t combined_idx = collapsed_index(multi_idx, ext_extents);
    (*extended_data)[combined_idx + ext_offset] =
        volume_data[i_vol + vol_offset];
  }

  for (size_t i_ghost = 0; i_ghost < num_ghost_pts; ++i_ghost) {
    Index<Dim> multi_idx = expanded_index(i_ghost, ghost_extents);
    if (!is_lower_side) {
      multi_idx[extended_dim] += vol_extents[extended_dim];
    }
    const size_t combined_idx = collapsed_index(multi_idx, ext_extents);
    (*extended_data)[combined_idx + ext_offset] =
        ghost_data[i_ghost + ghost_offset];
  }
}

}  // namespace
template <size_t Dim>
DataVector combine_volume_ghost_data(
    const DataVector& volume_data, const DataVector& ghost_data,
    const Index<Dim>& subcell_extents, const size_t ghost_zone_size,
    const Direction<Dim>& direction_to_extend) {
  const size_t num_vol_pts = subcell_extents.product();
  const size_t num_components = volume_data.size() / num_vol_pts;
  const size_t extended_dim = direction_to_extend.dimension();

  Index<Dim> ghost_extents = subcell_extents;
  ghost_extents[extended_dim] = ghost_zone_size;

  Index<Dim> ext_extents = subcell_extents;
  ext_extents[extended_dim] += ghost_zone_size;

  const size_t num_ghost_pts = ghost_extents.product();
  const size_t num_ext_pts = ext_extents.product();

  ASSERT(volume_data.size() % num_vol_pts == 0,
         "volume_data size (" << volume_data.size()
                              << ") not divisible by number of volume points ("
                              << num_vol_pts << ").");

  ASSERT(ghost_data.size() == num_components * num_ghost_pts,
         "Expected ghost_data size to be " << num_components * num_ghost_pts
                                           << " but got " << ghost_data.size()
                                           << ".");

  DataVector combined_data{num_ext_pts * num_components};

  for (size_t comp = 0; comp < num_components; ++comp) {
    // component offsets for volume, ghost, and extended data.
    const size_t vol_offset = comp * num_vol_pts;
    const size_t ghost_offset = comp * num_ghost_pts;
    const size_t ext_offset = comp * num_ext_pts;

    fill_combined_data(&combined_data, volume_data, ghost_data, ext_offset,
                       vol_offset, ghost_offset, ext_extents, subcell_extents,
                       ghost_extents, direction_to_extend);
  }

  return combined_data;
}

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                         \
  template DataVector combine_volume_ghost_data(                     \
      const DataVector&, const DataVector&, const Index<DIM(data)>&, \
      const size_t, const Direction<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

}  // namespace evolution::dg::subcell
