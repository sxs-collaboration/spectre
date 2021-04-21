// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/FiniteDifference/Reconstruct.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/StripeIterator.hpp"
#include "DataStructures/Transpose.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace fd::reconstruction {
namespace detail {
template <typename Reconstructor, size_t Dim, typename... ArgsForReconstructor>
void reconstruct_impl(
    const gsl::not_null<gsl::span<double>*> recons_upper,
    const gsl::not_null<gsl::span<double>*> recons_lower,
    const gsl::span<const double>& volume_vars,
    const gsl::span<const double>& lower_ghost_data,
    const gsl::span<const double>& upper_ghost_data,
    const Index<Dim>& volume_extents, const size_t number_of_variables,
    const ArgsForReconstructor&... args_for_reconstructor) noexcept {
  constexpr size_t stencil_width = Reconstructor::stencil_width();
  ASSERT(stencil_width % 2 == 1, "The stencil with should be odd but got "
                                     << stencil_width
                                     << " for the reconstructor.");
  const size_t ghost_zone_for_stencil = (stencil_width - 1) / 2;
  // Assume we send one extra ghost cell so we can reconstruct our neighbor's
  // external data.
  const size_t ghost_pts_in_neighbor_data = ghost_zone_for_stencil + 1;

  const size_t number_of_stripes =
      volume_extents.slice_away(0).product() * number_of_variables;

  std::array<double, stencil_width> q{};
  for (size_t slice = 0; slice < number_of_stripes; ++slice) {
    const size_t vars_slice_offset = slice * volume_extents[0];
    const size_t vars_neighbor_slice_offset =
        slice * ghost_pts_in_neighbor_data;
    const size_t recons_slice_offset = (volume_extents[0] + 1) * slice;

    // Deal with lower ghost data.
    //
    // There's one extra reconstruction for the upper face of the neighbor
    for (size_t j = 0; j < ghost_pts_in_neighbor_data; ++j) {
      q[j] = lower_ghost_data[vars_neighbor_slice_offset + j];
    }
    for (size_t j = ghost_pts_in_neighbor_data, k = 0; j < stencil_width;
         ++j, ++k) {
      gsl::at(q, j) = volume_vars[vars_slice_offset + k];
    }
    (*recons_lower)[recons_slice_offset] = Reconstructor::pointwise(
        q.data() + ghost_zone_for_stencil, 1, args_for_reconstructor...)[1];

    for (size_t i = 0; i < ghost_zone_for_stencil; ++i) {
      // offset comes from accounting for the 1 extra point in our ghost
      // cells plus how far away from the boundary we are reconstructing.
      for (size_t j = 0, offset = vars_neighbor_slice_offset +
                                  ghost_pts_in_neighbor_data -
                                  (ghost_zone_for_stencil - i);
           j < ghost_zone_for_stencil - i; ++j) {
        q[j] = lower_ghost_data[offset + j];
      }
      for (size_t j = ghost_zone_for_stencil - i, k = 0; j < stencil_width;
           ++j, ++k) {
        gsl::at(q, j) = volume_vars[vars_slice_offset + k];
      }
      const auto [upper_side_of_face, lower_side_of_face] =
          Reconstructor::pointwise(q.data() + ghost_zone_for_stencil, 1,
                                   args_for_reconstructor...);
      (*recons_upper)[recons_slice_offset + i] = upper_side_of_face;
      (*recons_lower)[recons_slice_offset + 1 + i] = lower_side_of_face;
    }

    // Reconstruct in the bulk
    const size_t slice_end = volume_extents[0] - ghost_zone_for_stencil;
    for (size_t vars_index = vars_slice_offset + ghost_zone_for_stencil,
                i = ghost_zone_for_stencil;
         i < slice_end; ++vars_index, ++i) {
      // Note: we keep the `stride` here because we may want to
      // experiment/support non-unit strides in the bulk in the future. For
      // cells where the reconstruction needs boundary data we copy into a
      // `std::array` buffer, which means we always have unit stride.
      constexpr int stride = 1;
      const auto [upper_side_of_face, lower_side_of_face] =
          Reconstructor::pointwise(&volume_vars[vars_index], stride,
                                   args_for_reconstructor...);
      (*recons_upper)[recons_slice_offset + i] = upper_side_of_face;
      (*recons_lower)[recons_slice_offset + 1 + i] = lower_side_of_face;
    }

    // Reconstruct using upper neighbor data
    for (size_t i = 0; i < ghost_zone_for_stencil; ++i) {
      // offset comes from accounting for the 1 extra point in our ghost
      // cells plus how far away from the boundary we are reconstructing.
      //
      // Note:
      // - q has size stencil_width
      // - we need to copy over (stencil_width - 1 - i) from the volume
      // - we need to copy (i + 1) from the neighbor
      //
      // Here is an example of a case with stencil_width 5:
      //
      //  Interior points| Neighbor points
      // x x x x x x x x | o o o
      //             ^
      //         c c c c | c
      //  c = points used for reconstruction
      size_t j = 0;
      for (size_t k =
               vars_slice_offset + volume_extents[0] - (stencil_width - 1 - i);
           j < stencil_width - 1 - i; ++j, ++k) {
        gsl::at(q, j) = volume_vars[k];
      }
      for (size_t k = 0; j < stencil_width; ++j, ++k) {
        gsl::at(q, j) = upper_ghost_data[vars_neighbor_slice_offset + k];
      }

      const auto [upper_side_of_face, lower_side_of_face] =
          Reconstructor::pointwise(q.data() + ghost_zone_for_stencil, 1,
                                   args_for_reconstructor...);
      (*recons_upper)[recons_slice_offset + slice_end + i] = upper_side_of_face;
      (*recons_lower)[recons_slice_offset + slice_end + i + 1] =
          lower_side_of_face;
    }

    // Reconstruct the upper side of the last face, this is what the
    // neighbor would've reconstructed.
    for (size_t j = 0; j < ghost_zone_for_stencil; ++j) {
      gsl::at(q, j) = volume_vars[vars_slice_offset + volume_extents[0] -
                                  ghost_zone_for_stencil + j];
    }
    for (size_t j = ghost_zone_for_stencil, k = 0; j < stencil_width;
         ++j, ++k) {
      gsl::at(q, j) = upper_ghost_data[vars_neighbor_slice_offset + k];
    }
    (*recons_upper)[recons_slice_offset + volume_extents[0]] =
        Reconstructor::pointwise(q.data() + ghost_zone_for_stencil, 1,
                                 args_for_reconstructor...)[0];
  }  // for slices
}

template <typename Reconstructor, size_t Dim, typename... ArgsForReconstructor>
void reconstruct(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_upper_side_of_face_vars,
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        reconstructed_lower_side_of_face_vars,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Index<Dim>& volume_extents, const size_t number_of_variables,
    const ArgsForReconstructor&... args_for_reconstructor) noexcept {
#ifdef SPECTRE_DEBUG
  ASSERT(volume_extents == Index<Dim>(volume_extents[0]),
         "The extents must be isotropic, but got " << volume_extents);
  const size_t number_of_points = volume_extents.product();
  for (size_t i = 0; i < Dim; ++i) {
    const size_t expected_pts =
        number_of_points / volume_extents[i] * (volume_extents[i] + 1);
    const size_t upper_num_pts =
        gsl::at(*reconstructed_upper_side_of_face_vars, i).size() /
        number_of_variables;
    ASSERT(upper_num_pts == expected_pts,
           "Incorrect number of points for "
           "reconstructed_upper_side_of_face_vars in direction "
               << i << ". Has " << upper_num_pts << " Expected "
               << expected_pts);
    const size_t lower_num_pts =
        gsl::at(*reconstructed_lower_side_of_face_vars, i).size() /
        number_of_variables;
    ASSERT(lower_num_pts == expected_pts,
           "Incorrect number of points for "
           "reconstructed_lower_side_of_face_vars in direction "
               << i << ". Has " << lower_num_pts << " Expected "
               << expected_pts);
  }
#endif  // SPECTRE_DEBUG

  ASSERT(ghost_cell_vars.contains(Direction<Dim>::lower_xi()),
         "Couldn't find lower ghost data in lower-xi");
  ASSERT(ghost_cell_vars.contains(Direction<Dim>::upper_xi()),
         "Couldn't find upper ghost data in upper-xi");
  reconstruct_impl<Reconstructor>(
      make_not_null(&(*reconstructed_upper_side_of_face_vars)[0]),
      make_not_null(&(*reconstructed_lower_side_of_face_vars)[0]), volume_vars,
      ghost_cell_vars.at(Direction<Dim>::lower_xi()),
      ghost_cell_vars.at(Direction<Dim>::upper_xi()), volume_extents,
      number_of_variables, args_for_reconstructor...);

  if constexpr (Dim > 1) {
    // We transpose from (x,y,z,vars) ordering to (y,z,vars,x) ordering
    // Might not be the most efficient (unclear), but easiest.
    // We use a single large buffer for both the y and z reconstruction
    // to reduce the number of memory allocations and improve data locality.
    const auto& lower_ghost = ghost_cell_vars.at(Direction<Dim>::lower_eta());
    const auto& upper_ghost = ghost_cell_vars.at(Direction<Dim>::upper_eta());
    std::vector<double> buffer(
        volume_vars.size() + lower_ghost.size() + upper_ghost.size() +
        2 * (*reconstructed_upper_side_of_face_vars)[1].size());
    raw_transpose(make_not_null(buffer.data()), volume_vars.data(),
                  volume_extents[0], volume_vars.size() / volume_extents[0]);
    raw_transpose(make_not_null(buffer.data() + volume_vars.size()),
                  lower_ghost.data(), volume_extents[0],
                  lower_ghost.size() / volume_extents[0]);
    raw_transpose(
        make_not_null(buffer.data() + volume_vars.size() + lower_ghost.size()),
        upper_ghost.data(), volume_extents[0],
        upper_ghost.size() / volume_extents[0]);

    // Note: assumes isotropic extents
    const size_t recons_offset_in_buffer =
        volume_vars.size() + lower_ghost.size() + upper_ghost.size();
    const size_t recons_size =
        (*reconstructed_upper_side_of_face_vars)[1].size();
    gsl::span<double> recons_upper_view =
        gsl::make_span(buffer.data() + recons_offset_in_buffer, recons_size);
    gsl::span<double> recons_lower_view = gsl::make_span(
        buffer.data() + recons_offset_in_buffer + recons_size, recons_size);
    reconstruct_impl<Reconstructor>(
        make_not_null(&recons_upper_view), make_not_null(&recons_lower_view),
        gsl::make_span(buffer),
        gsl::make_span(buffer.data() + volume_vars.size(), lower_ghost.size()),
        gsl::make_span(buffer.data() + volume_vars.size() + lower_ghost.size(),
                       upper_ghost.size()),
        volume_extents, number_of_variables, args_for_reconstructor...);
    // Transpose result back
    raw_transpose(
        make_not_null((*reconstructed_upper_side_of_face_vars)[1].data()),
        recons_upper_view.data(), recons_upper_view.size() / volume_extents[0],
        volume_extents[0]);
    raw_transpose(
        make_not_null((*reconstructed_lower_side_of_face_vars)[1].data()),
        recons_lower_view.data(), recons_lower_view.size() / volume_extents[0],
        volume_extents[0]);

    if constexpr (Dim > 2) {
      size_t chunk_size = volume_extents[0] * volume_extents[1];
      size_t number_of_volume_chunks = volume_vars.size() / chunk_size;
      size_t number_of_neighbor_chunks =
          ghost_cell_vars.at(Direction<Dim>::lower_zeta()).size() / chunk_size;

      raw_transpose(make_not_null(buffer.data()), volume_vars.data(),
                    chunk_size, number_of_volume_chunks);
      raw_transpose(make_not_null(buffer.data() + volume_vars.size()),
                    ghost_cell_vars.at(Direction<Dim>::lower_zeta()).data(),
                    chunk_size, number_of_neighbor_chunks);
      raw_transpose(make_not_null(buffer.data() + volume_vars.size() +
                                  lower_ghost.size()),
                    ghost_cell_vars.at(Direction<Dim>::upper_zeta()).data(),
                    chunk_size, number_of_neighbor_chunks);

      reconstruct_impl<Reconstructor>(
          make_not_null(&recons_upper_view), make_not_null(&recons_lower_view),
          gsl::make_span(buffer),
          gsl::make_span(buffer.data() + volume_vars.size(),
                         lower_ghost.size()),
          gsl::make_span(
              buffer.data() + volume_vars.size() + lower_ghost.size(),
              upper_ghost.size()),
          volume_extents, number_of_variables, args_for_reconstructor...);
      // Transpose result back
      raw_transpose(
          make_not_null((*reconstructed_upper_side_of_face_vars)[2].data()),
          recons_upper_view.data(), recons_upper_view.size() / chunk_size,
          chunk_size);
      raw_transpose(
          make_not_null((*reconstructed_lower_side_of_face_vars)[2].data()),
          recons_lower_view.data(), recons_lower_view.size() / chunk_size,
          chunk_size);
    }
  }
}
}  // namespace detail

template <Side LowerOrUpperSide, typename Reconstructor, size_t Dim,
          typename... ArgsForReconstructor>
void reconstruct_neighbor(
    const gsl::not_null<DataVector*> face_data, const DataVector& volume_data,
    const DataVector& neighbor_data, const Index<Dim>& volume_extents,
    const Index<Dim>& ghost_data_extents,
    const Direction<Dim>& direction_to_reconstruct,
    const ArgsForReconstructor&... args_for_reconstructor) noexcept {
  ASSERT(LowerOrUpperSide == direction_to_reconstruct.side(),
         "The template parameter LowerOrUpperSide ("
             << LowerOrUpperSide
             << ") must match the direction to reconstruct, "
             << direction_to_reconstruct
             << ". Note that we pass the Side in as a template parameter to "
                "avoid runtime branches in tight loops.");
  static_assert(Reconstructor::stencil_width() == 3 or
                    Reconstructor::stencil_width() == 5,
                "currently only support stencil widths of 3 and 5.");

  constexpr size_t index_of_pointwise = LowerOrUpperSide == Side::Upper ? 0 : 1;
  constexpr size_t neighbor_ghost_size =
      (Reconstructor::stencil_width() + 1) / 2;
  constexpr size_t offset_into_u_to_reconstruct =
      (Reconstructor::stencil_width() - 1) / 2;
  std::array<double, Reconstructor::stencil_width()> u_to_reconstruct{};
  if constexpr (Dim == 1) {
    (void)ghost_data_extents;
    (void)direction_to_reconstruct;

    constexpr bool upper_side = LowerOrUpperSide == Side::Upper;
    const size_t volume_index = upper_side ? volume_extents[0] - 1 : 0;
    if constexpr (Reconstructor::stencil_width() == 3) {
      u_to_reconstruct =
          std::array{upper_side ? volume_data[volume_index] : neighbor_data[0],
                     upper_side ? neighbor_data[0] : neighbor_data[1],
                     upper_side ? neighbor_data[1] : volume_data[volume_index]};
    } else if constexpr (Reconstructor::stencil_width() == 5) {
      u_to_reconstruct = std::array{
          upper_side ? volume_data[volume_index - 1] : neighbor_data[0],
          upper_side ? volume_data[volume_index] : neighbor_data[1],
          upper_side ? neighbor_data[0] : neighbor_data[2],
          upper_side ? neighbor_data[1] : volume_data[volume_index],
          upper_side ? neighbor_data[2] : volume_data[volume_index + 1]};
    }
    (*face_data)[0] = Reconstructor::pointwise(
        u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
        args_for_reconstructor...)[index_of_pointwise];
  } else if constexpr (Dim == 2) {
    (void)ghost_data_extents;
    if (direction_to_reconstruct == Direction<Dim>::lower_xi()) {
      for (size_t j = 0; j < volume_extents[1]; ++j) {
        if constexpr (Reconstructor::stencil_width() == 3) {
          u_to_reconstruct = std::array{
              neighbor_data[j * neighbor_ghost_size],
              neighbor_data[j * neighbor_ghost_size + 1],
              volume_data[collapsed_index(Index<Dim>(0, j), volume_extents)]};
        } else if constexpr (Reconstructor::stencil_width() == 5) {
          u_to_reconstruct = std::array{
              neighbor_data[j * neighbor_ghost_size],
              neighbor_data[j * neighbor_ghost_size + 1],
              neighbor_data[j * neighbor_ghost_size + 2],
              volume_data[collapsed_index(Index<Dim>(0, j), volume_extents)],
              volume_data[collapsed_index(Index<Dim>(1, j), volume_extents)]};
        }
        (*face_data)[j] = Reconstructor::pointwise(
            u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
            args_for_reconstructor...)[index_of_pointwise];
      }
    } else if (direction_to_reconstruct == Direction<Dim>::upper_xi()) {
      for (size_t j = 0; j < volume_extents[1]; ++j) {
        if constexpr (Reconstructor::stencil_width() == 3) {
          u_to_reconstruct = std::array{
              volume_data[collapsed_index(Index<Dim>(volume_extents[0] - 1, j),
                                          volume_extents)],
              neighbor_data[j * neighbor_ghost_size],
              neighbor_data[j * neighbor_ghost_size + 1]};
        } else if constexpr (Reconstructor::stencil_width() == 5) {
          u_to_reconstruct = std::array{
              volume_data[collapsed_index(Index<Dim>(volume_extents[0] - 2, j),
                                          volume_extents)],
              volume_data[collapsed_index(Index<Dim>(volume_extents[0] - 1, j),
                                          volume_extents)],
              neighbor_data[j * neighbor_ghost_size],
              neighbor_data[j * neighbor_ghost_size + 1],
              neighbor_data[j * neighbor_ghost_size + 2]};
        }
        (*face_data)[j] = Reconstructor::pointwise(
            u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
            args_for_reconstructor...)[index_of_pointwise];
      }
    } else if (direction_to_reconstruct == Direction<Dim>::lower_eta()) {
      for (size_t i = 0; i < volume_extents[0]; ++i) {
        if constexpr (Reconstructor::stencil_width() == 3) {
          u_to_reconstruct = std::array{
              neighbor_data[i], neighbor_data[i + volume_extents[0]],
              volume_data[collapsed_index(Index<Dim>(i, 0), volume_extents)]};
        } else if constexpr (Reconstructor::stencil_width() == 5) {
          u_to_reconstruct = std::array{
              neighbor_data[i], neighbor_data[i + volume_extents[0]],
              neighbor_data[i + 2 * volume_extents[0]],
              volume_data[collapsed_index(Index<Dim>(i, 0), volume_extents)],
              volume_data[collapsed_index(Index<Dim>(i, 1), volume_extents)]};
        }
        (*face_data)[i] = Reconstructor::pointwise(
            u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
            args_for_reconstructor...)[index_of_pointwise];
      }
    } else if (direction_to_reconstruct == Direction<Dim>::upper_eta()) {
      for (size_t i = 0; i < volume_extents[0]; ++i) {
        if constexpr (Reconstructor::stencil_width() == 3) {
          u_to_reconstruct = std::array{
              volume_data[collapsed_index(Index<Dim>(i, volume_extents[1] - 1),
                                          volume_extents)],
              neighbor_data[i], neighbor_data[i + volume_extents[0]]};
        } else if constexpr (Reconstructor::stencil_width() == 5) {
          u_to_reconstruct = std::array{
              volume_data[collapsed_index(Index<Dim>(i, volume_extents[1] - 2),
                                          volume_extents)],
              volume_data[collapsed_index(Index<Dim>(i, volume_extents[1] - 1),
                                          volume_extents)],
              neighbor_data[i], neighbor_data[i + volume_extents[0]],
              neighbor_data[i + 2 * volume_extents[0]]};
        }
        (*face_data)[i] = Reconstructor::pointwise(
            u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
            args_for_reconstructor...)[index_of_pointwise];
      }
    }
  } else {  // if constexpr (Dim == 3) is true
    if (direction_to_reconstruct == Direction<Dim>::lower_xi()) {
      const Index<Dim - 1> face_extents = volume_extents.slice_away(0);
      for (size_t k = 0; k < volume_extents[2]; ++k) {
        for (size_t j = 0; j < volume_extents[1]; ++j) {
          if constexpr (Reconstructor::stencil_width() == 3) {
            u_to_reconstruct =
                std::array{neighbor_data[collapsed_index(Index<Dim>(0, j, k),
                                                         ghost_data_extents)],
                           neighbor_data[collapsed_index(Index<Dim>(1, j, k),
                                                         ghost_data_extents)],
                           volume_data[collapsed_index(Index<Dim>(0, j, k),
                                                       volume_extents)]};
          } else if constexpr (Reconstructor::stencil_width() == 5) {
            u_to_reconstruct =
                std::array{neighbor_data[collapsed_index(Index<Dim>(0, j, k),
                                                         ghost_data_extents)],
                           neighbor_data[collapsed_index(Index<Dim>(1, j, k),
                                                         ghost_data_extents)],
                           neighbor_data[collapsed_index(Index<Dim>(2, j, k),
                                                         ghost_data_extents)],
                           volume_data[collapsed_index(Index<Dim>(0, j, k),
                                                       volume_extents)],
                           volume_data[collapsed_index(Index<Dim>(1, j, k),
                                                       volume_extents)]};
          }
          (*face_data)[collapsed_index(Index<Dim - 1>(j, k), face_extents)] =
              Reconstructor::pointwise(
                  u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
                  args_for_reconstructor...)[index_of_pointwise];
        }
      }
    } else if (direction_to_reconstruct == Direction<Dim>::upper_xi()) {
      const Index<Dim - 1> face_extents = volume_extents.slice_away(0);
      for (size_t k = 0; k < volume_extents[2]; ++k) {
        for (size_t j = 0; j < volume_extents[1]; ++j) {
          if constexpr (Reconstructor::stencil_width() == 3) {
            u_to_reconstruct = std::array{
                volume_data[collapsed_index(
                    Index<Dim>(volume_extents[0] - 1, j, k), volume_extents)],
                neighbor_data[collapsed_index(Index<Dim>(0, j, k),
                                              ghost_data_extents)],
                neighbor_data[collapsed_index(Index<Dim>(1, j, k),
                                              ghost_data_extents)]};
          } else if constexpr (Reconstructor::stencil_width() == 5) {
            u_to_reconstruct = std::array{
                volume_data[collapsed_index(
                    Index<Dim>(volume_extents[0] - 2, j, k), volume_extents)],
                volume_data[collapsed_index(
                    Index<Dim>(volume_extents[0] - 1, j, k), volume_extents)],
                neighbor_data[collapsed_index(Index<Dim>(0, j, k),
                                              ghost_data_extents)],
                neighbor_data[collapsed_index(Index<Dim>(1, j, k),
                                              ghost_data_extents)],
                neighbor_data[collapsed_index(Index<Dim>(2, j, k),
                                              ghost_data_extents)]};
          }
          (*face_data)[collapsed_index(Index<Dim - 1>(j, k), face_extents)] =
              Reconstructor::pointwise(
                  u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
                  args_for_reconstructor...)[index_of_pointwise];
        }
      }
    } else if (direction_to_reconstruct == Direction<Dim>::lower_eta()) {
      const Index<Dim - 1> face_extents = volume_extents.slice_away(1);
      for (size_t k = 0; k < volume_extents[2]; ++k) {
        for (size_t i = 0; i < volume_extents[0]; ++i) {
          if constexpr (Reconstructor::stencil_width() == 3) {
            u_to_reconstruct =
                std::array{neighbor_data[collapsed_index(Index<Dim>(i, 0, k),
                                                         ghost_data_extents)],
                           neighbor_data[collapsed_index(Index<Dim>(i, 1, k),
                                                         ghost_data_extents)],
                           volume_data[collapsed_index(Index<Dim>(i, 0, k),
                                                       volume_extents)]};
          } else if constexpr (Reconstructor::stencil_width() == 5) {
            u_to_reconstruct =
                std::array{neighbor_data[collapsed_index(Index<Dim>(i, 0, k),
                                                         ghost_data_extents)],
                           neighbor_data[collapsed_index(Index<Dim>(i, 1, k),
                                                         ghost_data_extents)],
                           neighbor_data[collapsed_index(Index<Dim>(i, 2, k),
                                                         ghost_data_extents)],
                           volume_data[collapsed_index(Index<Dim>(i, 0, k),
                                                       volume_extents)],
                           volume_data[collapsed_index(Index<Dim>(i, 1, k),
                                                       volume_extents)]};
          }
          (*face_data)[collapsed_index(Index<Dim - 1>(i, k), face_extents)] =
              Reconstructor::pointwise(
                  u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
                  args_for_reconstructor...)[index_of_pointwise];
        }
      }
    } else if (direction_to_reconstruct == Direction<Dim>::upper_eta()) {
      const Index<Dim - 1> face_extents = volume_extents.slice_away(1);
      for (size_t k = 0; k < volume_extents[2]; ++k) {
        for (size_t i = 0; i < volume_extents[0]; ++i) {
          if constexpr (Reconstructor::stencil_width() == 3) {
            u_to_reconstruct = std::array{
                volume_data[collapsed_index(
                    Index<Dim>(i, volume_extents[1] - 1, k), volume_extents)],
                neighbor_data[collapsed_index(Index<Dim>(i, 0, k),
                                              ghost_data_extents)],
                neighbor_data[collapsed_index(Index<Dim>(i, 1, k),
                                              ghost_data_extents)]};
          } else if constexpr (Reconstructor::stencil_width() == 5) {
            u_to_reconstruct = std::array{
                volume_data[collapsed_index(
                    Index<Dim>(i, volume_extents[1] - 2, k), volume_extents)],
                volume_data[collapsed_index(
                    Index<Dim>(i, volume_extents[1] - 1, k), volume_extents)],
                neighbor_data[collapsed_index(Index<Dim>(i, 0, k),
                                              ghost_data_extents)],
                neighbor_data[collapsed_index(Index<Dim>(i, 1, k),
                                              ghost_data_extents)],
                neighbor_data[collapsed_index(Index<Dim>(i, 2, k),
                                              ghost_data_extents)]};
          }
          (*face_data)[collapsed_index(Index<Dim - 1>(i, k), face_extents)] =
              Reconstructor::pointwise(
                  u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
                  args_for_reconstructor...)[index_of_pointwise];
        }
      }
    } else if (direction_to_reconstruct == Direction<Dim>::lower_zeta()) {
      const Index<Dim - 1> face_extents = volume_extents.slice_away(2);
      for (size_t j = 0; j < volume_extents[1]; ++j) {
        for (size_t i = 0; i < volume_extents[0]; ++i) {
          if constexpr (Reconstructor::stencil_width() == 3) {
            u_to_reconstruct =
                std::array{neighbor_data[collapsed_index(Index<Dim>(i, j, 0),
                                                         ghost_data_extents)],
                           neighbor_data[collapsed_index(Index<Dim>(i, j, 1),
                                                         ghost_data_extents)],
                           volume_data[collapsed_index(Index<Dim>(i, j, 0),
                                                       volume_extents)]};
          } else if constexpr (Reconstructor::stencil_width() == 5) {
            u_to_reconstruct =
                std::array{neighbor_data[collapsed_index(Index<Dim>(i, j, 0),
                                                         ghost_data_extents)],
                           neighbor_data[collapsed_index(Index<Dim>(i, j, 1),
                                                         ghost_data_extents)],
                           neighbor_data[collapsed_index(Index<Dim>(i, j, 2),
                                                         ghost_data_extents)],
                           volume_data[collapsed_index(Index<Dim>(i, j, 0),
                                                       volume_extents)],
                           volume_data[collapsed_index(Index<Dim>(i, j, 1),
                                                       volume_extents)]};
          }
          (*face_data)[collapsed_index(Index<Dim - 1>(i, j), face_extents)] =
              Reconstructor::pointwise(
                  u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
                  args_for_reconstructor...)[index_of_pointwise];
        }
      }
    } else if (direction_to_reconstruct == Direction<Dim>::upper_zeta()) {
      const Index<Dim - 1> face_extents = volume_extents.slice_away(2);
      for (size_t j = 0; j < volume_extents[1]; ++j) {
        for (size_t i = 0; i < volume_extents[0]; ++i) {
          if constexpr (Reconstructor::stencil_width() == 3) {
            u_to_reconstruct = std::array{
                volume_data[collapsed_index(
                    Index<Dim>(i, j, volume_extents[2] - 1), volume_extents)],
                neighbor_data[collapsed_index(Index<Dim>(i, j, 0),
                                              ghost_data_extents)],
                neighbor_data[collapsed_index(Index<Dim>(i, j, 1),
                                              ghost_data_extents)]};
          } else if constexpr (Reconstructor::stencil_width() == 5) {
            u_to_reconstruct = std::array{
                volume_data[collapsed_index(
                    Index<Dim>(i, j, volume_extents[2] - 2), volume_extents)],
                volume_data[collapsed_index(
                    Index<Dim>(i, j, volume_extents[2] - 1), volume_extents)],
                neighbor_data[collapsed_index(Index<Dim>(i, j, 0),
                                              ghost_data_extents)],
                neighbor_data[collapsed_index(Index<Dim>(i, j, 1),
                                              ghost_data_extents)],
                neighbor_data[collapsed_index(Index<Dim>(i, j, 2),
                                              ghost_data_extents)]};
          }
          (*face_data)[collapsed_index(Index<Dim - 1>(i, j), face_extents)] =
              Reconstructor::pointwise(
                  u_to_reconstruct.data() + offset_into_u_to_reconstruct, 1,
                  args_for_reconstructor...)[index_of_pointwise];
        }
      }
    }
  }
}
}  // namespace fd::reconstruction
