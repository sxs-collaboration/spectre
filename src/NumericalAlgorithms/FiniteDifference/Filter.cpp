// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/FiniteDifference/PartialDerivatives.hpp"

#include <array>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Transpose.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace fd {
namespace {
template <size_t Order, bool UnitStride>
struct KoDissipationImpl;

template <bool UnitStride>
struct KoDissipationImpl<2, UnitStride> {
  static constexpr size_t fd_order = 2;

  SPECTRE_ALWAYS_INLINE static double pointwise(const double* const q,
                                                const int stride,
                                                const double epsilon) {
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      return epsilon *
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             ((q[-1] + q[1]) - 2.0 * q[0]);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return epsilon * ((q[-stride] + q[stride]) - 2.0 * q[0]);
    }
  }
};

template <bool UnitStride>
struct KoDissipationImpl<4, UnitStride> {
  static constexpr size_t fd_order = 4;

  SPECTRE_ALWAYS_INLINE static double pointwise(const double* const q,
                                                const int stride,
                                                const double epsilon) {
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      return -epsilon *
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             (0.0625 * (q[-2] + q[2]) - 0.25 * (q[-1] + q[1]) + 0.375 * q[0]);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return -epsilon * (0.0625 * (q[-2 * stride] + q[2 * stride]) -
                         0.25 * (q[-stride] + q[stride]) + 0.375 * q[0]);
    }
  }
};

template <bool UnitStride>
struct KoDissipationImpl<6, UnitStride> {
  static constexpr size_t fd_order = 6;

  SPECTRE_ALWAYS_INLINE static double pointwise(const double* const q,
                                                const int stride,
                                                const double epsilon) {
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return epsilon * (0.015625 * (q[-3] + q[3]) - 0.09375 * (q[-2] + q[2]) +
                        0.234375 * (q[-1] + q[1]) - 0.3125 * q[0]);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return epsilon * (0.015625 * (q[-3 * stride] + q[3 * stride]) -
                        0.09375 * (q[-2 * stride] + q[2 * stride]) +
                        0.234375 * (q[-stride] + q[stride]) - 0.3125 * q[0]);
    }
  }
};

template <bool UnitStride>
struct KoDissipationImpl<8, UnitStride> {
  static constexpr size_t fd_order = 8;

  SPECTRE_ALWAYS_INLINE static double pointwise(const double* const q,
                                                const int stride,
                                                const double epsilon) {
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return epsilon * (-0.00390625 * (q[-4] + q[4]) +
                        0.03125 * (q[-3] + q[3]) - 0.109375 * (q[-2] + q[2]) +
                        0.21875 * (q[-1] + q[1]) - 0.2734375 * q[0]);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return epsilon * (-0.00390625 * (q[-4 * stride] + q[4 * stride]) +
                        0.03125 * (q[-3 * stride] + q[3 * stride]) -
                        0.109375 * (q[-2 * stride] + q[2 * stride]) +
                        0.21875 * (q[-stride] + q[stride]) - 0.2734375 * q[0]);
    }
  }
};

template <bool UnitStride>
struct KoDissipationImpl<10, UnitStride> {
  static constexpr size_t fd_order = 10;

  SPECTRE_ALWAYS_INLINE static double pointwise(const double* const q,
                                                const int stride,
                                                const double epsilon) {
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return epsilon *
             (0.0009765625 * (q[-5] + q[5]) - 0.009765625 * (q[-4] + q[4]) +
              0.0439453125 * (q[-3] + q[3]) - 0.1171875 * (q[-2] + q[2]) +
              0.205078125 * (q[-1] + q[1]) - 0.24609375 * q[0]);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return epsilon *
             (0.0009765625 * (q[-5 * stride] + q[5 * stride]) -
              0.009765625 * (q[-4 * stride] + q[4 * stride]) +
              0.0439453125 * (q[-3 * stride] + q[3 * stride]) -
              0.1171875 * (q[-2 * stride] + q[2 * stride]) +
              0.205078125 * (q[-stride] + q[stride]) - 0.24609375 * q[0]);
    }
  }
};

template <size_t Order, bool UnitStride>
struct LowPassFilterImpl;

template <bool UnitStride>
struct LowPassFilterImpl<2, UnitStride> {
  static constexpr size_t fd_order = 2;

  SPECTRE_ALWAYS_INLINE static double pointwise(const double* const q,
                                                const int stride,
                                                const double epsilon) {
    // 3/8 = 0.375
    // 3/4 = 0.75
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      return epsilon * (0.375 * (q[-1] + q[1]) - 0.75 * q[0]);
    } else {
      return epsilon * (0.375 * (q[-stride] + q[stride]) - 0.75 * q[0]);
    }
  }
};

template <bool UnitStride>
struct LowPassFilterImpl<4, UnitStride> {
  static constexpr size_t fd_order = 4;

  SPECTRE_ALWAYS_INLINE static double pointwise(const double* const q,
                                                const int stride,
                                                const double epsilon) {
    // 125/896 = 0.13950892857142858
    // 125/224 = 0.5580357142857143
    // 375/448 = 0.8370535714285714
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      return epsilon *
             (-0.13950892857142858 * (q[-2] + q[2]) +
              0.5580357142857143 * (q[-1] + q[1]) - 0.8370535714285714 * q[0]);
    } else {
      return epsilon *
             (-0.13950892857142858 * (q[-2 * stride] + q[2 * stride]) +
              0.5580357142857143 * (q[-stride] + q[stride]) -
              0.8370535714285714 * q[0]);
    }
  }
};

template <bool UnitStride>
struct LowPassFilterImpl<6, UnitStride> {
  static constexpr size_t fd_order = 6;

  SPECTRE_ALWAYS_INLINE static double pointwise(const double* const q,
                                                const int stride,
                                                const double epsilon) {
    // 16807/304128 = 0.05526291561447812
    // 16807/50688 = 0.3315774936868687
    // 84035/101376 = 0.8289437342171717
    // 84035/76032 = 1.1052583122895623
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      return epsilon *
             (0.05526291561447812 * (q[-3] + q[3]) -
              0.3315774936868687 * (q[-2] + q[2]) +
              0.8289437342171717 * (q[-1] + q[1]) - 1.1052583122895623 * q[0]);
    } else {
      return epsilon * (0.05526291561447812 * (q[-3 * stride] + q[3 * stride]) -
                        0.3315774936868687 * (q[-2 * stride] + q[2 * stride]) +
                        0.8289437342171717 * (q[-stride] + q[stride]) -
                        1.1052583122895623 * q[0]);
    }
  }
};

template <bool UnitStride>
struct LowPassFilterImpl<8, UnitStride> {
  static constexpr size_t fd_order = 8;

  SPECTRE_ALWAYS_INLINE static double pointwise(const double* const q,
                                                const int stride,
                                                const double epsilon) {
    // 531441/6406400 * 35 / 128 = 0.022682926204654723
    // 531441/800800 * 35 / 128 = 0.18146340963723778
    // 531441/228800 * 35 / 128 = 0.6351219337303321
    // 531441/114400 * 35 / 128 = 1.2702438674606642
    // 531441/91520 * 35 / 128 = 1.5878048343258304
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      return epsilon *
             (-0.022682926204654723 * (q[-4] + q[4]) +
              0.18146340963723778 * (q[-3] + q[3]) -
              0.6351219337303321 * (q[-2] + q[2]) +
              1.2702438674606642 * (q[-1] + q[1]) - 1.5878048343258304 * q[0]);
    } else {
      return epsilon *
             (-0.022682926204654723 * (q[-4 * stride] + q[4 * stride]) +
              0.18146340963723778 * (q[-3 * stride] + q[3 * stride]) -
              0.6351219337303321 * (q[-2 * stride] + q[2 * stride]) +
              1.2702438674606642 * (q[-stride] + q[stride]) -
              1.5878048343258304 * q[0]);
    }
  }
};

template <bool UnitStride>
struct LowPassFilterImpl<10, UnitStride> {
  static constexpr size_t fd_order = 10;

  SPECTRE_ALWAYS_INLINE static double pointwise(const double* const /*q*/,
                                                const int /*stride*/,
                                                const double /*epsilon*/) {
    ERROR("Not implemented");
  }
};

template <typename FilterComputer, size_t Dim, typename... Args>
void filter_fastest_dim(const gsl::not_null<gsl::span<double>*> derivative,
                        const gsl::span<const double>& volume_vars,
                        const gsl::span<const double>& lower_ghost_data,
                        const gsl::span<const double>& upper_ghost_data,
                        const Index<Dim>& volume_extents,
                        const size_t number_of_variables, const Args&... args) {
  constexpr size_t fd_order = FilterComputer::fd_order;
  ASSERT(
      fd_order % 2 == 0,
      "The finite difference order with should be even but got " << fd_order);
  constexpr size_t stencil_width = fd_order + 1;
  ASSERT(volume_extents[0] >= stencil_width - 1,
         " Subcell volume extent (current value: "
             << volume_extents[0]
             << ") must be not smaller than the stencil width (current value: "
             << stencil_width << ") minus 1");
  constexpr size_t ghost_zone_for_stencil = fd_order / 2;
  // Compute the number of ghost cells.
  const size_t number_of_stripes =
      volume_extents.slice_away(0).product() * number_of_variables;
  ASSERT(lower_ghost_data.size() % number_of_stripes == 0,
         "The lower ghost data must be a multiple of the number of stripes ("
             << number_of_stripes
             << "), which is defined as the number of variables ("
             << number_of_variables
             << ") times the number of grid points on a 2d slice ("
             << volume_extents.slice_away(0).product() << ")");
  ASSERT(upper_ghost_data.size() == lower_ghost_data.size(),
         "The lower ghost data size ("
             << lower_ghost_data.size()
             << ") must match the upper ghost data size, "
             << upper_ghost_data.size());
  const size_t ghost_pts_in_neighbor_data =
      lower_ghost_data.size() / number_of_stripes;

  std::array<double, stencil_width> q{};
  for (size_t slice = 0; slice < number_of_stripes; ++slice) {
    const size_t vars_slice_offset = slice * volume_extents[0];
    const size_t vars_neighbor_slice_offset =
        slice * ghost_pts_in_neighbor_data;

    // Deal with lower ghost data.
    for (size_t i = 0; i < ghost_zone_for_stencil; ++i) {
      // offset comes from accounting for the 1 extra point in our ghost
      // cells plus how far away from the boundary we are differentiating.
      for (size_t j = 0, offset = vars_neighbor_slice_offset +
                                  ghost_pts_in_neighbor_data -
                                  (ghost_zone_for_stencil - i);
           j < ghost_zone_for_stencil - i; ++j) {
        gsl::at(q, j) = lower_ghost_data[offset + j];
      }
      for (size_t j = ghost_zone_for_stencil - i, k = 0; j < stencil_width;
           ++j, ++k) {
        gsl::at(q, j) = volume_vars[vars_slice_offset + k];
      }
      (*derivative)[vars_slice_offset + i] = FilterComputer::pointwise(
          q.data() + ghost_zone_for_stencil, 1, args...);
    }

    // Differentiate in the bulk
    const size_t slice_end = volume_extents[0] - ghost_zone_for_stencil;
    for (size_t vars_index = vars_slice_offset + ghost_zone_for_stencil,
                i = ghost_zone_for_stencil;
         i < slice_end; ++vars_index, ++i) {
      // Note: we keep the `stride` here because we may want to
      // experiment/support non-unit strides in the bulk in the future. For
      // cells where the derivative needs boundary data we copy into a
      // `std::array` buffer, which means we always have unit stride.
      constexpr int stride = 1;
      (*derivative)[vars_slice_offset + i] =
          FilterComputer::pointwise(&volume_vars[vars_index], stride, args...);
    }

    // Differentiate using upper neighbor data
    for (size_t i = 0; i < ghost_zone_for_stencil; ++i) {
      // offset comes from accounting for the 1 extra point in our ghost
      // cells plus how far away from the boundary we are differentiating.
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
      //  c = points used for derivative

      size_t j = 0;
      for (size_t k =
               vars_slice_offset + volume_extents[0] - (stencil_width - 1 - i);
           j < stencil_width - 1 - i; ++j, ++k) {
        gsl::at(q, j) = volume_vars[k];
      }
      for (size_t k = 0; j < stencil_width; ++j, ++k) {
        gsl::at(q, j) = upper_ghost_data[vars_neighbor_slice_offset + k];
      }

      (*derivative)[vars_slice_offset + slice_end + i] =
          FilterComputer::pointwise(q.data() + ghost_zone_for_stencil, 1,
                                    args...);
    }
  }  // for slices
}

template <typename FilterComputer, size_t Dim, typename... Args>
void filter_impl(
    const gsl::not_null<gsl::span<double>*> filtered_data,
    gsl::span<double>* const in_buffer,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables,
    const Args&... args) {
#ifdef SPECTRE_DEBUG
  ASSERT(volume_mesh == Mesh<Dim>(volume_mesh.extents(0), volume_mesh.basis(0),
                                  volume_mesh.quadrature(0)),
         "The mesh must be isotropic, but got " << volume_mesh);
  ASSERT(
      volume_mesh.basis(0) == Spectral::Basis::FiniteDifference,
      "Mesh basis must be FiniteDifference but got " << volume_mesh.basis(0));
  ASSERT(volume_mesh.quadrature(0) == Spectral::Quadrature::CellCentered,
         "Mesh quadrature must be CellCentered but got "
             << volume_mesh.quadrature(0));
  const size_t number_of_points = volume_mesh.number_of_grid_points();
  ASSERT(volume_vars.size() == number_of_points * number_of_variables,
         "The size of the volume vars must be the number of points ("
             << number_of_points << ") times the number of variables ("
             << number_of_variables << ") but is " << volume_vars.size());
  ASSERT(filtered_data->size() == volume_vars.size(),
         "The logical derivatives must have size " << volume_vars.size()
                                                   << " but has size "
                                                   << filtered_data->size());
#endif  // SPECTRE_DEBUG

  ASSERT(ghost_cell_vars.contains(Direction<Dim>::lower_xi()),
         "Couldn't find lower ghost data in lower-xi");
  ASSERT(ghost_cell_vars.contains(Direction<Dim>::upper_xi()),
         "Couldn't find upper ghost data in upper-xi");
  const Index<Dim>& volume_extents = volume_mesh.extents();

  filter_fastest_dim<FilterComputer>(
      filtered_data, volume_vars,
      ghost_cell_vars.at(Direction<Dim>::lower_xi()),
      ghost_cell_vars.at(Direction<Dim>::upper_xi()), volume_extents,
      number_of_variables, args...);

  if constexpr (Dim > 1) {
    ASSERT(ghost_cell_vars.contains(Direction<Dim>::lower_eta()),
           "Couldn't find lower ghost data in lower-eta");
    ASSERT(ghost_cell_vars.contains(Direction<Dim>::upper_eta()),
           "Couldn't find upper ghost data in upper-eta");

    // We transpose from (x,y,z,vars) ordering to (y,z,vars,x) ordering
    // Might not be the most efficient (unclear), but easiest.
    // We use a single large buffer for both the y and z derivatives
    // to reduce the number of memory allocations and improve data locality.
    const auto& lower_ghost = ghost_cell_vars.at(Direction<Dim>::lower_eta());
    const auto& upper_ghost = ghost_cell_vars.at(Direction<Dim>::upper_eta());
    const size_t filter_size = filtered_data->size();
    DataVector buffer{};
    if (in_buffer != nullptr) {
      ASSERT((in_buffer->size() >= volume_vars.size() + lower_ghost.size() +
                                       upper_ghost.size() + 3 * filter_size),
             "The buffer must have size greater than or equal to "
                 << (volume_vars.size() + lower_ghost.size() +
                     upper_ghost.size() + 3 * filter_size)
                 << " but has size " << in_buffer->size());
      buffer.set_data_ref(in_buffer->data(), in_buffer->size());
    } else {
      buffer = DataVector{volume_vars.size() + lower_ghost.size() +
                          upper_ghost.size() + 3 * filter_size};
    }
    raw_transpose(make_not_null(buffer.data()), volume_vars.data(),
                  volume_extents[0], volume_vars.size() / volume_extents[0]);
    raw_transpose(make_not_null(&buffer[volume_vars.size()]),
                  lower_ghost.data(), volume_extents[0],
                  lower_ghost.size() / volume_extents[0]);
    raw_transpose(
        make_not_null(&buffer[volume_vars.size() + lower_ghost.size()]),
        upper_ghost.data(), volume_extents[0],
        upper_ghost.size() / volume_extents[0]);

    // Note: assumes isotropic extents
    const size_t filter_offset_in_buffer =
        volume_vars.size() + lower_ghost.size() + upper_ghost.size();
    gsl::span<double> filter_view =
        gsl::make_span(&buffer[filter_offset_in_buffer], filter_size);

    filter_fastest_dim<FilterComputer>(
        make_not_null(&filter_view),
        gsl::make_span(buffer.data(), volume_vars.size()),
        gsl::make_span(&buffer[volume_vars.size()], lower_ghost.size()),
        gsl::make_span(&buffer[volume_vars.size() + lower_ghost.size()],
                       upper_ghost.size()),
        volume_extents, number_of_variables, args...);
    // Transpose result back and add to filtered_data
    gsl::span<double> filter_data_in_xyz_order =
        gsl::make_span(&buffer[filter_offset_in_buffer + volume_vars.size()],
                       volume_vars.size());
    raw_transpose(make_not_null(filter_data_in_xyz_order.data()),
                  filter_view.data(), filter_view.size() / volume_extents[0],
                  volume_extents[0]);
    {
      DataVector t0{filter_data_in_xyz_order.data(),
                    filter_data_in_xyz_order.size()};
      DataVector t1{filtered_data->data(), filtered_data->size()};
      t1 += t0;
    }

    if constexpr (Dim > 2) {
      ASSERT(ghost_cell_vars.contains(Direction<Dim>::lower_zeta()),
             "Couldn't find lower ghost data in lower-zeta");
      ASSERT(ghost_cell_vars.contains(Direction<Dim>::upper_zeta()),
             "Couldn't find upper ghost data in upper-zeta");

      const size_t chunk_size = volume_extents[0] * volume_extents[1];
      const size_t number_of_volume_chunks = volume_vars.size() / chunk_size;
      const size_t number_of_neighbor_chunks =
          ghost_cell_vars.at(Direction<Dim>::lower_zeta()).size() / chunk_size;

      raw_transpose(make_not_null(buffer.data()), volume_vars.data(),
                    chunk_size, number_of_volume_chunks);
      raw_transpose(make_not_null(&buffer[volume_vars.size()]),
                    ghost_cell_vars.at(Direction<Dim>::lower_zeta()).data(),
                    chunk_size, number_of_neighbor_chunks);
      raw_transpose(
          make_not_null(&buffer[volume_vars.size() + lower_ghost.size()]),
          ghost_cell_vars.at(Direction<Dim>::upper_zeta()).data(), chunk_size,
          number_of_neighbor_chunks);

      filter_fastest_dim<FilterComputer>(
          make_not_null(&filter_view),
          gsl::make_span(buffer.data(), volume_vars.size()),
          gsl::make_span(&buffer[volume_vars.size()], lower_ghost.size()),
          gsl::make_span(&buffer[volume_vars.size() + lower_ghost.size()],
                         upper_ghost.size()),
          volume_extents, number_of_variables, args...);
      // Transpose result back
      raw_transpose(make_not_null(filter_data_in_xyz_order.data()),
                    filter_view.data(), filter_view.size() / chunk_size,
                    chunk_size);
      {
        DataVector t0{filter_data_in_xyz_order.data(),
                      filter_data_in_xyz_order.size()};
        DataVector t1{filtered_data->data(), filtered_data->size()};
        t1 += t0;
      }
    }
  }
  // Add volume vars to filter
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  const DataVector view_volume_vars{const_cast<double*>(volume_vars.data()),
                                    volume_vars.size()};
  DataVector view_filtered_data{filtered_data->data(), filtered_data->size()};
  view_filtered_data += view_volume_vars;
}

template <template <size_t Order, bool UnitStride> class FilterComputer,
          size_t Dim, typename... Args>
void forward_to_filter_impl(
    const gsl::not_null<gsl::span<double>*> filtered_data,
    gsl::span<double>* const buffer, const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables,
    const size_t fd_order, const Args&... args) {
  switch (fd_order) {
    case 2:
      ::fd::filter_impl<FilterComputer<2, true>>(
          filtered_data, buffer, volume_vars, ghost_cell_vars, volume_mesh,
          number_of_variables, args...);
      break;
    case 4:
      ::fd::filter_impl<FilterComputer<4, true>>(
          filtered_data, buffer, volume_vars, ghost_cell_vars, volume_mesh,
          number_of_variables, args...);
      break;
    case 6:
      ::fd::filter_impl<FilterComputer<6, true>>(
          filtered_data, buffer, volume_vars, ghost_cell_vars, volume_mesh,
          number_of_variables, args...);
      break;
    case 8:
      ::fd::filter_impl<FilterComputer<8, true>>(
          filtered_data, buffer, volume_vars, ghost_cell_vars, volume_mesh,
          number_of_variables, args...);
      break;
    case 10:
      ::fd::filter_impl<FilterComputer<10, true>>(
          filtered_data, buffer, volume_vars, ghost_cell_vars, volume_mesh,
          number_of_variables, args...);
      break;
    default:
      ERROR("Cannot do finite difference filter of order " << fd_order);
  };
}
}  // namespace

template <size_t Dim>
void low_pass_filter(
    const gsl::not_null<gsl::span<double>*> filtered_data,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables,
    const size_t fd_order, const double epsilon) {
  forward_to_filter_impl<LowPassFilterImpl>(
      filtered_data, nullptr, volume_vars, ghost_cell_vars, volume_mesh,
      number_of_variables, fd_order, epsilon);
}

template <size_t Dim>
void kreiss_oliger_filter(
    const gsl::not_null<gsl::span<double>*> filtered_data,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables,
    const size_t fd_order, const double epsilon) {
  forward_to_filter_impl<KoDissipationImpl>(
      filtered_data, nullptr, volume_vars, ghost_cell_vars, volume_mesh,
      number_of_variables, fd_order, epsilon);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template void low_pass_filter(                                               \
      gsl::not_null<gsl::span<double>*> filtered_data,                         \
      const gsl::span<const double>& volume_vars,                              \
      const DirectionMap<DIM(data), gsl::span<const double>>& ghost_cell_vars, \
      const Mesh<DIM(data)>& volume_mesh, size_t number_of_variables,          \
      size_t fd_order, double epsilon);                                        \
  template void kreiss_oliger_filter(                                          \
      gsl::not_null<gsl::span<double>*> filtered_data,                         \
      const gsl::span<const double>& volume_vars,                              \
      const DirectionMap<DIM(data), gsl::span<const double>>& ghost_cell_vars, \
      const Mesh<DIM(data)>& volume_mesh, size_t number_of_variables,          \
      size_t fd_order, double epsilon);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace fd
