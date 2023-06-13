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

namespace fd {
namespace {
template <size_t Order, bool UnitStride>
struct ComputeImpl;

template <bool UnitStride>
struct ComputeImpl<2, UnitStride> {
  static constexpr size_t fd_order = 2;

  SPECTRE_ALWAYS_INLINE static double pointwise(
      const double* const q, const int stride,
      const std::array<double, 1>& weights) {
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[0] * (q[1] - q[-1]);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[0] * (q[stride] - q[-stride]);
    }
  }

  static constexpr std::array<double, 1> derivative_weights(
      const double one_over_delta) {
    return {{0.5 * one_over_delta}};
  }
};

template <bool UnitStride>
struct ComputeImpl<4, UnitStride> {
  static constexpr size_t fd_order = 4;

  SPECTRE_ALWAYS_INLINE static double pointwise(
      const double* const q, const int stride,
      const std::array<double, 2>& weights) {
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[1] * (q[-2] - q[2]) + weights[0] * (q[1] - q[-1]);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[1] * (q[-2 * stride] - q[2 * stride]) +
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[0] * (q[stride] - q[-stride]);
    }
  }

  static constexpr std::array<double, 2> derivative_weights(
      const double one_over_delta) {
    return {{0.6666666666666666 * one_over_delta,
             0.08333333333333333 * one_over_delta}};
  }
};

template <bool UnitStride>
struct ComputeImpl<6, UnitStride> {
  static constexpr size_t fd_order = 6;

  SPECTRE_ALWAYS_INLINE static double pointwise(
      const double* const q, const int stride,
      const std::array<double, 3>& weights) {
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[2] * (q[3] - q[-3]) - weights[1] * (q[2] - q[-2]) +
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[0] * (q[1] - q[-1]);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[2] * (q[3 * stride] - q[-3 * stride]) -
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[1] * (q[2 * stride] - q[-2 * stride]) +
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[0] * (q[stride] - q[-stride]);
    }
  }

  static constexpr std::array<double, 3> derivative_weights(
      const double one_over_delta) {
    return {{0.75 * one_over_delta, 0.15 * one_over_delta,
             0.016666666666666666 * one_over_delta}};
  }
};

template <bool UnitStride>
struct ComputeImpl<8, UnitStride> {
  static constexpr size_t fd_order = 8;

  SPECTRE_ALWAYS_INLINE static double pointwise(
      const double* const q, const int stride,
      const std::array<double, 4>& weights) {
    if constexpr (UnitStride) {
      ASSERT(stride == 1, "UnitStride is true but got stride " << stride);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[3] * (q[4] - q[-4]) + weights[2] * (q[3] - q[-3]) -
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[1] * (q[2] - q[-2]) + weights[0] * (q[1] - q[-1]);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      return weights[3] * (q[4 * stride] - q[-4 * stride]) +
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[2] * (q[3 * stride] - q[-3 * stride]) -
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[1] * (q[2 * stride] - q[-2 * stride]) +
             // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
             weights[0] * (q[stride] - q[-stride]);
    }
  }

  static constexpr std::array<double, 4> derivative_weights(
      const double one_over_delta) {
    return {{0.8 * one_over_delta, 0.2 * one_over_delta,
             0.0380952380952381 * one_over_delta,
             -0.0035714285714285713 * one_over_delta}};
  }
};

template <typename DerivativeComputer, size_t Dim>
void logical_partial_derivatives_fastest_dim(
    const gsl::not_null<gsl::span<double>*> derivative,
    const gsl::span<const double>& volume_vars,
    const gsl::span<const double>& lower_ghost_data,
    const gsl::span<const double>& upper_ghost_data,
    const Index<Dim>& volume_extents, const size_t number_of_variables,
    const double delta) {
  constexpr size_t fd_order = DerivativeComputer::fd_order;
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

  // Precompute derivative weights to minimize FLOPs
  const std::array<double, fd_order / 2> derivative_weights =
      DerivativeComputer::derivative_weights(1.0 / delta);

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
      (*derivative)[vars_slice_offset + i] = DerivativeComputer::pointwise(
          q.data() + ghost_zone_for_stencil, 1, derivative_weights);
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
      (*derivative)[vars_slice_offset + i] = DerivativeComputer::pointwise(
          &volume_vars[vars_index], stride, derivative_weights);
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
          DerivativeComputer::pointwise(q.data() + ghost_zone_for_stencil, 1,
                                        derivative_weights);
    }
  }  // for slices
}

template <typename DerivativeComputer, size_t Dim>
void logical_partial_derivatives_impl(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        logical_derivatives,
    gsl::span<double>* const in_buffer,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables) {
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
  for (size_t i = 0; i < Dim; ++i) {
    ASSERT(gsl::at(*logical_derivatives, i).size() == volume_vars.size(),
           "The logical derivatives must have size "
               << volume_vars.size() << " but has size "
               << gsl::at(*logical_derivatives, i).size() << " in dimension "
               << i);
  }
#endif  // SPECTRE_DEBUG

  ASSERT(ghost_cell_vars.contains(Direction<Dim>::lower_xi()),
         "Couldn't find lower ghost data in lower-xi");
  ASSERT(ghost_cell_vars.contains(Direction<Dim>::upper_xi()),
         "Couldn't find upper ghost data in upper-xi");
  const Index<Dim>& volume_extents = volume_mesh.extents();

  const auto& logical_xi_coords =
      Spectral::collocation_points<Spectral::Basis::FiniteDifference,
                                   Spectral::Quadrature::CellCentered>(
          volume_mesh.extents(0));
  logical_partial_derivatives_fastest_dim<DerivativeComputer>(
      make_not_null(&(*logical_derivatives)[0]), volume_vars,
      ghost_cell_vars.at(Direction<Dim>::lower_xi()),
      ghost_cell_vars.at(Direction<Dim>::upper_xi()), volume_extents,
      number_of_variables, logical_xi_coords[1] - logical_xi_coords[0]);

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
    const size_t derivative_size = (*logical_derivatives)[1].size();
    DataVector buffer{};
    if (in_buffer != nullptr) {
      ASSERT((in_buffer->size() >= volume_vars.size() + lower_ghost.size() +
                                       upper_ghost.size() + derivative_size),
             "The buffer must have size greater than or equal to "
                 << (volume_vars.size() + lower_ghost.size() +
                     upper_ghost.size() + derivative_size)
                 << " but has size " << in_buffer->size());
      buffer.set_data_ref(in_buffer->data(), in_buffer->size());
    } else {
      buffer = DataVector{volume_vars.size() + lower_ghost.size() +
                          upper_ghost.size() + derivative_size};
    }
    raw_transpose(make_not_null(&buffer[0]), volume_vars.data(),
                  volume_extents[0], volume_vars.size() / volume_extents[0]);
    raw_transpose(make_not_null(&buffer[volume_vars.size()]),
                  lower_ghost.data(), volume_extents[0],
                  lower_ghost.size() / volume_extents[0]);
    raw_transpose(
        make_not_null(&buffer[volume_vars.size() + lower_ghost.size()]),
        upper_ghost.data(), volume_extents[0],
        upper_ghost.size() / volume_extents[0]);

    // Note: assumes isotropic extents
    const size_t derivative_offset_in_buffer =
        volume_vars.size() + lower_ghost.size() + upper_ghost.size();
    gsl::span<double> derivative_view =
        gsl::make_span(&buffer[derivative_offset_in_buffer], derivative_size);

    const auto& logical_eta_coords =
        Spectral::collocation_points<Spectral::Basis::FiniteDifference,
                                     Spectral::Quadrature::CellCentered>(
            volume_mesh.extents(1));

    logical_partial_derivatives_fastest_dim<DerivativeComputer>(
        make_not_null(&derivative_view),
        gsl::make_span(&buffer[0], volume_vars.size()),
        gsl::make_span(&buffer[volume_vars.size()], lower_ghost.size()),
        gsl::make_span(&buffer[volume_vars.size() + lower_ghost.size()],
                       upper_ghost.size()),
        volume_extents, number_of_variables,
        logical_eta_coords[1] - logical_eta_coords[0]);
    // Transpose result back
    raw_transpose(
        make_not_null((*logical_derivatives)[1].data()), derivative_view.data(),
        derivative_view.size() / volume_extents[0], volume_extents[0]);

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

      const auto& logical_zeta_coords =
          Spectral::collocation_points<Spectral::Basis::FiniteDifference,
                                       Spectral::Quadrature::CellCentered>(
              volume_mesh.extents(2));

      logical_partial_derivatives_fastest_dim<DerivativeComputer>(
          make_not_null(&derivative_view),
          gsl::make_span(&buffer[0], volume_vars.size()),
          gsl::make_span(&buffer[volume_vars.size()], lower_ghost.size()),
          gsl::make_span(&buffer[volume_vars.size() + lower_ghost.size()],
                         upper_ghost.size()),
          volume_extents, number_of_variables,
          logical_zeta_coords[1] - logical_zeta_coords[0]);
      // Transpose result back
      raw_transpose(make_not_null((*logical_derivatives)[2].data()),
                    derivative_view.data(), derivative_view.size() / chunk_size,
                    chunk_size);
    }
  }
}
}  // namespace

namespace detail {
template <size_t Dim>
void logical_partial_derivatives_impl(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        logical_derivatives,
    gsl::span<double>* const buffer, const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables,
    const size_t fd_order) {
  switch (fd_order) {
    case 2:
      ::fd::logical_partial_derivatives_impl<ComputeImpl<2, true>>(
          logical_derivatives, buffer, volume_vars, ghost_cell_vars,
          volume_mesh, number_of_variables);
      break;
    case 4:
      ::fd::logical_partial_derivatives_impl<ComputeImpl<4, true>>(
          logical_derivatives, buffer, volume_vars, ghost_cell_vars,
          volume_mesh, number_of_variables);
      break;
    case 6:
      ::fd::logical_partial_derivatives_impl<ComputeImpl<6, true>>(
          logical_derivatives, buffer, volume_vars, ghost_cell_vars,
          volume_mesh, number_of_variables);
      break;
    case 8:
      ::fd::logical_partial_derivatives_impl<ComputeImpl<8, true>>(
          logical_derivatives, buffer, volume_vars, ghost_cell_vars,
          volume_mesh, number_of_variables);
      break;
    default:
      ERROR("Cannot do finite difference derivative of order " << fd_order);
  };
}
}  // namespace detail

template <size_t Dim>
void logical_partial_derivatives(
    const gsl::not_null<std::array<gsl::span<double>, Dim>*>
        logical_derivatives,
    const gsl::span<const double>& volume_vars,
    const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
    const Mesh<Dim>& volume_mesh, const size_t number_of_variables,
    const size_t fd_order) {
  detail::logical_partial_derivatives_impl(
      logical_derivatives, nullptr, volume_vars, ghost_cell_vars, volume_mesh,
      number_of_variables, fd_order);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template void detail::logical_partial_derivatives_impl(                      \
      gsl::not_null<std::array<gsl::span<double>, DIM(data)>*> derivative,     \
      gsl::span<double>* const buffer,                                         \
      const gsl::span<const double>& volume_vars,                              \
      const DirectionMap<DIM(data), gsl::span<const double>>& ghost_cell_vars, \
      const Mesh<DIM(data)>& volume_fd_mesh, size_t number_of_variables,       \
      size_t fd_order);                                                        \
  template void logical_partial_derivatives(                                   \
      gsl::not_null<std::array<gsl::span<double>, DIM(data)>*> derivative,     \
      const gsl::span<const double>& volume_vars,                              \
      const DirectionMap<DIM(data), gsl::span<const double>>& ghost_cell_vars, \
      const Mesh<DIM(data)>& volume_fd_mesh, size_t number_of_variables,       \
      size_t fd_order);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace fd
