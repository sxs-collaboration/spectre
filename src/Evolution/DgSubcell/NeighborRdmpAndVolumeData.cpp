// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/NeighborRdmpAndVolumeData.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <optional>
#include <utility>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Matrices.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MemoryHelpers.hpp"

namespace evolution::dg::subcell::detail {
namespace {
// Projects ghost data for a ZernikeB1 neighbor using parity-split projection
// matrices
void project_zernike_b1_ghost_data(
    const gsl::not_null<DataVector*> computed_ghost_data,
    const DataVector& neighbor_data_without_rdmp_vars,
    const size_t number_of_vars, const Mesh<3>& neighbor_mesh,
    const Mesh<3>& subcell_mesh, const size_t number_of_ghost_zones,
    const Direction<3>& direction, const gsl::span<const size_t> parity_list,
    const size_t num_even, const size_t num_odd) {
  ASSERT(neighbor_mesh.basis(0) == Spectral::Basis::ZernikeB1 and
             direction.dimension() != 0,
         "Neighbor setup is not appropriate to call ZernikeB1-specific "
         "projection, got neighbor mesh = "
             << neighbor_mesh << ", direction = " << direction);
  const size_t num_dg_pts = neighbor_mesh.number_of_grid_points();
  const size_t num_ghost_pts_per_var =
      number_of_ghost_zones *
      subcell_mesh.extents().slice_away(direction.dimension()).product();
  if (UNLIKELY(computed_ghost_data->size() !=
               num_ghost_pts_per_var * number_of_vars)) {
    computed_ghost_data->destructive_resize(num_ghost_pts_per_var *
                                            number_of_vars);
  }

  auto buffer =
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      cpp20::make_unique_for_overwrite<double[]>(
          (num_even + num_odd) * (num_dg_pts + num_ghost_pts_per_var));
  DataVector even_input{};
  even_input.set_data_ref(&buffer[0], num_even * num_dg_pts);
  DataVector odd_input{};
  odd_input.set_data_ref(&buffer[num_even * num_dg_pts], num_odd * num_dg_pts);
  DataVector even_output{};
  even_input.set_data_ref(&buffer[(num_even + num_odd) * num_dg_pts],
                          num_even * num_ghost_pts_per_var);
  DataVector odd_output{};
  odd_input.set_data_ref(&buffer[(num_even + num_odd) * num_dg_pts +
                                 num_even * num_ghost_pts_per_var],
                         num_odd * num_ghost_pts_per_var);

  // Sort input components into even/odd parity buffers
  const double* p_in = neighbor_data_without_rdmp_vars.data();
  double* p_even_in = even_input.data();
  double* p_odd_in = odd_input.data();
  bool is_even = true;
  for (const size_t seg_size : parity_list) {
    if (seg_size == 0) {
      if (is_even) {
        is_even = false;
        continue;
      } else {
        break;
      }
    }
    if (is_even) {
      std::copy(p_in, p_in + seg_size * num_dg_pts, p_even_in);  // NOLINT
      p_even_in += seg_size * num_dg_pts;                        // NOLINT
    } else {
      std::copy(p_in, p_in + seg_size * num_dg_pts, p_odd_in);  // NOLINT
      p_odd_in += seg_size * num_dg_pts;                        // NOLINT
    }
    p_in += seg_size * num_dg_pts;  // NOLINT
    is_even = not is_even;
  }

  // Build matrix arrays: dim 0 gets parity-dependent ZernikeB1 matrices;
  // other dims are shared between even and odd batches.
  const Matrix empty{};
  auto even_ghost_mat = make_array<3>(std::cref(empty));
  auto odd_ghost_mat = make_array<3>(std::cref(empty));
  even_ghost_mat[0] = std::cref(fd::projection_matrix(
      neighbor_mesh.slice_through(0), subcell_mesh.extents(0),
      Spectral::Quadrature::CellCentered, Spectral::Parity::Even));
  odd_ghost_mat[0] = std::cref(fd::projection_matrix(
      neighbor_mesh.slice_through(0), subcell_mesh.extents(0),
      Spectral::Quadrature::CellCentered, Spectral::Parity::Odd));
  for (size_t i = 1; i < 3; ++i) {
    if (i == direction.dimension()) {
      const auto& ghost_mat = fd::projection_matrix(
          neighbor_mesh.slice_through(i), subcell_mesh.extents(i),
          number_of_ghost_zones, direction.opposite().side());
      gsl::at(even_ghost_mat, i) = std::cref(ghost_mat);
      gsl::at(odd_ghost_mat, i) = std::cref(ghost_mat);
    } else {
      const auto& other_mat = fd::projection_matrix(
          neighbor_mesh.slice_through(i), subcell_mesh.extents(i),
          Spectral::Quadrature::CellCentered, Spectral::Parity::Uninitialized);
      gsl::at(even_ghost_mat, i) = std::cref(other_mat);
      gsl::at(odd_ghost_mat, i) = std::cref(other_mat);
    }
  }

  if (num_even > 0) {
    apply_matrices(make_not_null(&even_output), even_ghost_mat, even_input,
                   neighbor_mesh.extents());
  }
  if (num_odd > 0) {
    apply_matrices(make_not_null(&odd_output), odd_ghost_mat, odd_input,
                   neighbor_mesh.extents());
  }

  // Reassemble output in original component order
  double* p_out = computed_ghost_data->data();
  const double* p_even_out = even_output.data();
  const double* p_odd_out = odd_output.data();
  is_even = true;
  for (const size_t seg_size : parity_list) {
    if (seg_size == 0) {
      if (is_even) {
        is_even = false;
        continue;
      } else {
        break;
      }
    }
    if (is_even) {
      std::copy(p_even_out,
                p_even_out + seg_size * num_ghost_pts_per_var,  // NOLINT
                p_out);
      p_even_out += seg_size * num_ghost_pts_per_var;  // NOLINT
    } else {
      std::copy(p_odd_out,
                p_odd_out + seg_size * num_ghost_pts_per_var,  // NOLINT
                p_out);
      p_odd_out += seg_size * num_ghost_pts_per_var;  // NOLINT
    }
    p_out += seg_size * num_ghost_pts_per_var;  // NOLINT
    is_even = not is_even;
  }
}
}  // namespace

template <bool InsertIntoMap, size_t Dim>
void insert_or_update_neighbor_volume_data_impl(
    const gsl::not_null<DirectionalIdMap<Dim, GhostData>*> ghost_data_ptr,
    const DataVector& neighbor_subcell_data,
    const size_t number_of_rdmp_vars_in_buffer,
    const DirectionalId<Dim>& directional_element_id,
    const Mesh<Dim>& neighbor_mesh, const Element<Dim>& element,
    const Mesh<Dim>& subcell_mesh, const size_t number_of_ghost_zones,
    const DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>&
        neighbor_dg_to_fd_interpolants,
    const gsl::span<const size_t> parity_list, const size_t num_even,
    const size_t num_odd) {
  fd::verify_subcell_mesh(neighbor_mesh, true);
  ASSERT(neighbor_subcell_data.size() != 0,
         "neighbor_subcell_data must be non-empty");
  const size_t end_of_volume_data =
      neighbor_subcell_data.size() - 2 * number_of_rdmp_vars_in_buffer;

  if constexpr (InsertIntoMap) {
    (*ghost_data_ptr)[directional_element_id] = GhostData{1};
  }

  DataVector& ghost_data = (*ghost_data_ptr)[directional_element_id]
                               .neighbor_ghost_data_for_reconstruction();
  DataVector computed_ghost_data{};
  if (neighbor_mesh.basis(0) == Spectral::Basis::FiniteDifference) {
    ASSERT(neighbor_mesh == subcell_mesh,
           "Neighbor mesh ("
               << neighbor_mesh << ") and my mesh (" << subcell_mesh
               << ") must be the same if we are both doing subcell.");
    if (not InsertIntoMap and
        neighbor_subcell_data.data() == ghost_data.data()) {
      // Short-circuit if we are already doing FD and we would be
      // self-assigning, so elide copy and move.
      return;
    }
    // Copy over the ghost cell data for subcell reconstruction. In this case
    // the neighbor would have reoriented/interpolated the data for us.
    computed_ghost_data.destructive_resize(end_of_volume_data);
    std::copy(
        neighbor_subcell_data.begin(),
        std::prev(neighbor_subcell_data.end(),
                  2 * static_cast<typename std::iterator_traits<
                          typename DataVector::iterator>::difference_type>(
                          number_of_rdmp_vars_in_buffer)),
        computed_ghost_data.begin());
  } else {
    ASSERT(fd::mesh(neighbor_mesh) == subcell_mesh,
           "Neighbor subcell mesh computed from the neighbor DG mesh ("
               << fd::mesh(neighbor_mesh) << ") and my mesh (" << subcell_mesh
               << ") must be the same.");
    const Direction<Dim>& direction = directional_element_id.direction();
    const size_t total_number_of_ghost_zones =
        number_of_ghost_zones *
        subcell_mesh.extents().slice_away(direction.dimension()).product();
    ASSERT(
        end_of_volume_data % neighbor_mesh.number_of_grid_points() == 0,
        "The number of DG volume grid points times the number of variables "
        "sent for reconstruction ("
            << end_of_volume_data
            << ") must be a multiple of the number of DG volume grid points: "
            << neighbor_mesh.number_of_grid_points() << " number of RDMP vars "
            << number_of_rdmp_vars_in_buffer << "\nThis element: " << element
            << "\nDirection: " << direction);
    const size_t number_of_vars =
        end_of_volume_data / neighbor_mesh.number_of_grid_points();

    computed_ghost_data.destructive_resize(total_number_of_ghost_zones *
                                           number_of_vars);
    if (const auto iter =
            neighbor_dg_to_fd_interpolants.find(directional_element_id);
        iter != neighbor_dg_to_fd_interpolants.end() and
        iter->second.has_value()) {
      // Our neighbor is in a different Block, so we assume a different map
      // and need to interpolate to our neighbors' ghost zone coordinates.
      gsl::span<double> result{computed_ghost_data.data(),
                               computed_ghost_data.size()};
      iter->second.value().interpolate(
          make_not_null(&result),
          gsl::span<const double>{neighbor_subcell_data.data(),
                                  end_of_volume_data});
    } else {
      // If our neighbor is in our block we can do simple dim-by-dim
      // interpolation.
      const DataVector neighbor_data_without_rdmp_vars{
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<double*>(neighbor_subcell_data.data()),
          end_of_volume_data};

      if constexpr (Dim == 3) {
        if (neighbor_mesh.basis(0) == Spectral::Basis::ZernikeB1 and
            direction.dimension() != 0) {
          // ZernikeB1 is in dimension 0 and is not the ghost direction, so
          // we need to project even- and odd-parity components separately.
          project_zernike_b1_ghost_data(
              make_not_null(&computed_ghost_data),
              neighbor_data_without_rdmp_vars, number_of_vars, neighbor_mesh,
              subcell_mesh, number_of_ghost_zones, direction, parity_list,
              num_even, num_odd);
          ghost_data = std::move(computed_ghost_data);
          return;
        }
      }

      const Matrix empty{};
      auto ghost_projection_mat = make_array<Dim>(std::cref(empty));
      for (size_t i = 0; i < Dim; ++i) {
        if (i == direction.dimension()) {
          gsl::at(ghost_projection_mat, i) = std::cref(fd::projection_matrix(
              neighbor_mesh.slice_through(i), subcell_mesh.extents(i),
              number_of_ghost_zones, direction.opposite().side()));
        } else {
          gsl::at(ghost_projection_mat, i) = std::cref(fd::projection_matrix(
              neighbor_mesh.slice_through(i), subcell_mesh.extents(i),
              Spectral::Quadrature::CellCentered,
              Spectral::Parity::Uninitialized));
        }
      }
      apply_matrices(make_not_null(&computed_ghost_data), ghost_projection_mat,
                     neighbor_data_without_rdmp_vars, neighbor_mesh.extents());
    }
  }

  ghost_data = std::move(computed_ghost_data);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_INSERT(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data)                                                \
  template void insert_or_update_neighbor_volume_data_impl<GET_INSERT(data)>( \
      gsl::not_null<DirectionalIdMap<GET_DIM(data), GhostData>*>              \
          ghost_data_ptr,                                                     \
      const DataVector& neighbor_subcell_data,                                \
      size_t number_of_rdmp_vars_in_buffer,                                   \
      const DirectionalId<GET_DIM(data)>& directional_element_id,             \
      const Mesh<GET_DIM(data)>& neighbor_mesh,                               \
      const Element<GET_DIM(data)>& element,                                  \
      const Mesh<GET_DIM(data)>& subcell_mesh, size_t number_of_ghost_zones,  \
      const DirectionalIdMap<GET_DIM(data),                                   \
                             std::optional<intrp::Irregular<GET_DIM(data)>>>& \
          neighbor_dg_to_fd_interpolants,                                     \
      gsl::span<const size_t> parity_list, size_t num_even, size_t num_odd);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (true, false))

#undef INSTANTIATION
#undef GET_INSERT
#undef GET_DIM
}  // namespace evolution::dg::subcell::detail
