// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace TestHelpers::fd::reconstruction {
/*!
 * \brief Compare the output of reconstruction done once in python and once in
 * C++.
 *
 * - `extents` The extents are the volume extents of the subcell grid in a
 * DG-subcell scheme. In general, this should be the size of the reconstruction
 * stencil plus one or larger to provide a good test.
 * - `stencil_width` the width of the reconstruction stencil. For example, for
 * minmod or MC this would be 3, for 5-point 5th-order WENO this would
 * be 5, and for three three-point stencil CWENO this would be 5 as well.
 * -`python_test_file` the file in which the python function to compare the
 * output result is defined in.
 * - `python_function` the python function whose output will be compared to
 * - `recons_function` an invokable (usually a lambda) that has the following
 * signature:
 * \code{.cpp}
 *  (const gsl::not_null<std::array<gsl::span<double>, Dim>*>
 *              reconstructed_upper_side_of_face_vars,
 *  const gsl::not_null<std::array<gsl::span<double>, Dim>*>
 *              reconstructed_lower_side_of_face_vars,
 *  const gsl::span<const double>& volume_vars,
 *  const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
 *  const Index<Dim>& volume_extents, const size_t number_of_variables)
 * \endcode
 */
template <size_t Dim, typename ReconsFunction, typename F1>
void test_with_python(const Index<Dim>& extents, const size_t stencil_width,
                      const std::string& python_test_file,
                      const std::string& python_function,
                      const ReconsFunction& recons_function,
                      const F1& invoke_reconstruct_neighbor) {
  CAPTURE(extents);
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);

  const size_t ghost_zone_size = (stencil_width - 1) / 2;
  const size_t ghost_cells_from_neighbor = ghost_zone_size + 1;
  Index<Dim> extents_with_ghosts = extents;
  for (size_t i = 0; i < Dim; ++i) {
    // 2 because we extend in both upper and lower direction.
    extents_with_ghosts[i] += 2 * ghost_cells_from_neighbor;
  }
  const size_t number_of_vars = 1;
  std::vector<double> vars_with_ghosts(extents_with_ghosts.product() *
                                       number_of_vars);
  fill_with_random_values(make_not_null(&vars_with_ghosts), make_not_null(&gen),
                          make_not_null(&dist));
  // Copy out interior and neighbor data since that's what we will be passing
  // to the C++ implementation.
  const auto in_volume = [ghost_cells_from_neighbor, extents](
                             const auto& idx, const size_t dim) {
    return idx >= ghost_cells_from_neighbor and
           idx < extents[dim] + ghost_cells_from_neighbor;
  };
  const auto in_upper_neighbor = [ghost_cells_from_neighbor, extents](
                                     const auto& idx, const size_t dim) {
    return idx >= extents[dim] + ghost_cells_from_neighbor;
  };
  const auto in_lower_neighbor = [ghost_cells_from_neighbor](const auto& idx) {
    return idx < ghost_cells_from_neighbor;
  };

  DirectionMap<Dim, std::vector<double>> ghost_cells{};
  for (const auto& direction : Direction<Dim>::all_directions()) {
    ghost_cells[direction] = std::vector<double>{};
  }
  std::vector<double> volume_vars(extents.product() * number_of_vars);
  for (size_t k = 0; k < (Dim > 2 ? extents_with_ghosts[2] : 1); ++k) {
    for (size_t j = 0; j < (Dim > 1 ? extents_with_ghosts[1] : 1); ++j) {
      for (size_t i = 0; i < extents_with_ghosts[0]; ++i) {
        if constexpr (Dim > 1) {
          // Check if we are in a Voronoi neighbor (ignore then)
          std::vector<bool> has_neighbor(Dim);
          has_neighbor.at(0) = in_lower_neighbor(i) or in_upper_neighbor(i, 0);
          has_neighbor.at(1) = in_lower_neighbor(j) or in_upper_neighbor(j, 1);
          if constexpr (Dim > 2) {
            has_neighbor.at(2) =
                in_lower_neighbor(k) or in_upper_neighbor(k, 2);
          }
          if (std::count(has_neighbor.begin(), has_neighbor.end(), true) >= 2) {
            continue;
          }
        }

        std::array volume_with_ghost_indices{i, j, k};
        Index<Dim> volume_with_ghosts_index{};
        for (size_t d = 0; d < Dim; ++d) {
          volume_with_ghosts_index[d] = gsl::at(volume_with_ghost_indices, d);
        }

        // If in volume, copy over...
        if (in_volume(i, 0) and
            (Dim == 1 or (in_volume(j, 1) and (Dim == 2 or in_volume(k, 2))))) {
          Index<Dim> volume_index{};
          for (size_t d = 0; d < Dim; ++d) {
            volume_index[d] = gsl::at(volume_with_ghost_indices, d) -
                              (extents_with_ghosts[d] - extents[d]) / 2;
          }

          volume_vars[collapsed_index(volume_index, extents)] =
              vars_with_ghosts[collapsed_index(volume_with_ghosts_index,
                                               extents_with_ghosts)];
        } else {
          // We are dealing with something in the neighbor data. We need to
          // identify which neighbor and then set.
          Side side = Side::Lower;
          size_t dimension = 0;
          if (in_upper_neighbor(i, 0)) {
            side = Side::Upper;
          }
          if (Dim > 1 and in_lower_neighbor(j)) {
            dimension = 1;
          }
          if (Dim > 1 and in_upper_neighbor(j, 1)) {
            side = Side::Upper;
            dimension = 1;
          }
          if (Dim > 2 and in_lower_neighbor(k)) {
            dimension = 2;
          }
          if (Dim > 2 and in_upper_neighbor(k, 2)) {
            side = Side::Upper;
            dimension = 2;
          }

          const Direction<Dim> direction{dimension, side};
          ghost_cells.at(direction).push_back(vars_with_ghosts[collapsed_index(
              volume_with_ghosts_index, extents_with_ghosts)]);
        }
      }
    }
  }

  const size_t reconstructed_num_pts =
      (extents[0] + 1) * extents.slice_away(0).product();
  std::array<std::vector<double>, Dim> reconstructed_upper_side_of_face_vars =
      make_array<Dim>(
          std::vector<double>(reconstructed_num_pts * number_of_vars));
  std::array<std::vector<double>, Dim> reconstructed_lower_side_of_face_vars =
      make_array<Dim>(
          std::vector<double>(reconstructed_num_pts * number_of_vars));

  std::array<gsl::span<double>, Dim> recons_upper_side_of_face{};
  std::array<gsl::span<double>, Dim> recons_lower_side_of_face{};
  for (size_t i = 0; i < Dim; ++i) {
    auto& upper = gsl::at(reconstructed_upper_side_of_face_vars, i);
    auto& lower = gsl::at(reconstructed_lower_side_of_face_vars, i);
    gsl::at(recons_upper_side_of_face, i) =
        gsl::make_span(upper.data(), upper.size());
    gsl::at(recons_lower_side_of_face, i) =
        gsl::make_span(lower.data(), lower.size());
  }

  DirectionMap<Dim, gsl::span<const double>> ghost_cell_vars{};
  for (const auto& [direction, data] : ghost_cells) {
    ghost_cell_vars[direction] = gsl::make_span(data.data(), data.size());
  }

  recons_function(make_not_null(&recons_upper_side_of_face),
                  make_not_null(&recons_lower_side_of_face),
                  gsl::make_span(volume_vars), ghost_cell_vars, extents,
                  number_of_vars);

  const std::vector<size_t> extents_with_ghosts_vector(
      extents_with_ghosts.begin(), extents_with_ghosts.end());
  const std::vector<size_t> ghost_zones(Dim, ghost_zone_size);
  for (size_t var_index = 0; var_index < number_of_vars; ++var_index) {
    CAPTURE(var_index);
    const std::vector<double> var(
        vars_with_ghosts.begin() +
            static_cast<std::ptrdiff_t>(var_index *
                                        extents_with_ghosts.product()),
        vars_with_ghosts.begin() +
            static_cast<std::ptrdiff_t>((var_index + 1) *
                                        extents_with_ghosts.product()));
    const auto result =
        pypp::call<std::vector<std::vector<std::vector<double>>>>(
            python_test_file, python_function, var, extents_with_ghosts_vector,
            Dim);

    const std::vector<std::vector<double>>& python_recons_on_lower = result[0];
    const std::vector<std::vector<double>>& python_recons_on_upper = result[1];
    for (size_t d = 0; d < Dim; ++d) {
      CAPTURE(d);
      const std::vector<double> recons_upper_side_this_var(
          gsl::at(recons_upper_side_of_face, d).begin() +
              var_index * reconstructed_num_pts,
          gsl::at(recons_upper_side_of_face, d).begin() +
              (var_index + 1) * reconstructed_num_pts);
      CHECK_ITERABLE_APPROX(recons_upper_side_this_var,
                            python_recons_on_upper[d]);

      const std::vector<double> recons_lower_side_this_var(
          gsl::at(recons_lower_side_of_face, d).begin() +
              var_index * reconstructed_num_pts,
          gsl::at(recons_lower_side_of_face, d).begin() +
              (var_index + 1) * reconstructed_num_pts);
      CHECK_ITERABLE_APPROX(recons_lower_side_this_var,
                            python_recons_on_lower[d]);

      // Test fd::reconstruction::reconstruct_neighbor
      Index<Dim> ghost_data_extents = extents;
      ghost_data_extents[d] = (stencil_width + 1) / 2;
      Index<Dim> extents_with_faces = extents;
      ++extents_with_faces[d];
      const DataVector dv_var{
          const_cast<double*>(volume_vars.data() +
                              var_index * extents.product()),
          extents.product()};

      const Direction<Dim> upper_direction{d, Side::Upper};
      DataVector upper_face_var{extents.slice_away(d).product()};
      auto& upper_neighbor_data = ghost_cells.at(upper_direction);
      DataVector upper_neighbor_var{
          // NOLINTNEXTLINE
          upper_neighbor_data.data() + var_index * ghost_data_extents.product(),
          ghost_data_extents.product()};
      invoke_reconstruct_neighbor(make_not_null(&upper_face_var), dv_var,
                                  upper_neighbor_var, extents,
                                  ghost_data_extents, upper_direction);
      for (SliceIterator si(extents_with_faces, d, extents_with_faces[d] - 1);
           si; ++si) {
        CHECK(approx(upper_face_var[si.slice_offset()]) ==
              recons_upper_side_this_var[si.volume_offset()]);
      }

      const Direction<Dim> lower_direction{d, Side::Lower};
      DataVector lower_face_var{extents.slice_away(d).product()};
      auto& lower_neighbor_data = ghost_cells.at(lower_direction);
      DataVector lower_neighbor_var{
          // NOLINTNEXTLINE
          lower_neighbor_data.data() + var_index * ghost_data_extents.product(),
          ghost_data_extents.product()};
      invoke_reconstruct_neighbor(make_not_null(&lower_face_var), dv_var,
                                  lower_neighbor_var, extents,
                                  ghost_data_extents, lower_direction);
      for (SliceIterator si(extents_with_faces, d, 0); si; ++si) {
        CHECK(approx(lower_face_var[si.slice_offset()]) ==
              recons_lower_side_this_var[si.volume_offset()]);
      }
    }
  }
}
}  // namespace TestHelpers::fd::reconstruction
