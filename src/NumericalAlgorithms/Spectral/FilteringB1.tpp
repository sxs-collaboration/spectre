// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/Spectral/FilteringB1.hpp"

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ParityFromSymmetry.hpp"
#include "Utilities/MemoryHelpers.hpp"

namespace Spectral::filtering {
template <typename VariablesTags>
void zernike_b1_exponential_filter(gsl::not_null<Variables<VariablesTags>*> u,
                                   const Mesh<3>& mesh, const double alpha,
                                   const unsigned half_power) {
  const Matrix empty{};
  std::array<Matrix, 3> filter_even = make_array<3>(empty);
  std::array<Matrix, 3> filter_odd = make_array<3>(empty);

  for (size_t d = 0; d < 3; d++) {
    gsl::at(filter_even, d) = Spectral::filtering::exponential_filter(
        mesh.slice_through(d), alpha, half_power, Spectral::Parity::Even);
    gsl::at(filter_odd, d) = Spectral::filtering::exponential_filter(
        mesh.slice_through(d), alpha, half_power, Spectral::Parity::Odd);
  }

  double* p_data = u->data();

  constexpr auto parity_info = Spectral::compute_parity_list<VariablesTags>();
  const auto parity_list = std::get<0>(parity_info);
  const auto num_even_comps = std::get<1>(parity_info);
  const auto num_odd_comps = std::get<2>(parity_info);

  const size_t num_grid_points = mesh.number_of_grid_points();
  // NOLINTNEXTLINE(modernize-avoid-c-arrays)
  const auto buffer = cpp20::make_unique_for_overwrite<double[]>(
      (num_even_comps + num_odd_comps) * num_grid_points);
  DataVector even_components{};
  even_components.set_data_ref(&buffer[0], num_even_comps * num_grid_points);
  DataVector odd_components{};
  odd_components.set_data_ref(&buffer[num_even_comps * num_grid_points],
                              num_odd_comps * num_grid_points);
  double* p_even_components = even_components.data();
  double* p_odd_components = odd_components.data();

  bool even = true;
  for (const auto seg_size : parity_list) {
    if (UNLIKELY(seg_size == 0)) {
      if (even) {
        // Will have leading zero if first element is odd
        even = false;
        continue;
      } else {
        // Iterated through all non-zero values
        break;
      }
    }
    if (even) {
      std::copy(p_data, p_data + seg_size * num_grid_points, p_even_components);
      p_even_components += seg_size * num_grid_points;
    } else {
      std::copy(p_data, p_data + seg_size * num_grid_points, p_odd_components);
      p_odd_components += seg_size * num_grid_points;
    }
    p_data += seg_size * num_grid_points;
    even = not even;
  }
  if (num_even_comps > 0) {
    even_components =
        apply_matrices(filter_even, even_components, mesh.extents());
  }
  if (num_odd_comps > 0) {
    odd_components = apply_matrices(filter_odd, odd_components, mesh.extents());
  }
  // reset pointers to beginning of buffers
  p_data = u->data();
  p_even_components = even_components.data();
  p_odd_components = odd_components.data();
  even = true;
  for (const auto seg_size : parity_list) {
    if (UNLIKELY(seg_size == 0)) {
      if (even) {
        // Will have leading zero if first element is odd
        even = false;
        continue;
      } else {
        // Iterated through all non-zero values
        break;
      }
    }
    // Now copy the data from the buffer to the correct location in u
    if (even) {
      std::copy(p_even_components,
                p_even_components + seg_size * num_grid_points, p_data);
      p_even_components += seg_size * num_grid_points;
    } else {
      std::copy(p_odd_components, p_odd_components + seg_size * num_grid_points,
                p_data);
      p_odd_components += seg_size * num_grid_points;
    }
    p_data += seg_size * num_grid_points;
    even = not even;
  }
}

}  // namespace Spectral::filtering
