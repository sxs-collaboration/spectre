// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
// @{
/*!
 * \brief Compute and add the 2nd-order flux divergence on a Cartesian mesh to
 * the cell-centered time derivatives.
 */
void add_cartesian_flux_divergence(const gsl::not_null<DataVector*> dt_var,
                                   const double one_over_delta,
                                   const DataVector& inv_jacobian,
                                   const DataVector& boundary_correction,
                                   const Index<1>& subcell_extents,
                                   const size_t dimension) noexcept {
  (void)dimension;
  ASSERT(dimension == 0, "dimension must be 0 but is " << dimension);
  for (size_t i = 0; i < subcell_extents[0]; ++i) {
    (*dt_var)[i] += one_over_delta * inv_jacobian[i] *
                    (boundary_correction[i + 1] - boundary_correction[i]);
  }
}

void add_cartesian_flux_divergence(const gsl::not_null<DataVector*> dt_var,
                                   const double one_over_delta,
                                   const DataVector& inv_jacobian,
                                   const DataVector& boundary_correction,
                                   const Index<2>& subcell_extents,
                                   const size_t dimension) noexcept {
  ASSERT(dimension == 0 or dimension == 1,
         "dimension must be 0 or 1 but is " << dimension);
  Index<2> subcell_face_extents = subcell_extents;
  ++subcell_face_extents[dimension];
  for (size_t j = 0; j < subcell_extents[1]; ++j) {
    for (size_t i = 0; i < subcell_extents[0]; ++i) {
      Index<2> index(i, j);
      const size_t volume_index = collapsed_index(index, subcell_extents);
      const size_t boundary_correction_lower_index =
          collapsed_index(index, subcell_face_extents);
      ++index[dimension];
      const size_t boundary_correction_upper_index =
          collapsed_index(index, subcell_face_extents);
      (*dt_var)[volume_index] +=
          one_over_delta * inv_jacobian[volume_index] *
          (boundary_correction[boundary_correction_upper_index] -
           boundary_correction[boundary_correction_lower_index]);
    }
  }
}

void add_cartesian_flux_divergence(const gsl::not_null<DataVector*> dt_var,
                                   const double one_over_delta,
                                   const DataVector& inv_jacobian,
                                   const DataVector& boundary_correction,
                                   const Index<3>& subcell_extents,
                                   const size_t dimension) noexcept {
  ASSERT(dimension == 0 or dimension == 1 or dimension == 2,
         "dimension must be 0, 1, or 2 but is " << dimension);
  Index<3> subcell_face_extents = subcell_extents;
  ++subcell_face_extents[dimension];
  for (size_t k = 0; k < subcell_extents[2]; ++k) {
    for (size_t j = 0; j < subcell_extents[1]; ++j) {
      for (size_t i = 0; i < subcell_extents[0]; ++i) {
        Index<3> index(i, j, k);
        const size_t volume_index = collapsed_index(index, subcell_extents);
        const size_t boundary_correction_lower_index =
            collapsed_index(index, subcell_face_extents);
        ++index[dimension];
        const size_t boundary_correction_upper_index =
            collapsed_index(index, subcell_face_extents);
        (*dt_var)[volume_index] +=
            one_over_delta * inv_jacobian[volume_index] *
            (boundary_correction[boundary_correction_upper_index] -
             boundary_correction[boundary_correction_lower_index]);
      }
    }
  }
}
// @}
}  // namespace evolution::dg::subcell
