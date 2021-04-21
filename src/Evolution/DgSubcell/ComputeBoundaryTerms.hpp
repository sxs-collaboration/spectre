// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>

#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell {
template <typename... CorrectionTags, typename BoundaryCorrection,
          typename... PackageFieldTags>
void compute_boundary_terms(
    const gsl::not_null<Variables<tmpl::list<CorrectionTags...>>*>
        boundary_corrections_on_face,
    const BoundaryCorrection& boundary_correction,
    const Variables<tmpl::list<PackageFieldTags...>>& upper_packaged_data,
    const Variables<tmpl::list<PackageFieldTags...>>&
        lower_packaged_data) noexcept {
  ASSERT(
      upper_packaged_data.number_of_grid_points() ==
          lower_packaged_data.number_of_grid_points(),
      "The number of grid points must be the same for the upper packaged data ("
          << upper_packaged_data.number_of_grid_points()
          << ") and the lower packaged data ("
          << lower_packaged_data.number_of_grid_points() << ')');
  ASSERT(upper_packaged_data.number_of_grid_points() ==
             boundary_corrections_on_face->number_of_grid_points(),
         "The number of grid points must be the same for the packaged data ("
             << upper_packaged_data.number_of_grid_points()
             << ") and the boundary corrections on the faces ("
             << boundary_corrections_on_face->number_of_grid_points() << ')');
  boundary_correction.dg_boundary_terms(
      make_not_null(&get<CorrectionTags>(*boundary_corrections_on_face))...,
      get<PackageFieldTags>(upper_packaged_data)...,
      get<PackageFieldTags>(lower_packaged_data)...,
      // FD schemes are basically weak form FV scheme
      ::dg::Formulation::WeakInertial);
}
}  // namespace evolution::dg::subcell
