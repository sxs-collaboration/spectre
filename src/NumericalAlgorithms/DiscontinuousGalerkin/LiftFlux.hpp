// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function lift_flux.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"

namespace dg {
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Lifts the flux contribution from an interface to the volume.
///
/// The lifting operation takes the (d-1)-dimensional flux term at the
/// interface and computes the corresponding d-dimensional term in the
/// volume. SpECTRE implements an efficient DG method in which each
/// interface grid point contributes only to that same grid point of the
/// volume.
///
/// \details
/// SpECTRE implements a DG method with a diagonalized mass matrix (also
/// known as a mass-lumping scheme). This choice gives a large
/// reduction in the computational cost of the lifting operation, however,
/// the scheme is slightly less accurate, especially when the grid is
/// deformed by non-trivial Jacobians. For more details on the
/// diagonalization of the mass matrix and its implications, see Saul's
/// DG formulation paper, especially Section 3:
/// https://arxiv.org/abs/1510.01190.
///
/// \note The result is still provided only on the boundary grid.  The
/// values away from the boundary are zero and are not stored.
template <typename... NormalDotFluxTags, typename NormalDotNumericalFluxTags>
Variables<tmpl::list<
    db::add_tag_prefix<Tags::dt, db::remove_tag_prefix<NormalDotFluxTags>>...>>
lift_flux(const Variables<tmpl::list<NormalDotFluxTags...>>& local_flux,
          Variables<NormalDotNumericalFluxTags> numerical_flux,
          const size_t extent_perpendicular_to_boundary,
          DataVector magnitude_of_face_normal) noexcept {
  auto& lift_factor = magnitude_of_face_normal;
  lift_factor *= -0.5 * (extent_perpendicular_to_boundary *
                         (extent_perpendicular_to_boundary - 1));

  auto lifted_data = std::move(numerical_flux);
  lifted_data -= local_flux;
  lifted_data *= lift_factor;
  return Variables<tmpl::list<db::add_tag_prefix<
      Tags::dt, db::remove_tag_prefix<NormalDotFluxTags>>...>>(
      std::move(lifted_data));
}
}  // namespace dg
