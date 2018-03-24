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
///
/// \param local_data Data containing the local NormalDotFlux values
/// and possibly other values, which will be ignored.
/// \param numerical_flux The numerical flux
/// \param extent_perpendicular_to_boundary The extent perpendicular
/// to the boundary
/// \param magnitude_of_face_normal The magnitude of the face normal
template <typename LocalDataTags, typename... NormalDotNumericalFluxTags>
auto lift_flux(
    const Variables<LocalDataTags>& local_data,
    Variables<tmpl::list<NormalDotNumericalFluxTags...>> numerical_flux,
    const size_t extent_perpendicular_to_boundary,
    Scalar<DataVector> magnitude_of_face_normal) noexcept
    -> Variables<
        tmpl::list<db::remove_tag_prefix<NormalDotNumericalFluxTags>...>> {
  auto lift_factor = std::move(get(magnitude_of_face_normal));
  lift_factor *= -0.5 * (extent_perpendicular_to_boundary *
                         (extent_perpendicular_to_boundary - 1));

  Variables<tmpl::list<db::remove_tag_prefix<NormalDotNumericalFluxTags>...>>
      lifted_data(std::move(numerical_flux));
  tmpl::for_each<typename decltype(lifted_data)::tags_list>(
      [&lifted_data, &local_data](auto tag) noexcept {
        using Tag = tmpl::type_from<decltype(tag)>;
        auto& lifted_tensor = get<Tag>(lifted_data);
        const auto& local_tensor =
            get<db::add_tag_prefix<Tags::NormalDotFlux, Tag>>(local_data);
        auto local_it = local_tensor.begin();
        for (auto lifted_it = lifted_tensor.begin();
             lifted_it != lifted_tensor.end();
             ++lifted_it, ++local_it) {
          *lifted_it -= *local_it;
        }
      });

  lifted_data *= lift_factor;
  return lifted_data;
}
}  // namespace dg
