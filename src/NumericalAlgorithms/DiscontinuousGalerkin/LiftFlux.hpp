// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function lift_flux.

#pragma once

#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/TMPL.hpp"

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "NumericalAlgorithms/LinearOperators/Mass.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

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
/// diagonalization of the mass matrix and its implications,
/// \cite Teukolsky2015ega, especially Section 3.
///
/// \note The result is still provided only on the boundary grid.  The
/// values away from the boundary are zero and are not stored.
template <typename... FluxTags>
auto lift_flux(Variables<tmpl::list<FluxTags...>> flux,
               const size_t extent_perpendicular_to_boundary,
               Scalar<DataVector> magnitude_of_face_normal) noexcept
    -> Variables<tmpl::list<db::remove_tag_prefix<FluxTags>...>> {
  auto lift_factor = std::move(get(magnitude_of_face_normal));
  // For an Nth degree basis (i.e., one with N+1 basis functions), the LGL
  // weights are:
  //   w_i = 2 / ((N + 1) * N * (P_{N}(xi_i))^2)
  // and so at the end points (xi = +/- 1) we get:
  //   w_0 = 2 / ((N + 1) * N)
  //   w_N = 2 / ((N + 1) * N)
  // The negative sign comes from bringing the boundary correction over to the
  // RHS of the equal sign (e.g. `du/dt=Source - div Flux - boundary corr`),
  // while the magnitude of the normal vector above accounts for the ratios of
  // spatial metrics and Jacobians.
  lift_factor *=
      -0.5 * static_cast<double>((extent_perpendicular_to_boundary *
                                  (extent_perpendicular_to_boundary - 1)));

  Variables<tmpl::list<db::remove_tag_prefix<FluxTags>...>> lifted_data(
      std::move(flux));
  lifted_data *= lift_factor;
  return lifted_data;
}

template <size_t Dim, typename... FluxTags>
auto lift_flux_no_mass_lumping(
    Variables<tmpl::list<FluxTags...>> flux, const Mesh<Dim>& volume_mesh,
    const Direction<Dim>& direction,
    Scalar<DataVector> magnitude_of_face_normal) noexcept
    -> Variables<tmpl::list<db::remove_tag_prefix<FluxTags>...>> {
  const auto& inv_mass_matrix = blaze::inv(
      Spectral::mass_matrix(volume_mesh.slice_through(direction.dimension())));
  std::array<Matrix, Dim> lifting_operator{};
  lifting_operator[direction.dimension()] = inv_mass_matrix;
  Variables<tmpl::list<db::remove_tag_prefix<FluxTags>...>> flux_in_volume{
      volume_mesh.number_of_grid_points(), 0.};
  // TODO: handle jacobian correctly, resolve it better on curved meshes e.g. by
  // taking the numerical integral of the mass matrix
  add_slice_to_data(make_not_null(&flux_in_volume),
                    -get(magnitude_of_face_normal) * flux,
                    volume_mesh.extents(), direction.dimension(),
                    index_to_slice_at(volume_mesh.extents(), direction));
  return apply_matrices(lifting_operator, flux_in_volume,
                        volume_mesh.extents());
}

template <size_t FaceDim, typename... FluxTags>
auto lift_flux_massive_no_mass_lumping(
    Variables<tmpl::list<FluxTags...>> flux, const Mesh<FaceDim>& face_mesh,
    const Scalar<DataVector>& surface_jacobian) noexcept
    -> Variables<tmpl::list<db::remove_tag_prefix<FluxTags>...>> {
  return Variables<tmpl::list<db::remove_tag_prefix<FluxTags>...>>(mass_on_face(
      Variables<tmpl::list<FluxTags...>>(-flux), face_mesh, surface_jacobian));
}

}  // namespace dg
