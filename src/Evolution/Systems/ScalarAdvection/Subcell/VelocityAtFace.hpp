// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/VelocityField.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
template <size_t VolumeDim>
struct Mesh;
}  // namespace Tags
}  // namespace domain
/// \endcond

namespace ScalarAdvection::subcell {
/*!
 * \brief Allocate and set the velocity field needed for evolving
 * ScalarAdvection system when using a DG-subcell hybrid scheme.
 *
 * Uses:
 * - DataBox:
 *   * `Initialization::Tags::InitialTime`
 *   * `evolution::dg::subcell::Tags::Mesh<Dim>`
 *   * `domain::Tags::ElementMap<Dim, Frame::Grid>`
 *   * `Tags::CoordinateMap<Dim, Frame::Grid, Frame::Inertial>`
 *   * `domain::Tags::FunctionsOfTime`
 *   * `evolution::dg::subcell::Tags::Coordinates<Dim, Frame::ElementLogical>`
 *
 * DataBox changes:
 * - Adds:
 *   * `evolution::dg::subcell::Tags::Inactive<velocity_field>`
 *   * `evolution::dg::subcell::Tags::OnSubcellFaces<velocity_field, Dim>`
 * where `velocity_field` is `ScalarAdvection::Tags::VelocityField<Dim>`.
 *
 * - Removes: nothing
 * - Modifies: nothing
 *
 * \note This mutator is meant to be used with
 * `Initialization::Actions::AddSimpleTags` (which, in turn, uses
 * `Actions::SetupDataBox` aggregated initialization mechanism).
 */
template <size_t Dim>
struct VelocityAtFace {
  using initialization_tags = tmpl::list<Initialization::Tags::InitialTime>;

  using velocity_field = ::ScalarAdvection::Tags::VelocityField<Dim>;
  using subcell_velocity_field =
      ::evolution::dg::subcell::Tags::Inactive<velocity_field>;
  using subcell_faces_velocity_field =
      ::evolution::dg::subcell::Tags::OnSubcellFaces<velocity_field, Dim>;

  using vars = typename velocity_field::type;
  using subcell_vars = typename subcell_velocity_field::type;
  using face_vars = typename subcell_faces_velocity_field::type::value_type;

  using return_tags =
      tmpl::list<subcell_velocity_field, subcell_faces_velocity_field>;

  using argument_tags = tmpl::list<
      Initialization::Tags::InitialTime,
      evolution::dg::subcell::Tags::Mesh<Dim>,
      domain::Tags::ElementMap<Dim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>,
      domain::Tags::FunctionsOfTime,
      evolution::dg::subcell::Tags::Coordinates<Dim, Frame::ElementLogical>>;

  static void apply(
      const gsl::not_null<subcell_vars*> cell_centered_vars,
      const gsl::not_null<std::array<face_vars, Dim>*> face_centered_vars,
      const double initial_time, const Mesh<Dim>& subcell_mesh,
      const ElementMap<Dim, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>&
          grid_to_inertial_map,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const tnsr::I<DataVector, Dim, Frame::ElementLogical>&
          subcell_logical_coordinates) {
    // check if the subcell mesh is isotropic
    ASSERT(Mesh<Dim>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                     subcell_mesh.quadrature(0)) == subcell_mesh,
           "The subcell mesh must have isotropic basis, quadrature. and "
           "extents but got "
               << subcell_mesh);

    const size_t num_grid_points = subcell_mesh.number_of_grid_points();
    const auto cell_centered_inertial_coords =
        grid_to_inertial_map(logical_to_grid_map(subcell_logical_coordinates),
                             initial_time, functions_of_time);

    // Set cell-centered vars. Need to first do without prefix then move into
    // prefixed Variables.
    vars no_prefix_cell_centered_vars{num_grid_points};
    // Note: This assumes the velocity field is the hard-coded one in the
    // compute tag
    ::ScalarAdvection::Tags::VelocityFieldCompute<Dim>::function(
        make_not_null(&no_prefix_cell_centered_vars),
        cell_centered_inertial_coords);
    *cell_centered_vars = std::move(no_prefix_cell_centered_vars);

    // Set face-centered vars.
    for (size_t dim = 0; dim < Dim; ++dim) {
      const auto basis = make_array<Dim>(subcell_mesh.basis(0));
      auto quadrature = make_array<Dim>(subcell_mesh.quadrature(0));
      auto extents = make_array<Dim>(subcell_mesh.extents(0));
      gsl::at(extents, dim) = subcell_mesh.extents(0) + 1;
      gsl::at(quadrature, dim) = Spectral::Quadrature::FaceCentered;
      const Mesh<Dim> face_centered_mesh{extents, basis, quadrature};
      const auto face_centered_logical_coords =
          logical_coordinates(face_centered_mesh);
      const auto face_centered_inertial_coords = grid_to_inertial_map(
          logical_to_grid_map(face_centered_logical_coords), initial_time,
          functions_of_time);

      ::ScalarAdvection::Tags::VelocityFieldCompute<Dim>::function(
          make_not_null(&(gsl::at(*face_centered_vars, dim))),
          face_centered_inertial_coords);
    }
  }
};
}  // namespace ScalarAdvection::subcell
