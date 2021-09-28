// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Initialization {
namespace Tags {
struct InitialTime;
}  // namespace Tags
}  // namespace Initialization
namespace Tags {
struct AnalyticSolutionOrData;
}  // namespace Tags
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
template <size_t VolumeDim>
struct Mesh;
}  // namespace Tags
}  // namespace domain
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace Initialization {
/// Generic initialization actions and mutators for the subcell solver.
namespace subcell {
/*!
 * \ingroup InitializationGroup
 * \brief Allocate and set general relativity quantities needed for evolution
 * of hydro systems when using a DG-subcell hybrid scheme.
 *
 * Uses:
 * - DataBox:
 *   * `evolution::dg::subcell::Tags::Mesh<Dim>`
 *   * `domain::Tags::ElementMap<Dim, Frame::Grid>`
 *   * `Tags::CoordinateMap<Dim, Frame::Grid, Frame::Inertial>`
 *   * `domain::Tags::FunctionsOfTime`
 *   * `evolution::dg::subcell::Tags::Coordinates<Dim, Frame::ElementLogical>`
 *   * `::Tags::AnalyticSolutionOrData`
 *
 * DataBox changes:
 * - Adds:
 *   * `dg::subcell::Tags::Inactive<System::spacetime_variables_tag>`
 *   * `Tags::OnSubcellFaces<System::flux_spacetime_variables_tag, Dim>`
 *
 * - Removes: nothing
 * - Modifies: nothing
 *
 * \note This action relies on the `SetupDataBox` aggregated initialization
 * mechanism, so `Actions::SetupDataBox` must be present in the
 * `Initialization` phase action list prior to this action.
 */
template <typename System, size_t Dim>
struct GrTagsForHydro {
  using initialization_tags = tmpl::list<Initialization::Tags::InitialTime>;

  using gr_tag = typename System::spacetime_variables_tag;
  using subcell_gr_tag = evolution::dg::subcell::Tags::Inactive<gr_tag>;
  using subcell_faces_gr_tag = evolution::dg::subcell::Tags::OnSubcellFaces<
      typename System::flux_spacetime_variables_tag, Dim>;
  using GrVars = typename gr_tag::type;
  using SubcellGrVars = typename subcell_gr_tag::type;
  using FaceGrVars = typename subcell_faces_gr_tag::type::value_type;

  using return_tags = tmpl::list<subcell_gr_tag, subcell_faces_gr_tag>;
  using argument_tags = tmpl::list<
      Initialization::Tags::InitialTime,
      evolution::dg::subcell::Tags::Mesh<Dim>,
      domain::Tags::ElementMap<Dim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                  Frame::Inertial>,
      domain::Tags::FunctionsOfTime,
      evolution::dg::subcell::Tags::Coordinates<Dim, Frame::ElementLogical>,
      ::Tags::AnalyticSolutionOrData>;

  template <typename AnalyticDataOrSolution>
  static void apply(
      const gsl::not_null<SubcellGrVars*> cell_centered_gr_vars,
      const gsl::not_null<std::array<FaceGrVars, Dim>*> face_centered_gr_vars,
      const double initial_time, const Mesh<Dim>& subcell_mesh,
      const ElementMap<Dim, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>&
          grid_to_inertial_map,
      const std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const tnsr::I<DataVector, Dim, Frame::ElementLogical>&
          subcell_logical_coordinates,
      const AnalyticDataOrSolution& analytic_data_or_solution) noexcept {
    const size_t num_grid_points = subcell_mesh.number_of_grid_points();
    const auto cell_centered_inertial_coords =
        grid_to_inertial_map(logical_to_grid_map(subcell_logical_coordinates),
                             initial_time, functions_of_time);

    // Set cell-centered vars. Need to first do without prefix then move into
    // prefixed Variables.
    GrVars no_prefix_cell_centered_gr_vars{num_grid_points};
    no_prefix_cell_centered_gr_vars.assign_subset(evolution::initial_data(
        analytic_data_or_solution, cell_centered_inertial_coords, initial_time,
        typename GrVars::tags_list{}));
    *cell_centered_gr_vars = std::move(no_prefix_cell_centered_gr_vars);

    // Set GR variables needed for computing the fluxes on the faces.
    ASSERT(Mesh<Dim>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                     subcell_mesh.quadrature(0)) == subcell_mesh,
           "The subcell mesh must have isotropic basis, quadrature. and "
           "extents but got "
               << subcell_mesh);
    for (size_t d = 0; d < Dim; ++d) {
      const auto basis = make_array<Dim>(subcell_mesh.basis(0));
      auto quadrature = make_array<Dim>(subcell_mesh.quadrature(0));
      auto extents = make_array<Dim>(subcell_mesh.extents(0));
      gsl::at(extents, d) = subcell_mesh.extents(0) + 1;
      gsl::at(quadrature, d) = Spectral::Quadrature::FaceCentered;
      const Mesh<Dim> face_centered_mesh{extents, basis, quadrature};
      const auto face_centered_logical_coords =
          logical_coordinates(face_centered_mesh);
      const auto face_centered_inertial_coords = grid_to_inertial_map(
          logical_to_grid_map(face_centered_logical_coords), initial_time,
          functions_of_time);

      gsl::at(*face_centered_gr_vars, d)
          .initialize(face_centered_mesh.number_of_grid_points());
      gsl::at(*face_centered_gr_vars, d)
          .assign_subset(evolution::initial_data(
              analytic_data_or_solution, face_centered_inertial_coords,
              initial_time, typename FaceGrVars::tags_list{}));
    }
  }
};
}  // namespace subcell
}  // namespace Initialization
