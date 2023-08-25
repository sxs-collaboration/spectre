// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/DgSubcell/Tags/OnSubcellFaces.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell {

/*!
 * \brief Allocate or assign background general relativity quantities on
 * cell-centered and face-centered FD grid points, for evolution systems run on
 * a curved spacetime without solving Einstein equations (e.g. ValenciaDivclean,
 * ForceFree),
 *
 * \warning This mutator assumes that the GR analytic data or solution
 * specifying background spacetime metric is time-independent.
 *
 */
template <typename System, typename Metavariables, bool UsingRuntimeId,
          bool ComputeOnlyOnRollback>
struct BackgroundGrVars : tt::ConformsTo<db::protocols::Mutator> {
  static constexpr size_t volume_dim = System::volume_dim;

  using gr_vars_tag = typename System::spacetime_variables_tag;
  using inactive_gr_vars_tag =
      evolution::dg::subcell::Tags::Inactive<gr_vars_tag>;
  using subcell_faces_gr_tag = evolution::dg::subcell::Tags::OnSubcellFaces<
      typename System::flux_spacetime_variables_tag, volume_dim>;

  using GrVars = typename gr_vars_tag::type;
  using InactiveGrVars = typename inactive_gr_vars_tag::type;
  using SubcellFaceGrVars = typename subcell_faces_gr_tag::type;

  using argument_tags = tmpl::list<
      ::Tags::Time, domain::Tags::FunctionsOfTime,
      domain::Tags::Domain<volume_dim>, domain::Tags::Element<volume_dim>,
      domain::Tags::ElementMap<volume_dim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<volume_dim, Frame::Grid,
                                                  Frame::Inertial>,
      evolution::dg::subcell::Tags::Mesh<volume_dim>,
      evolution::dg::subcell::Tags::Coordinates<3, Frame::Inertial>,
      subcell::Tags::DidRollback,
      tmpl::conditional_t<UsingRuntimeId,
                          evolution::initial_data::Tags::InitialData,
                          ::Tags::AnalyticSolutionOrData>>;

  using return_tags =
      tmpl::list<gr_vars_tag, inactive_gr_vars_tag, subcell_faces_gr_tag>;

  template <typename T>
  static void apply(
      const gsl::not_null<GrVars*> active_gr_vars,
      const gsl::not_null<InactiveGrVars*> inactive_gr_vars,
      const gsl::not_null<SubcellFaceGrVars*> subcell_face_gr_vars,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const Domain<volume_dim>& domain, const Element<volume_dim>& element,
      const ElementMap<volume_dim, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, volume_dim>&
          grid_to_inertial_map,
      const Mesh<volume_dim>& subcell_mesh,
      const tnsr::I<DataVector, volume_dim, Frame::Inertial>&
          subcell_inertial_coords,
      const bool did_rollback, const T& solution_or_data) {
    const size_t num_subcell_pts = subcell_mesh.number_of_grid_points();

    if (gsl::at(*subcell_face_gr_vars, 0).number_of_grid_points() != 0) {
      // Evolution phase

      // Check if the mesh is actually moving i.e. block coordinate map is
      // time-dependent. If not, we can skip the evaluation of GR variables
      // since they may stay with their values assigned at the initialization
      // phase.
      const auto& element_id = element.id();
      const size_t block_id = element_id.block_id();
      const Block<volume_dim>& block = domain.blocks()[block_id];

      if (block.is_time_dependent()) {
        if (did_rollback or not ComputeOnlyOnRollback) {
          if (did_rollback) {
            // Right after rollback, subcell GR vars are stored in the
            // `inactive` one.
            ASSERT(inactive_gr_vars->number_of_grid_points() == num_subcell_pts,
                   "The size of subcell GR variables ("
                       << inactive_gr_vars->number_of_grid_points()
                       << ") is not equal to the number of FD grid points ("
                       << subcell_mesh.number_of_grid_points() << ").");

            cell_centered_impl(inactive_gr_vars, time, subcell_inertial_coords,
                               solution_or_data);

          } else {
            // In this case the element didn't rollback but started from FD.
            // Therefore subcell GR vars are in the `active` one.
            ASSERT(active_gr_vars->number_of_grid_points() == num_subcell_pts,
                   "The size of subcell GR variables ("
                       << active_gr_vars->number_of_grid_points()
                       << ") is not equal to the number of FD grid points ("
                       << subcell_mesh.number_of_grid_points() << ").");

            cell_centered_impl(active_gr_vars, time, subcell_inertial_coords,
                               solution_or_data);
          }

          face_centered_impl(subcell_face_gr_vars, time, functions_of_time,
                             logical_to_grid_map, grid_to_inertial_map,
                             subcell_mesh, solution_or_data);
        }
      }

    } else {
      // Initialization phase
      (*inactive_gr_vars).initialize(num_subcell_pts);

      ASSERT(Mesh<volume_dim>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                              subcell_mesh.quadrature(0)) == subcell_mesh,
             "The subcell mesh must have isotropic basis, quadrature. and "
             "extents but got "
                 << subcell_mesh);
      const size_t num_face_centered_mesh_grid_pts =
          (subcell_mesh.extents(0) + 1) * subcell_mesh.extents(1) *
          subcell_mesh.extents(2);
      for (size_t d = 0; d < volume_dim; ++d) {
        gsl::at(*subcell_face_gr_vars, d)
            .initialize(num_face_centered_mesh_grid_pts);
      }

      cell_centered_impl(inactive_gr_vars, time, subcell_inertial_coords,
                         solution_or_data);
      face_centered_impl(subcell_face_gr_vars, time, functions_of_time,
                         logical_to_grid_map, grid_to_inertial_map,
                         subcell_mesh, solution_or_data);
    }
  }

 private:
  template <typename Vars, typename T>
  static void cell_centered_impl(
      const gsl::not_null<Vars*> background_gr_vars, const double time,
      const tnsr::I<DataVector, volume_dim, Frame::Inertial>& inertial_coords,
      const T& solution_or_data) {
    GrVars temp{background_gr_vars->data(), background_gr_vars->size()};

    if constexpr (UsingRuntimeId) {
      using derived_classes =
          tmpl::at<typename Metavariables::factory_creation::factory_classes,
                   evolution::initial_data::InitialData>;
      call_with_dynamic_type<void, derived_classes>(
          &solution_or_data, [&temp, &inertial_coords,
                              &time](const auto* const solution_or_data_ptr) {
            temp.assign_subset(evolution::Initialization::initial_data(
                *solution_or_data_ptr, inertial_coords, time,
                typename GrVars::tags_list{}));
          });
    } else {
      temp.assign_subset(evolution::Initialization::initial_data(
          solution_or_data, inertial_coords, time,
          typename GrVars::tags_list{}));
    }
  }

  template <typename T>
  static void face_centered_impl(
      const gsl::not_null<SubcellFaceGrVars*> face_centered_gr_vars,
      const double time,
      const std::unordered_map<
          std::string,
          std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const ElementMap<volume_dim, Frame::Grid>& logical_to_grid_map,
      const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, volume_dim>&
          grid_to_inertial_map,
      const Mesh<volume_dim>& subcell_mesh, const T& solution_or_data) {
    ASSERT(Mesh<volume_dim>(subcell_mesh.extents(0), subcell_mesh.basis(0),
                            subcell_mesh.quadrature(0)) == subcell_mesh,
           "The subcell mesh must have isotropic basis, quadrature. and "
           "extents but got "
               << subcell_mesh);

    for (size_t dim = 0; dim < volume_dim; ++dim) {
      const auto basis = make_array<volume_dim>(subcell_mesh.basis(0));
      auto quadrature = make_array<volume_dim>(subcell_mesh.quadrature(0));
      auto extents = make_array<volume_dim>(subcell_mesh.extents(0));

      gsl::at(extents, dim) = subcell_mesh.extents(0) + 1;
      gsl::at(quadrature, dim) =
          SpatialDiscretization::Quadrature::FaceCentered;

      const Mesh<volume_dim> face_centered_mesh{extents, basis, quadrature};
      const auto face_centered_logical_coords =
          logical_coordinates(face_centered_mesh);
      const auto face_centered_inertial_coords = grid_to_inertial_map(
          logical_to_grid_map(face_centered_logical_coords), time,
          functions_of_time);

      if constexpr (UsingRuntimeId) {
        using derived_classes =
            tmpl::at<typename Metavariables::factory_creation::factory_classes,
                     evolution::initial_data::InitialData>;
        call_with_dynamic_type<void, derived_classes>(
            &solution_or_data,
            [&face_centered_gr_vars, &face_centered_inertial_coords, &dim,
             &time](const auto* const solution_or_data_ptr) {
              gsl::at(*face_centered_gr_vars, dim)
                  .assign_subset(evolution::Initialization::initial_data(
                      *solution_or_data_ptr, face_centered_inertial_coords,
                      time,
                      typename SubcellFaceGrVars::value_type::tags_list{}));
            });
      } else {
        gsl::at(*face_centered_gr_vars, dim)
            .assign_subset(evolution::Initialization::initial_data(
                solution_or_data, face_centered_inertial_coords, time,
                typename SubcellFaceGrVars::value_type::tags_list{}));
      }
    }
  }
};

}  // namespace evolution::dg::subcell
