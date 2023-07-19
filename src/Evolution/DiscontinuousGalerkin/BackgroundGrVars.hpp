// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Block.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"

namespace evolution::dg {

namespace {
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(flux_spacetime_variables_tag)

struct EmptyStruct {
  using tags_list = tmpl::list<>;
};
}  // namespace

/*!
 * \brief Allocate or assign background general relativity quantities needed
 * for evolution systems run on a curved spacetime without solving Einstein
 * equations (e.g. ValenciaDivclean, ForceFree).
 *
 * \warning This mutator assumes that the GR analytic data or solution
 * specifying background spacetime metric is time-independent.
 *
 */
template <typename System, typename Metavariables, bool UsingRuntimeId>
struct BackgroundGrVars {
  static constexpr size_t volume_dim = System::volume_dim;

  // Collect all the GR quantities used in the templated evolution system
  using gr_variables_tag = ::Tags::Variables<tmpl::remove_duplicates<
      tmpl::append<typename System::spacetime_variables_tag::tags_list,
                   typename get_flux_spacetime_variables_tag_or_default_t<
                       System, EmptyStruct>::tags_list,
                   tmpl::list<typename System::inverse_spatial_metric_tag>>>>;

  using GrVars = typename gr_variables_tag::type;

  using argument_tags =
      tmpl::list<::Tags::Time, domain::Tags::Domain<volume_dim>,
                 domain::Tags::Coordinates<volume_dim, Frame::Inertial>,
                 domain::Tags::Mesh<volume_dim>,
                 domain::Tags::Element<volume_dim>,
                 tmpl::conditional_t<UsingRuntimeId,
                                     evolution::initial_data::Tags::InitialData,
                                     ::Tags::AnalyticSolutionOrData>>;

  using return_tags = tmpl::list<gr_variables_tag>;

  template <typename T>
  static void apply(
      const gsl::not_null<GrVars*> background_gr_vars, const double time,
      const Domain<volume_dim>& domain,
      const tnsr::I<DataVector, volume_dim, Frame::Inertial>& inertial_coords,
      const Mesh<volume_dim>& mesh, const Element<volume_dim>& element,
      const T& solution_or_data) {
    const size_t num_grid_pts = mesh.number_of_grid_points();

    if (background_gr_vars->number_of_grid_points() != 0) {  // Evolution phase
      ASSERT((*background_gr_vars).number_of_grid_points() == num_grid_pts,
             "The size of GR variables ("
                 << (*background_gr_vars).number_of_grid_points()
                 << ") is not equal to the number of grid points ("
                 << mesh.number_of_grid_points() << ").");

      // Check if the mesh is actually moving i.e. block coordinate map is
      // time-dependent. If not, we can skip the evaluation of GR variables
      // since they may stay with their values assigned at the initialization
      // phase.
      const auto& element_id = element.id();
      const size_t block_id = element_id.block_id();
      const Block<volume_dim>& block = domain.blocks()[block_id];
      if (block.is_time_dependent()) {
        impl(background_gr_vars, time, inertial_coords, solution_or_data);
      }
    } else {
      // Initialization phase
      (*background_gr_vars).initialize(num_grid_pts);
      impl(background_gr_vars, time, inertial_coords, solution_or_data);
    }
  }

 private:
  template <typename T>
  static void impl(
      const gsl::not_null<GrVars*> background_gr_vars, const double time,
      const tnsr::I<DataVector, volume_dim, Frame::Inertial>& inertial_coords,
      const T& solution_or_data) {
    if constexpr (UsingRuntimeId) {
      using derived_classes =
          tmpl::at<typename Metavariables::factory_creation::factory_classes,
                   evolution::initial_data::InitialData>;
      call_with_dynamic_type<void, derived_classes>(
          &solution_or_data, [&background_gr_vars, &inertial_coords,
                              &time](const auto* const solution_or_data_ptr) {
            (*background_gr_vars)
                .assign_subset(evolution::Initialization::initial_data(
                    *solution_or_data_ptr, inertial_coords, time,
                    typename GrVars::tags_list{}));
          });
    } else {
      (*background_gr_vars)
          .assign_subset(evolution::Initialization::initial_data(
              solution_or_data, inertial_coords, time,
              typename GrVars::tags_list{}));
    }
  }
};

}  // namespace evolution::dg
