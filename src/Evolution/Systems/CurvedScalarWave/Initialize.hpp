// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Initialization/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "Time/Time.hpp"

namespace CurvedScalarWave::Initialization {
/// \ingroup InitializationGroup
/// \brief Mutator meant to be used with
/// `Initialization::Actions::AddSimpleTags` to initialize the constraint
/// damping parameters of the CurvedScalarWave system
///
/// DataBox changes:
/// - Adds:
///   * `CurvedScalarWave::Tags::ConstraintGamma1`
///   * `CurvedScalarWave::Tags::ConstraintGamma2`
/// - Removes: nothing
/// - Modifies: nothing

template <size_t Dim>
struct InitializeConstraintDampingGammas {
  using return_tags =
      tmpl::list<Tags::ConstraintGamma1, Tags::ConstraintGamma2>;
  using argument_tags = tmpl::list<domain::Tags::Mesh<Dim>>;

  static void apply(const gsl::not_null<Scalar<DataVector>*> gamma1,
                    const gsl::not_null<Scalar<DataVector>*> gamma2,
                    const Mesh<Dim>& mesh) {
    const size_t number_of_grid_points = mesh.number_of_grid_points();
    *gamma1 = Scalar<DataVector>{number_of_grid_points, 0.};
    *gamma2 = Scalar<DataVector>{number_of_grid_points, 1.};
  }
};

/// \ingroup InitializationGroup
/// \brief Mutator meant to be used with
/// `Initialization::Actions::AddSimpleTags` to initialize items related to the
/// spacetime background of the CurvedScalarWave system
///
/// DataBox changes:
/// - Adds:
///   * `CurvedScalarWave::System::spacetime_variables_tag`
/// - Removes: nothing
/// - Modifies: nothing

template <size_t Dim>
struct InitializeGrVars {
  using gr_vars_tag =
      typename CurvedScalarWave::System<Dim>::spacetime_variables_tag;
  using GrVars = typename gr_vars_tag::type;
  using return_tags = tmpl::list<gr_vars_tag>;
  using argument_tags =
      tmpl::list<::Initialization::Tags::InitialTime,
                 domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 ::Tags::AnalyticSolutionOrData>;

  template <typename AnalyticSolutionOrData>
  static void apply(
      const gsl::not_null<GrVars*> gr_vars, const double initial_time,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
      const AnalyticSolutionOrData& analytic_solution_or_data) {
    gr_vars->initialize(get<0>(inertial_coords).size());
    // Set initial data from analytic solution
    gr_vars->assign_subset(evolution::Initialization::initial_data(
        analytic_solution_or_data, inertial_coords, initial_time,
        typename GrVars::tags_list{}));
  }
};

}  // namespace CurvedScalarWave::Initialization
