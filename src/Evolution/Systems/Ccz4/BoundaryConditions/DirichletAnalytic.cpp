// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/BoundaryConditions/DirichletAnalytic.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Ccz4/Solutions/Factory.hpp"
#include "Utilities/CallWithDynamicType.hpp"

namespace Ccz4::BoundaryConditions {

DirichletAnalytic::DirichletAnalytic(const DirichletAnalytic& rhs)
    : BoundaryCondition{dynamic_cast<const BoundaryCondition&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()) {}

DirichletAnalytic& DirichletAnalytic::operator=(const DirichletAnalytic& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  return *this;
}
DirichletAnalytic::DirichletAnalytic(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription)
    : analytic_prescription_(std::move(analytic_prescription)) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletAnalytic::get_clone() const {
  return std::make_unique<DirichletAnalytic>(*this);
}

void DirichletAnalytic::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
  p | analytic_prescription_;
}
// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletAnalytic::my_PUP_ID = 0;

void DirichletAnalytic::fd_ghost(
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        conformal_metric,
    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
    const gsl::not_null<Scalar<DataVector>*> conformal_factor,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*> a_tilde,
    const gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
    const gsl::not_null<Scalar<DataVector>*> theta,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> gamma_hat,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        auxiliary_shift_b,
    const Direction<3>& direction,

    // fd_interior_temporary_tags
    const Mesh<3>& subcell_mesh,

    // fd_gridless_tags
    double time,
    const std::unordered_map<
        std::string,
        std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const ElementMap<3, Frame::Grid>& logical_to_grid_map,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial_map,
    const fd::Reconstructor& reconstructor) const {
  const size_t ghost_zone_size = reconstructor.ghost_zone_size();

  const auto ghost_logical_coords =
      evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
          subcell_mesh, ghost_zone_size, direction);

  const auto ghost_inertial_coords = grid_to_inertial_map(
      logical_to_grid_map(ghost_logical_coords), time, functions_of_time);

  // Compute FD ghost data with the analytic data or solution
  auto boundary_values = call_with_dynamic_type<
      tuples::TaggedTuple<
          Tags::ConformalMetric<DataVector, 3>, gr::Tags::Lapse<DataVector>,
          gr::Tags::Shift<DataVector, 3>, Tags::ConformalFactor<DataVector>,
          Tags::ATilde<DataVector, 3>,
          gr::Tags::TraceExtrinsicCurvature<DataVector>,
          Tags::Theta<DataVector>, Tags::GammaHat<DataVector, 3>,
          Tags::AuxiliaryShiftB<DataVector, 3>>,
      Ccz4::Solutions::all_solutions>(
      analytic_prescription_.get(),
      [&ghost_inertial_coords, &time](const auto* const initial_data) {
        using spacetime_tags = tmpl::list<
            Tags::ConformalMetric<DataVector, 3>, gr::Tags::Lapse<DataVector>,
            gr::Tags::Shift<DataVector, 3>, Tags::ConformalFactor<DataVector>,
            Tags::ATilde<DataVector, 3>,
            gr::Tags::TraceExtrinsicCurvature<DataVector>,
            Tags::Theta<DataVector>, Tags::GammaHat<DataVector, 3>,
            Tags::AuxiliaryShiftB<DataVector, 3>>;
        if constexpr (is_analytic_solution_v<
                          std::decay_t<decltype(*initial_data)>>) {
          return initial_data->variables(ghost_inertial_coords, time,
                                         spacetime_tags{});
        } else if constexpr (evolution::is_numeric_initial_data_v<
                                 std::decay_t<decltype(*initial_data)>>) {
          ERROR(
              "Cannot currently use numeric initial data as an analytic "
              "prescription for boundary conditions.");
        } else {
          (void)time;
          return initial_data->variables(ghost_inertial_coords,
                                         spacetime_tags{});
        }
      });
  *conformal_metric =
      get<Tags::ConformalMetric<DataVector, 3>>(boundary_values);
  *lapse = get<gr::Tags::Lapse<DataVector>>(boundary_values);
  *shift = get<gr::Tags::Shift<DataVector, 3>>(boundary_values);
  *conformal_factor = get<Tags::ConformalFactor<DataVector>>(boundary_values);
  *a_tilde = get<Tags::ATilde<DataVector, 3>>(boundary_values);
  *trace_extrinsic_curvature =
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(boundary_values);
  *theta = get<Tags::Theta<DataVector>>(boundary_values);
  *gamma_hat = get<Tags::GammaHat<DataVector, 3>>(boundary_values);
  *auxiliary_shift_b =
      get<Tags::AuxiliaryShiftB<DataVector, 3>>(boundary_values);
}
}  // namespace Ccz4::BoundaryConditions
