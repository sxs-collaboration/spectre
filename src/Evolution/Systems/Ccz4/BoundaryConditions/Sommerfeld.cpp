// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/BoundaryConditions/Sommerfeld.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "Utilities/CallWithDynamicType.hpp"

namespace Ccz4::BoundaryConditions {

Sommerfeld::Sommerfeld(const Sommerfeld& rhs)
    : PUP::able(rhs),
      BoundaryCondition{dynamic_cast<const BoundaryCondition&>(rhs)} {}

Sommerfeld& Sommerfeld::operator=(const Sommerfeld& /*rhs*/) { return *this; }

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Sommerfeld::get_clone() const {
  return std::make_unique<Sommerfeld>(*this);
}

void Sommerfeld::pup(PUP::er& p) { BoundaryCondition::pup(p); }
#if defined(SPECTRE_USE_CHARM)
// NOLINTNEXTLINE
PUP::able::PUP_ID Sommerfeld::my_PUP_ID = 0;
#endif  // SPECTRE_USE_CHARM

void Sommerfeld::fd_ghost(
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

    // fd_interior_evolved_variables_tags (variables_tag_list order)
    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_conformal_metric,
    const Scalar<DataVector>& interior_conformal_factor,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_a_tilde,
    const Scalar<DataVector>& interior_trace_extrinsic_curvature,
    const Scalar<DataVector>& interior_theta,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_gamma_hat,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_auxiliary_shift_b,

    // fd_interior_temporary_tags
    const Mesh<3>& subcell_mesh,

    // fd_gridless_tags
    const fd::Reconstructor& reconstructor) {
  const size_t ghost_zone_size = reconstructor.ghost_zone_size();

  const auto ghost_logical_coords =
      evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
          subcell_mesh, ghost_zone_size, direction);

  // Extrapolate the interior variables into the ghost zone of external bdry.
  // Modification to the time derivatives per Sommerfeld BC is handled in the
  // time derivative computation, not here.
  // Should add option to use higher order interpolant once that PR (#6999) is
  // merged.
  const intrp::Irregular<3> irregular_interpolant(subcell_mesh,
                                                  ghost_logical_coords);

  Variables<typename ::Ccz4::fd::System::variables_tag_list> interior_var{
      subcell_mesh.number_of_grid_points()};
  get<Tags::ConformalMetric<DataVector, 3>>(interior_var) =
      interior_conformal_metric;
  get<gr::Tags::Lapse<DataVector>>(interior_var) = interior_lapse;
  get<gr::Tags::Shift<DataVector, 3>>(interior_var) = interior_shift;
  get<Tags::ConformalFactor<DataVector>>(interior_var) =
      interior_conformal_factor;
  get<Tags::ATilde<DataVector, 3>>(interior_var) = interior_a_tilde;
  get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(interior_var) =
      interior_trace_extrinsic_curvature;
  get<Tags::Theta<DataVector>>(interior_var) = interior_theta;
  get<Tags::GammaHat<DataVector, 3>>(interior_var) = interior_gamma_hat;
  get<Tags::AuxiliaryShiftB<DataVector, 3>>(interior_var) =
      interior_auxiliary_shift_b;

  const auto boundary_values = irregular_interpolant.interpolate(interior_var);

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
