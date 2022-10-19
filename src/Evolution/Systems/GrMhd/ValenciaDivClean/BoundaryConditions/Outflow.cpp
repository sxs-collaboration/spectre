// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/Outflow.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
Outflow::Outflow(CkMigrateMessage* const msg) : BoundaryCondition(msg) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Outflow::get_clone() const {
  return std::make_unique<Outflow>(*this);
}

void Outflow::pup(PUP::er& p) { BoundaryCondition::pup(p); }

std::optional<std::string> Outflow::dg_outflow(
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
    /*outward_directed_normal_vector*/,

    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& lapse) {
  double min_speed = std::numeric_limits<double>::signaling_NaN();
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>> buffer{
      get(lapse).size()};
  auto& normal_dot_shift = get<::Tags::TempScalar<0>>(buffer);
  dot_product(make_not_null(&normal_dot_shift),
              outward_directed_normal_covector, shift);
  if (face_mesh_velocity.has_value()) {
    auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<1>>(buffer);
    dot_product(make_not_null(&normal_dot_mesh_velocity),
                outward_directed_normal_covector, face_mesh_velocity.value());
    get(normal_dot_shift) += get(normal_dot_mesh_velocity);
  }
  // The characteristic speeds are bounded by \pm \alpha - \beta_n, and
  // saturate that bound, so there is no need to check the hydro-dependent
  // characteristic speeds.
  min_speed = std::min(min(-get(lapse) - get(normal_dot_shift)),
                       min(get(lapse) - get(normal_dot_shift)));
  if (min_speed < 0.0) {
    return {MakeString{} << "Outflow boundary condition violated. Speed: "
                         << min_speed << "\nn_i: "
                         << outward_directed_normal_covector << "\n"};
  }
  return std::nullopt;
}

void Outflow::fd_outflow(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        lorentz_factor_times_spatial_velocity,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,

    const Direction<3>& direction,

    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_covector,

    // fd_interior_temporary_tags
    const Mesh<3>& subcell_mesh,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& lapse,

    // fd_interior_primitive_variables_tags
    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_electron_fraction,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_divergence_cleaning_field,

    // fd_gridless_tags
    const fd::Reconstructor& reconstructor) {
  double min_char_speed = std::numeric_limits<double>::signaling_NaN();

  const size_t ghost_zone_size{reconstructor.ghost_zone_size()};
  const size_t dim_direction{direction.dimension()};

  const auto subcell_extents{subcell_mesh.extents()};
  const size_t num_face_pts{
      subcell_extents.slice_away(dim_direction).product()};

  // The outflow boundary condition below simply uses the outermost values on
  // cell-centered FD grid points to compute face values on the external
  // boundary. This is equivalent to adopting the piecewise constant (lowest
  // order) FD reconstruction for FD cells at the external boundaries.
  //
  // In the future we may want to use more accurate methods (for instance,
  // one-sided characteristic reconstruction using WENO) for imposing
  // higher-order outflow boundary condition.

  auto lapse_at_boundary = evolution::dg::subcell::slice_tensor_for_subcell(
      lapse, subcell_extents, 1, direction);
  auto shift_at_boundary = evolution::dg::subcell::slice_tensor_for_subcell(
      shift, subcell_extents, 1, direction);

  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>> buffer{
      num_face_pts};

  auto& normal_dot_shift = get<::Tags::TempScalar<0>>(buffer);
  dot_product(make_not_null(&normal_dot_shift),
              outward_directed_normal_covector, shift_at_boundary);

  if (face_mesh_velocity.has_value()) {
    auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<1>>(buffer);
    dot_product(make_not_null(&normal_dot_mesh_velocity),
                outward_directed_normal_covector, face_mesh_velocity.value());
    get(normal_dot_shift) += get(normal_dot_mesh_velocity);
  }
  // The characteristic speeds are bounded by \pm \alpha - \beta_n, and
  // saturate that bound, so there is no need to check the hydro-dependent
  // characteristic speeds.
  min_char_speed =
      std::min(min(-get(lapse_at_boundary) - get(normal_dot_shift)),
               min(get(lapse_at_boundary) - get(normal_dot_shift)));

  if (min_char_speed < 0.0) {
    ERROR("Subcell outflow boundary condition violated. Speed: "
          << min_char_speed << "\nn_i: " << outward_directed_normal_covector
          << "\n");
  } else {
    // Once the outflow condition has been checked, we fill each slices of the
    // ghost data with the boundary values. The reason that we need this step is
    // to prevent floating point exceptions being raised while computing the
    // subcell time derivative because of NaN or uninitialized values in ghost
    // data.

    using RestMassDensity = hydro::Tags::RestMassDensity<DataVector>;
    using ElectronFraction = hydro::Tags::ElectronFraction<DataVector>;
    using Pressure = hydro::Tags::Pressure<DataVector>;
    using LorentzFactorTimesSpatialVelocity =
        hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>;
    using MagneticField = hydro::Tags::MagneticField<DataVector, 3>;
    using DivergenceCleaningField =
        hydro::Tags::DivergenceCleaningField<DataVector>;

    using prim_tags_for_reconstruction =
        tmpl::list<RestMassDensity, ElectronFraction, Pressure,
                   LorentzFactorTimesSpatialVelocity, MagneticField,
                   DivergenceCleaningField>;

    // Create a single large DV to reduce the number of Variables allocations
    const size_t buffer_size_per_grid_pts =
        (*rest_mass_density).size() + (*electron_fraction).size() +
        (*pressure).size() + (*lorentz_factor_times_spatial_velocity).size() +
        (*magnetic_field).size() + (*divergence_cleaning_field).size();
    DataVector buffer_for_boundary_and_ghost_vars{
        buffer_size_per_grid_pts * num_face_pts * (1 + ghost_zone_size), 0.0};

    // a Variables object to store prim variables on outermost layer of FD grid
    // points
    Variables<prim_tags_for_reconstruction> boundary_vars{
        buffer_for_boundary_and_ghost_vars.data(),
        buffer_size_per_grid_pts * num_face_pts};
    // a Variables object to store prim variables on ghost zone
    Variables<prim_tags_for_reconstruction> ghost_vars{
        buffer_for_boundary_and_ghost_vars.data() + boundary_vars.size(),
        buffer_size_per_grid_pts * num_face_pts * ghost_zone_size};

    auto get_boundary_val = [&direction, &subcell_extents](auto volume_tensor) {
      return evolution::dg::subcell::slice_tensor_for_subcell(
          volume_tensor, subcell_extents, 1, direction);
    };

    get<RestMassDensity>(boundary_vars) =
        get_boundary_val(interior_rest_mass_density);
    get<ElectronFraction>(boundary_vars) =
        get_boundary_val(interior_electron_fraction);
    get<Pressure>(boundary_vars) = get_boundary_val(interior_pressure);
    // Note : 'lorentz factor times spatial velocity' needs to be in the FD
    // ghost data for reconstruction, instead of lorentz factor and spatial
    // velocity separately.
    for (size_t i = 0; i < 3; ++i) {
      get<LorentzFactorTimesSpatialVelocity>(boundary_vars).get(i) =
          get(get_boundary_val(interior_lorentz_factor)) *
          get_boundary_val(interior_spatial_velocity).get(i);
    }
    get<MagneticField>(boundary_vars) =
        get_boundary_val(interior_magnetic_field);
    get<DivergenceCleaningField>(boundary_vars) =
        get_boundary_val(interior_divergence_cleaning_field);

    // Next, copy boundary values into each slices of ghost zone vars
    Index<3> ghost_data_extents = subcell_extents;
    ghost_data_extents[dim_direction] = ghost_zone_size;

    for (size_t i_ghost = 0; i_ghost < ghost_zone_size; ++i_ghost) {
      add_slice_to_data(make_not_null(&ghost_vars), boundary_vars,
                        ghost_data_extents, dim_direction, i_ghost);
    }

    *rest_mass_density = get<RestMassDensity>(ghost_vars);
    *electron_fraction = get<ElectronFraction>(ghost_vars);
    *pressure = get<Pressure>(ghost_vars);
    *lorentz_factor_times_spatial_velocity =
        get<LorentzFactorTimesSpatialVelocity>(ghost_vars);
    *magnetic_field = get<MagneticField>(ghost_vars);
    *divergence_cleaning_field = get<DivergenceCleaningField>(ghost_vars);
  }
}

// NOLINTNEXTLINE
PUP::able::PUP_ID Outflow::my_PUP_ID = 0;

}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
