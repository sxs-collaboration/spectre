// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/Outflow.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DgSubcell/SliceTensor.hpp"
#include "Evolution/Systems/NewtonianEuler/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::BoundaryConditions {
template <size_t Dim>
Outflow<Dim>::Outflow(CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Outflow<Dim>::get_clone() const {
  return std::make_unique<Outflow>(*this);
}

template <size_t Dim>
void Outflow<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID Outflow<Dim>::my_PUP_ID = 0;

template <size_t Dim>
template <size_t ThermodynamicDim>
std::optional<std::string> Outflow<Dim>::dg_outflow(
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        outward_directed_normal_covector,

    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& velocity,
    const Scalar<DataVector>& specific_internal_energy,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) {
  double min_char_speed = std::numeric_limits<double>::signaling_NaN();

  // Temp buffer for sound speed, normal dot velocity, and (optional) normal
  // dot mesh velocity
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>>>
      buffer{get(mass_density).size()};

  auto& sound_speed = get<::Tags::TempScalar<0>>(buffer);
  sound_speed_squared(make_not_null(&sound_speed), mass_density,
                      specific_internal_energy, equation_of_state);
  get(sound_speed) = sqrt(get(sound_speed));

  auto& normal_dot_velocity = get<::Tags::TempScalar<1>>(buffer);
  dot_product(make_not_null(&normal_dot_velocity),
              outward_directed_normal_covector, velocity);

  if (face_mesh_velocity.has_value()) {
    auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<2>>(buffer);
    dot_product(make_not_null(&normal_dot_mesh_velocity),
                outward_directed_normal_covector, face_mesh_velocity.value());

    min_char_speed = min(get(normal_dot_velocity) - get(sound_speed) -
                         get(normal_dot_mesh_velocity));
  } else {
    min_char_speed = min(get(normal_dot_velocity) - get(sound_speed));
  }

  if (min_char_speed < 0.0) {
    return {MakeString{} << "Outflow boundary condition violated with the "
                            "characteristic speed : "
                         << min_char_speed << "\n"};
  }

  return std::nullopt;
}

template <size_t Dim>
template <size_t ThermodynamicDim>
void Outflow<Dim>::fd_outflow(
    const gsl::not_null<Scalar<DataVector>*> mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const Direction<Dim>& direction,

    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        outward_directed_normal_covector,

    const Mesh<Dim>& subcell_mesh,
    const Scalar<DataVector>& interior_mass_density,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_velocity,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_specific_internal_energy,

    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state,
    const fd::Reconstructor<Dim>& reconstructor) {
  double min_char_speed = std::numeric_limits<double>::signaling_NaN();

  const size_t ghost_zone_size{reconstructor.ghost_zone_size()};
  const size_t dim_direction{direction.dimension()};

  const auto subcell_extents{subcell_mesh.extents()};
  const size_t num_face_pts{
      subcell_extents.slice_away(dim_direction).product()};

  // The outflow condition here simply uses the outermost values on
  // cell-centered FD grid points to compute face values on the external
  // boundary. This is equivalent to adopting the piecewise constant FD
  // reconstruction for FD cells at the external boundaries.
  //
  // In the future we may want to use more accurate methods (for instance,
  // one-sided characteristic reconstruction using WENO) for imposing
  // higher-order outflow boundary condition.
  //
  auto get_boundary_val = [&direction, &subcell_extents](auto volume_tensor) {
    return evolution::dg::subcell::slice_tensor_for_subcell(
        volume_tensor, subcell_extents, 1, direction);
  };
  auto boundary_mass_density = get_boundary_val(interior_mass_density);
  auto boundary_velocity = get_boundary_val(interior_velocity);
  auto boundary_specific_internal_energy =
      get_boundary_val(interior_specific_internal_energy);

  // compute sound speed on the boundary interface
  auto boundary_sound_speed =
      sound_speed_squared(boundary_mass_density,
                          boundary_specific_internal_energy, equation_of_state);
  get(boundary_sound_speed) = sqrt(get(boundary_sound_speed));

  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>> buffer{};
  buffer.initialize(num_face_pts);

  auto& normal_dot_velocity = get<::Tags::TempScalar<0>>(buffer);
  dot_product(make_not_null(&normal_dot_velocity),
              outward_directed_normal_covector, boundary_velocity);

  // Compute the largest ingoing char speed
  if (face_mesh_velocity.has_value()) {
    auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<1>>(buffer);
    dot_product(make_not_null(&normal_dot_mesh_velocity),
                outward_directed_normal_covector, face_mesh_velocity.value());

    min_char_speed = min(get(normal_dot_velocity) - get(boundary_sound_speed) -
                         get(normal_dot_mesh_velocity));
  } else {
    min_char_speed = min(get(normal_dot_velocity) - get(boundary_sound_speed));
  }

  // Check the outflow condition
  if (min_char_speed < 0.0) {
    ERROR(
        "Subcell outflow boundary condition violated with the characteristic "
        "speed : "
        << min_char_speed << "\n");
  } else {
    // Once the outflow condition has been checked, we fill each slices of the
    // ghost data with the boundary values. The reason that we need this step is
    // to prevent floating point exceptions being raised during computing
    // subcell time derivative due to NaN or uninitialized values in ghost data.

    using MassDensity = Tags::MassDensity<DataVector>;
    using Velocity = Tags::Velocity<DataVector, Dim>;
    using Pressure = Tags::Pressure<DataVector>;

    Variables<tmpl::list<MassDensity, Velocity, Pressure>> boundary_vars{};
    boundary_vars.initialize(num_face_pts);
    get<MassDensity>(boundary_vars) = boundary_mass_density;
    get<Velocity>(boundary_vars) = boundary_velocity;
    get<Pressure>(boundary_vars) = get_boundary_val(interior_pressure);

    Index<Dim> ghost_data_extents = subcell_extents;
    ghost_data_extents[dim_direction] = ghost_zone_size;

    Variables<tmpl::list<MassDensity, Velocity, Pressure>> ghost_vars{};
    ghost_vars.initialize(num_face_pts * ghost_zone_size, 0.0);

    for (size_t i_ghost = 0; i_ghost < ghost_zone_size; ++i_ghost) {
      add_slice_to_data(make_not_null(&ghost_vars), boundary_vars,
                        ghost_data_extents, dim_direction, i_ghost);
    }

    *mass_density = get<MassDensity>(ghost_vars);
    *velocity = get<Velocity>(ghost_vars);
    *pressure = get<Pressure>(ghost_vars);
  }
}  // namespace NewtonianEuler::BoundaryConditions

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) template class Outflow<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#define THERMODIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                              \
  template std::optional<std::string>                                       \
  Outflow<DIM(data)>::dg_outflow<THERMODIM(data)>(                          \
      const std::optional<tnsr::I<DataVector, DIM(data), Frame::Inertial>>& \
          face_mesh_velocity,                                               \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                \
          outward_directed_normal_covector,                                 \
      const Scalar<DataVector>& mass_density,                               \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& velocity,      \
      const Scalar<DataVector>& specific_internal_energy,                   \
      const EquationsOfState::EquationOfState<false, THERMODIM(data)>&      \
          equation_of_state);                                               \
  template void Outflow<DIM(data)>::fd_outflow<THERMODIM(data)>(            \
      const gsl::not_null<Scalar<DataVector>*> mass_density,                \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*> \
          velocity,                                                         \
      const gsl::not_null<Scalar<DataVector>*> pressure,                    \
      const Direction<DIM(data)>& direction,                                \
      const std::optional<tnsr::I<DataVector, DIM(data), Frame::Inertial>>& \
          face_mesh_velocity,                                               \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                \
          outward_directed_normal_covector,                                 \
      const Mesh<DIM(data)>& subcell_mesh,                                  \
      const Scalar<DataVector>& interior_mass_density,                      \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>&                \
          interior_velocity,                                                \
      const Scalar<DataVector>& interior_pressure,                          \
      const Scalar<DataVector>& interior_specific_internal_energy,          \
      const EquationsOfState::EquationOfState<false, THERMODIM(data)>&      \
          equation_of_state,                                                \
      const fd::Reconstructor<DIM(data)>& reconstructor);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))

#undef INSTANTIATION

#undef THERMODIM

#undef DIM
}  // namespace NewtonianEuler::BoundaryConditions
