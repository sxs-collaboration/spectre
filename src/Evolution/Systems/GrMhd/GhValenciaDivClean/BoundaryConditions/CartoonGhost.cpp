// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/CartoonGhost.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/NeutrinoSystems.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/CartoonGhost.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::BoundaryConditions {
template <typename System>
CartoonGhost<System>::CartoonGhost(CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

template <typename System>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
CartoonGhost<System>::get_clone() const {
  return std::make_unique<CartoonGhost>(*this);
}

template <typename System>
void CartoonGhost<System>::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
}

template <typename System>
// NOLINTNEXTLINE
PUP::able::PUP_ID CartoonGhost<System>::my_PUP_ID = 0;

template <typename System>
void CartoonGhost<System>::fd_ghost(
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
        spacetime_metric,
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
    const gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> phi,
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        lorentz_factor_times_spatial_velocity,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,
    const Direction<3>& direction,
    // fd_interior_evolved_variables_tags
    const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& interior_phi,
    // fd_interior_temporary_tags
    const Mesh<3>& subcell_mesh,
    // fd_interior_primitive_variables_tags
    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_electron_fraction,
    const Scalar<DataVector>& interior_temperature,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const Scalar<DataVector>& interior_lorentz_factor,
    const Scalar<DataVector>& interior_divergence_cleaning_field,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    // fd_gridless_tags
    const fd::Reconstructor<System>& reconstructor) {
  fd_ghost_gh_impl(spacetime_metric, pi, phi, direction,
                   interior_spacetime_metric, interior_pi, interior_phi,
                   subcell_mesh, reconstructor.ghost_zone_size());

  // Fluxes not computed for GrMhd: these are placeholders
  const tnsr::I<DataVector, 3> interior_shift{};
  const Scalar<DataVector> interior_lapse{};
  const tnsr::ii<DataVector, 3> interior_spatial_metric{};
  tnsr::ii<DataVector, 3> spatial_metric{};
  tnsr::II<DataVector, 3> inv_spatial_metric{};
  Scalar<DataVector> sqrt_det_spatial_metric{};
  Scalar<DataVector> lapse{};
  tnsr::I<DataVector, 3> shift{};
  Scalar<DataVector> pressure{};
  Scalar<DataVector> specific_internal_energy{};
  tnsr::I<DataVector, 3> spatial_velocity{};
  Scalar<DataVector> lorentz_factor{};

  grmhd::ValenciaDivClean::BoundaryConditions::CartoonGhost::fd_ghost_impl(
      rest_mass_density, electron_fraction, temperature,
      make_not_null(&pressure), make_not_null(&specific_internal_energy),
      lorentz_factor_times_spatial_velocity, make_not_null(&spatial_velocity),
      make_not_null(&lorentz_factor), magnetic_field, divergence_cleaning_field,
      make_not_null(&spatial_metric), make_not_null(&inv_spatial_metric),
      make_not_null(&sqrt_det_spatial_metric), make_not_null(&lapse),
      make_not_null(&shift),

      direction,

      subcell_mesh,

      interior_rest_mass_density, interior_electron_fraction,
      interior_temperature, interior_pressure,
      interior_specific_internal_energy, interior_lorentz_factor,
      interior_divergence_cleaning_field, interior_spatial_velocity,
      interior_magnetic_field,
      // metric vars not used when not computing fluxes
      interior_spatial_metric, interior_lapse, interior_shift,

      reconstructor.ghost_zone_size(), false);
}

template <typename System>
void CartoonGhost<System>::fd_ghost_gh_impl(
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*>
        spacetime_metric,
    const gsl::not_null<tnsr::aa<DataVector, 3, Frame::Inertial>*> pi,
    const gsl::not_null<tnsr::iaa<DataVector, 3, Frame::Inertial>*> phi,
    const Direction<3>& direction,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_spacetime_metric,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& interior_pi,
    const tnsr::iaa<DataVector, 3, Frame::Inertial>& interior_phi,
    const Mesh<3>& subcell_mesh, const size_t ghost_zone_size) {
  const size_t dim_direction{direction.dimension()};
  ASSERT(dim_direction == 0,
         "Cartoon BC can only be applied in the x-direction, got "
             << dim_direction);

  const Index<3> subcell_extents = subcell_mesh.extents();
  const size_t num_face_pts{
      subcell_extents.slice_away(dim_direction).product()};

  using gh_tags = tmpl::list<SpacetimeMetric, Pi, Phi>;
  constexpr size_t buffer_size_per_grid_pts =
      Variables<gh_tags>::number_of_independent_components;
  DataVector buffer_for_vars{
      num_face_pts * (1 + ghost_zone_size) * buffer_size_per_grid_pts, 0.0};

  // `outermost_vars` is scratch space for one ghost layer at a time.
  Variables<gh_tags> outermost_vars{buffer_for_vars.data(),
                                    num_face_pts * buffer_size_per_grid_pts};
  Variables<gh_tags> ghost_vars{
      outermost_vars.data() + outermost_vars.size(),
      num_face_pts * buffer_size_per_grid_pts * ghost_zone_size};

  // Parity under x-reflection (dim_direction == 0):
  //   tnsr::aa components (a,b) are odd when exactly one of {a,b} equals 1
  //   (the spacetime x-index). Odd components are: (0,1), (1,2), (1,3).
  //   All others are even and keep their value.
  const auto apply_aa_parity =
      [](tnsr::aa<DataVector, 3, Frame::Inertial>& field,
         const tnsr::aa<DataVector, 3, Frame::Inertial>& evolved_field) {
        field = evolved_field;
        get<0, 1>(field) *= -1.0;
        get<1, 2>(field) *= -1.0;
        get<1, 3>(field) *= -1.0;
      };

  // Parity under x-reflection for tnsr::iaa (Phi_{iab}):
  //   x-index count = (i==0 ? 1 : 0) + (a==1 ? 1 : 0) + (b==1 ? 1 : 0)
  //   Component is odd when this count is odd.
  //
  //   i=0: odd when neither or both are x-index. Odd components:
  //          (0,0,0),(0,0,2),(0,0,3),(0,1,1),(0,2,2),(0,2,3),(0,3,3)
  //   i=1,2: odd when exactly one of {a,b} is x-index. Odd components:
  //          (1,0,1),(1,1,2),(1,1,3) and (2,0,1),(2,1,2),(2,1,3).
  const auto apply_phi_parity =
      [](tnsr::iaa<DataVector, 3, Frame::Inertial>& field,
         const tnsr::iaa<DataVector, 3, Frame::Inertial>& evolved_field) {
        field = evolved_field;
        // i=0 odd components
        get<0, 0, 0>(field) *= -1.0;
        get<0, 0, 2>(field) *= -1.0;
        get<0, 0, 3>(field) *= -1.0;
        get<0, 1, 1>(field) *= -1.0;
        get<0, 2, 2>(field) *= -1.0;
        get<0, 2, 3>(field) *= -1.0;
        get<0, 3, 3>(field) *= -1.0;
        // i=1 odd components
        get<1, 0, 1>(field) *= -1.0;
        get<1, 1, 2>(field) *= -1.0;
        get<1, 1, 3>(field) *= -1.0;
        // i=2 odd components
        get<2, 0, 1>(field) *= -1.0;
        get<2, 1, 2>(field) *= -1.0;
        get<2, 1, 3>(field) *= -1.0;
      };

  // The ghost data stencil is ordered deepest-first: ghost[k=0] is the cell
  // furthest from the element at xi = -(ghost_zone_size-0.5)*dxi, and
  // ghost[k=ghost_zone_size-1] is nearest at xi = -0.5*dxi.
  Index<3> ghost_data_extents = subcell_extents;
  ghost_data_extents[dim_direction] = ghost_zone_size;

  for (size_t k = 0; k < ghost_zone_size; ++k) {
    const size_t interior_ix = ghost_zone_size - 1 - k;
    apply_aa_parity(get<SpacetimeMetric>(outermost_vars),
                    data_on_slice(interior_spacetime_metric, subcell_extents,
                                  dim_direction, interior_ix));
    apply_aa_parity(get<Pi>(outermost_vars),
                    data_on_slice(interior_pi, subcell_extents, dim_direction,
                                  interior_ix));
    apply_phi_parity(get<Phi>(outermost_vars),
                     data_on_slice(interior_phi, subcell_extents, dim_direction,
                                   interior_ix));
    add_slice_to_data(make_not_null(&ghost_vars), outermost_vars,
                      ghost_data_extents, dim_direction, k);
  }

  *spacetime_metric = get<SpacetimeMetric>(ghost_vars);
  *pi = get<Pi>(ghost_vars);
  *phi = get<Phi>(ghost_vars);
}

#define NEUTRINO(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) \
  template class CartoonGhost<GhValenciaDivClean::System<NEUTRINO(data)>>;

GENERATE_INSTANTIATIONS(INSTANTIATION, GHMHD_NEUTRINOS)

#undef INSTANTIATION
#undef NEUTRINO

}  // namespace grmhd::GhValenciaDivClean::BoundaryConditions
