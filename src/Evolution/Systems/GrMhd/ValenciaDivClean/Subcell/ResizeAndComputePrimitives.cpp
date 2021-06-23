// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/ResizeAndComputePrimitives.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::subcell {
template <typename OrderedListOfRecoverySchemes>
template <size_t ThermodynamicDim>
void ResizeAndComputePrims<OrderedListOfRecoverySchemes>::apply(
    const gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*> prim_vars,
    const evolution::dg::subcell::ActiveGrid active_grid,
    const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        eos) noexcept {
  if (active_grid == evolution::dg::subcell::ActiveGrid::Dg) {
    ASSERT(prim_vars->number_of_grid_points() ==
               subcell_mesh.number_of_grid_points(),
           "The number of grid points of the primitive variables should also "
           "be the number of grid points the subcell mesh has ("
               << subcell_mesh.number_of_grid_points() << ") but got "
               << prim_vars->number_of_grid_points() << ". The DG grid has "
               << dg_mesh.number_of_grid_points() << " grid points");
    const size_t num_grid_points =
        (active_grid == evolution::dg::subcell::ActiveGrid::Dg ? dg_mesh
                                                               : subcell_mesh)
            .number_of_grid_points();
    // Reconstruct a copy of the pressure from the FD grid to the DG grid to
    // provide a high-order initial guess.
    const Scalar<DataVector> fd_pressure =
        get<hydro::Tags::Pressure<DataVector>>(*prim_vars);
    prim_vars->initialize(num_grid_points);
    evolution::dg::subcell::fd::reconstruct(
        make_not_null(&get(get<hydro::Tags::Pressure<DataVector>>(*prim_vars))),
        get(fd_pressure), dg_mesh, subcell_mesh.extents());

    // We only need to compute the prims if we switched to the DG grid because
    // otherwise we computed the prims during the FD TCI.
    grmhd::ValenciaDivClean::
        PrimitiveFromConservative<OrderedListOfRecoverySchemes, true>::apply(
            make_not_null(
                &get<hydro::Tags::RestMassDensity<DataVector>>(*prim_vars)),
            make_not_null(&get<hydro::Tags::SpecificInternalEnergy<DataVector>>(
                *prim_vars)),
            make_not_null(
                &get<hydro::Tags::SpatialVelocity<DataVector, 3>>(*prim_vars)),
            make_not_null(
                &get<hydro::Tags::MagneticField<DataVector, 3>>(*prim_vars)),
            make_not_null(
                &get<hydro::Tags::DivergenceCleaningField<DataVector>>(
                    *prim_vars)),
            make_not_null(
                &get<hydro::Tags::LorentzFactor<DataVector>>(*prim_vars)),
            make_not_null(&get<hydro::Tags::Pressure<DataVector>>(*prim_vars)),
            make_not_null(
                &get<hydro::Tags::SpecificEnthalpy<DataVector>>(*prim_vars)),
            tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, spatial_metric,
            inv_spatial_metric, sqrt_det_spatial_metric, eos);
  }
}

namespace {
using NewmanThenPalenzuela =
    tmpl::list<PrimitiveRecoverySchemes::NewmanHamlin,
               PrimitiveRecoverySchemes::PalenzuelaEtAl>;
using KastaunThenNewmanThenPalenzuela =
    tmpl::list<PrimitiveRecoverySchemes::KastaunEtAl,
               PrimitiveRecoverySchemes::NewmanHamlin,
               PrimitiveRecoverySchemes::PalenzuelaEtAl>;
}  // namespace

#define RECOVERY(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATION(r, data)                                                \
  template void                                                               \
  ResizeAndComputePrims<RECOVERY(data)>::apply<THERMO_DIM(data)>(             \
      const gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*>          \
          prim_vars,                                                          \
      const evolution::dg::subcell::ActiveGrid active_grid,                   \
      const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,                    \
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau, \
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,                 \
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,                 \
      const Scalar<DataVector>& tilde_phi,                                    \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,         \
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,     \
      const Scalar<DataVector>& sqrt_det_spatial_metric,                      \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>&        \
          eos) noexcept;
GENERATE_INSTANTIATIONS(INSTANTIATION,
                        (tmpl::list<PrimitiveRecoverySchemes::KastaunEtAl>,
                         tmpl::list<PrimitiveRecoverySchemes::NewmanHamlin>,
                         tmpl::list<PrimitiveRecoverySchemes::PalenzuelaEtAl>,
                         NewmanThenPalenzuela, KastaunThenNewmanThenPalenzuela),
                        (1, 2))
#undef INSTANTIATION
#undef THERMO_DIM
#undef RECOVERY
}  // namespace grmhd::ValenciaDivClean::subcell
