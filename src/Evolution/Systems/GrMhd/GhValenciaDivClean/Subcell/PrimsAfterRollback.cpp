// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/PrimsAfterRollback.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::GhValenciaDivClean::subcell {
template <typename OrderedListOfRecoverySchemes>
template <size_t ThermodynamicDim>
void PrimsAfterRollback<OrderedListOfRecoverySchemes>::apply(
    const gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*> prim_vars,
    const bool did_rollback, const Mesh<3>& dg_mesh,
    const Mesh<3>& subcell_mesh, const Scalar<DataVector>& tilde_d,
    const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,
    const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos) {
  if (did_rollback) {
    const size_t num_grid_points = subcell_mesh.number_of_grid_points();
    ASSERT(prim_vars->number_of_grid_points() != num_grid_points,
           "Doing rollback and the prim_vars should not be the size of the "
           "subcell mesh");
    // Project a copy of the pressure from the DG grid to the FD grid to give
    // high-order initial guess for prim recovery.
    const Scalar<DataVector> dg_pressure =
        get<hydro::Tags::Pressure<DataVector>>(*prim_vars);
    prim_vars->initialize(num_grid_points);
    evolution::dg::subcell::fd::project(
        make_not_null(&get(get<hydro::Tags::Pressure<DataVector>>(*prim_vars))),
        get(dg_pressure), dg_mesh, subcell_mesh.extents());

    // Compute the spatial metric, inverse spatial metric, and sqrt{det{spatial
    // metric}} on the subcells since we need these for the prim recovery.
    //
    // Note: This same computation needs to be done in the time derivative
    // computation, so it is more efficient to move the code there in the long
    // term, but that likely requires more refactoring anyway.
    Variables<tmpl::list<gr::Tags::SpatialMetric<3>,
                         gr::Tags::InverseSpatialMetric<3>,
                         gr::Tags::SqrtDetSpatialMetric<>>>
        temp_buffer{num_grid_points};
    auto& spatial_metric = get<gr::Tags::SpatialMetric<3>>(temp_buffer);
    gr::spatial_metric(make_not_null(&spatial_metric), spacetime_metric);
    auto& inverse_spatial_metric =
        get<gr::Tags::InverseSpatialMetric<3>>(temp_buffer);
    auto& sqrt_det_spatial_metric =
        get<gr::Tags::SqrtDetSpatialMetric<>>(temp_buffer);
    determinant_and_inverse(make_not_null(&sqrt_det_spatial_metric),
                            make_not_null(&inverse_spatial_metric),
                            spatial_metric);
    get(sqrt_det_spatial_metric) = sqrt(get(sqrt_det_spatial_metric));

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
            inverse_spatial_metric, sqrt_det_spatial_metric, eos);
  }
}

namespace {
using NewmanThenPalenzuela =
    tmpl::list<ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin,
               ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>;
using KastaunThenNewmanThenPalenzuela =
    tmpl::list<ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl,
               ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin,
               ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>;
}  // namespace

#define RECOVERY(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATION(r, data)                                                \
  template void PrimsAfterRollback<RECOVERY(data)>::apply<THERMO_DIM(data)>(  \
      const gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*>          \
          prim_vars,                                                          \
      bool did_rollback, const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh, \
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau, \
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,                 \
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,                 \
      const Scalar<DataVector>& tilde_phi,                                    \
      const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,       \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& eos);
GENERATE_INSTANTIATIONS(
    INSTANTIATION,
    (tmpl::list<ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>,
     tmpl::list<ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>,
     tmpl::list<ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>,
     NewmanThenPalenzuela, KastaunThenNewmanThenPalenzuela),
    (1, 2))
#undef INSTANTIATION
#undef THERMO_DIM
#undef RECOVERY
}  // namespace grmhd::GhValenciaDivClean::subcell
