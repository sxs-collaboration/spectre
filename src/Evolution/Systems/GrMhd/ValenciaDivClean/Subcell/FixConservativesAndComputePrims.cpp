// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/FixConservativesAndComputePrims.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::subcell {
template <typename OrderedListOfRecoverySchemes>
template <size_t ThermodynamicDim>
void FixConservativesAndComputePrims<OrderedListOfRecoverySchemes>::apply(
    const gsl::not_null<bool*> needed_fixing,
    const gsl::not_null<typename System::variables_tag::type*>
        conserved_vars_ptr,
    const gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*>
        primitive_vars_ptr,
    const grmhd::ValenciaDivClean::FixConservatives& fix_conservatives,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric) noexcept {
  *needed_fixing = fix_conservatives(
      make_not_null(&get<Tags::TildeD>(*conserved_vars_ptr)),
      make_not_null(&get<Tags::TildeTau>(*conserved_vars_ptr)),
      make_not_null(&get<Tags::TildeS<Frame::Inertial>>(*conserved_vars_ptr)),
      get<Tags::TildeB<Frame::Inertial>>(*conserved_vars_ptr), spatial_metric,
      inv_spatial_metric, sqrt_det_spatial_metric);
  grmhd::ValenciaDivClean::
      PrimitiveFromConservative<OrderedListOfRecoverySchemes, true>::apply(
          make_not_null(&get<hydro::Tags::RestMassDensity<DataVector>>(
              *primitive_vars_ptr)),
          make_not_null(&get<hydro::Tags::SpecificInternalEnergy<DataVector>>(
              *primitive_vars_ptr)),
          make_not_null(&get<hydro::Tags::SpatialVelocity<DataVector, 3>>(
              *primitive_vars_ptr)),
          make_not_null(&get<hydro::Tags::MagneticField<DataVector, 3>>(
              *primitive_vars_ptr)),
          make_not_null(&get<hydro::Tags::DivergenceCleaningField<DataVector>>(
              *primitive_vars_ptr)),
          make_not_null(&get<hydro::Tags::LorentzFactor<DataVector>>(
              *primitive_vars_ptr)),
          make_not_null(
              &get<hydro::Tags::Pressure<DataVector>>(*primitive_vars_ptr)),
          make_not_null(&get<hydro::Tags::SpecificEnthalpy<DataVector>>(
              *primitive_vars_ptr)),
          get<Tags::TildeD>(*conserved_vars_ptr),
          get<Tags::TildeTau>(*conserved_vars_ptr),
          get<Tags::TildeS<Frame::Inertial>>(*conserved_vars_ptr),
          get<Tags::TildeB<Frame::Inertial>>(*conserved_vars_ptr),
          get<Tags::TildePhi>(*conserved_vars_ptr), spatial_metric,
          inv_spatial_metric, sqrt_det_spatial_metric, eos);
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

#define INSTANTIATION(r, data)                                              \
  template void                                                             \
  FixConservativesAndComputePrims<RECOVERY(data)>::apply<THERMO_DIM(data)>( \
      const gsl::not_null<bool*> needed_fixing,                             \
      const gsl::not_null<typename System::variables_tag::type*>            \
          conserved_vars_ptr,                                               \
      const gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*>        \
          primitive_vars_ptr,                                               \
      const grmhd::ValenciaDivClean::FixConservatives& fix_conservatives,   \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>& eos, \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,       \
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,   \
      const Scalar<DataVector>& sqrt_det_spatial_metric) noexcept;

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
