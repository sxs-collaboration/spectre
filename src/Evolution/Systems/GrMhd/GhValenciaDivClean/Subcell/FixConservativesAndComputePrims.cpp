// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Subcell/FixConservativesAndComputePrims.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FixConservatives.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/KastaunEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/NewmanHamlin.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PalenzuelaEtAl.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/PrimitiveFromConservative.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::GhValenciaDivClean::subcell {
template <typename OrderedListOfRecoverySchemes>
template <size_t ThermodynamicDim>
void FixConservativesAndComputePrims<OrderedListOfRecoverySchemes>::apply(
    const gsl::not_null<bool*> needed_fixing,
    const gsl::not_null<typename System::variables_tag::type*>
        conserved_vars_ptr,
    const gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*>
        primitive_vars_ptr,
    const grmhd::ValenciaDivClean::FixConservatives& fix_conservatives,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos) {
  // Compute the spatial metric, inverse spatial metric, and sqrt{det{spatial
  // metric}}. Storing the allocation or the result in the DataBox might
  // actually be useful here, but unclear. We will need to profile.
  Variables<
      tmpl::list<gr::Tags::SpatialMetric<3>, gr::Tags::InverseSpatialMetric<3>,
                 gr::Tags::SqrtDetSpatialMetric<>>>
      temp_buffer{conserved_vars_ptr->number_of_grid_points()};
  auto& spatial_metric = get<gr::Tags::SpatialMetric<3>>(temp_buffer);
  gr::spatial_metric(make_not_null(&spatial_metric),
                     get<gr::Tags::SpacetimeMetric<3>>(*conserved_vars_ptr));
  auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<3>>(temp_buffer);
  auto& sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<>>(temp_buffer);
  determinant_and_inverse(make_not_null(&sqrt_det_spatial_metric),
                          make_not_null(&inverse_spatial_metric),
                          spatial_metric);
  get(sqrt_det_spatial_metric) = sqrt(get(sqrt_det_spatial_metric));

  *needed_fixing = fix_conservatives(
      make_not_null(&get<ValenciaDivClean::Tags::TildeD>(*conserved_vars_ptr)),
      make_not_null(
          &get<ValenciaDivClean::Tags::TildeTau>(*conserved_vars_ptr)),
      make_not_null(&get<ValenciaDivClean::Tags::TildeS<Frame::Inertial>>(
          *conserved_vars_ptr)),
      get<ValenciaDivClean::Tags::TildeB<Frame::Inertial>>(*conserved_vars_ptr),
      spatial_metric, inverse_spatial_metric, sqrt_det_spatial_metric);
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
          get<ValenciaDivClean::Tags::TildeD>(*conserved_vars_ptr),
          get<ValenciaDivClean::Tags::TildeTau>(*conserved_vars_ptr),
          get<ValenciaDivClean::Tags::TildeS<Frame::Inertial>>(
              *conserved_vars_ptr),
          get<ValenciaDivClean::Tags::TildeB<Frame::Inertial>>(
              *conserved_vars_ptr),
          get<ValenciaDivClean::Tags::TildePhi>(*conserved_vars_ptr),
          spatial_metric, inverse_spatial_metric, sqrt_det_spatial_metric, eos);
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

#define INSTANTIATION(r, data)                                              \
  template void                                                             \
  FixConservativesAndComputePrims<RECOVERY(data)>::apply<THERMO_DIM(data)>( \
      const gsl::not_null<bool*> needed_fixing,                             \
      const gsl::not_null<typename System::variables_tag::type*>            \
          conserved_vars_ptr,                                               \
      const gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*>        \
          primitive_vars_ptr,                                               \
      const grmhd::ValenciaDivClean::FixConservatives& fix_conservatives,   \
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
