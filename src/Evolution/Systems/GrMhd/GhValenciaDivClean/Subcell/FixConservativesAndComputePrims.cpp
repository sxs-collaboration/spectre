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
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/ErrorHandling/Exceptions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::GhValenciaDivClean::subcell {
template <typename OrderedListOfRecoverySchemes>
void FixConservativesAndComputePrims<OrderedListOfRecoverySchemes>::apply(
    const gsl::not_null<bool*> needed_fixing,
    const gsl::not_null<typename System::variables_tag::type*>
        conserved_vars_ptr,
    const gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*>
        primitive_vars_ptr,
    const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_coords,
    const grmhd::ValenciaDivClean::FixConservatives& fix_conservatives,
    const EquationsOfState::EquationOfState<true, 3>& eos,
    const grmhd::ValenciaDivClean::PrimitiveFromConservativeOptions&
        primitive_from_conservative_options) {
  CAPTURE_FOR_ERROR(subcell_coords);
  const auto& cons_vars = *conserved_vars_ptr;
  CAPTURE_FOR_ERROR(cons_vars);
  // Compute the spatial metric, inverse spatial metric, and sqrt{det{spatial
  // metric}}. Storing the allocation or the result in the DataBox might
  // actually be useful here, but unclear. We will need to profile.
  Variables<tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, 3>,
                       gr::Tags::DetSpatialMetric<DataVector>,
                       gr::Tags::SqrtDetSpatialMetric<DataVector>>>
      temp_buffer{conserved_vars_ptr->number_of_grid_points()};
  const tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      make_const_view(
          make_not_null(&spatial_metric.get(i, j)),
          get<gr::Tags::SpacetimeMetric<DataVector, 3>>(*conserved_vars_ptr)
              .get(i + 1, j + 1),
          0, temp_buffer.number_of_grid_points());
    }
  }
  auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(temp_buffer);
  auto& det_spatial_metric =
      get<gr::Tags::DetSpatialMetric<DataVector>>(temp_buffer);
  auto& sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(temp_buffer);
  determinant_and_inverse(make_not_null(&det_spatial_metric),
                          make_not_null(&inverse_spatial_metric),
                          spatial_metric);
  CAPTURE_FOR_ERROR(det_spatial_metric);
  get(sqrt_det_spatial_metric) = sqrt(get(det_spatial_metric));

  *needed_fixing = fix_conservatives(
      make_not_null(&get<ValenciaDivClean::Tags::TildeD>(*conserved_vars_ptr)),
      make_not_null(&get<ValenciaDivClean::Tags::TildeYe>(*conserved_vars_ptr)),
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
          make_not_null(&get<hydro::Tags::ElectronFraction<DataVector>>(
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
          make_not_null(
              &get<hydro::Tags::Temperature<DataVector>>(*primitive_vars_ptr)),
          get<ValenciaDivClean::Tags::TildeD>(*conserved_vars_ptr),
          get<ValenciaDivClean::Tags::TildeYe>(*conserved_vars_ptr),
          get<ValenciaDivClean::Tags::TildeTau>(*conserved_vars_ptr),
          get<ValenciaDivClean::Tags::TildeS<Frame::Inertial>>(
              *conserved_vars_ptr),
          get<ValenciaDivClean::Tags::TildeB<Frame::Inertial>>(
              *conserved_vars_ptr),
          get<ValenciaDivClean::Tags::TildePhi>(*conserved_vars_ptr),
          spatial_metric, inverse_spatial_metric, sqrt_det_spatial_metric, eos,
          primitive_from_conservative_options);
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

#define INSTANTIATION(r, data) \
  template struct FixConservativesAndComputePrims<RECOVERY(data)>;

GENERATE_INSTANTIATIONS(
    INSTANTIATION,
    (tmpl::list<ValenciaDivClean::PrimitiveRecoverySchemes::KastaunEtAl>,
     tmpl::list<ValenciaDivClean::PrimitiveRecoverySchemes::NewmanHamlin>,
     tmpl::list<ValenciaDivClean::PrimitiveRecoverySchemes::PalenzuelaEtAl>,
     NewmanThenPalenzuela, KastaunThenNewmanThenPalenzuela))

#undef INSTANTIATION
#undef RECOVERY
}  // namespace grmhd::GhValenciaDivClean::subcell
