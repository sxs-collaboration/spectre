// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/RadiationTransport/M1Grey/HomogeneousSphere.hpp"

#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Math.hpp"

namespace RadiationTransport::M1Grey::AnalyticData {
HomogeneousSphere::HomogeneousSphere(const double radius,
                                     const double emissivity_and_opacity,
                                     const double outer_opacity,
                                     const double boundary_roundness)
    : radius_(radius),
      emissivity_and_opacity_(emissivity_and_opacity),
      outer_opacity_(outer_opacity),
      boundary_roundness_(boundary_roundness) {}

namespace {

// This function returns the radius squared of the 3-dimensional
// position vector.
DataVector radius_squared(const tnsr::I<DataVector, 3>& x) {
  return square(get<0>(x)) + square(get<1>(x)) + square(get<2>(x));
}
}  // namespace

namespace {
// This function is used to round the edges of the homogeneous sphere, as
// opposed to a pure, rectangular step function.
Scalar<DataVector> rounded_step_function(const DataVector& x,
                                         const double inner_value,
                                         const double outer_value,
                                         const double sphere_radius,
                                         const double boundary_roundness) {
  return Scalar<DataVector>{
      (inner_value - outer_value) / M_PI *
          atan((x - sphere_radius) / -boundary_roundness) +
      0.5 * (inner_value + outer_value)};
}
}  // namespace

template <typename NeutrinoSpecies>
auto HomogeneousSphere::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<RadiationTransport::M1Grey::Tags::TildeE<
        Frame::Inertial, NeutrinoSpecies>> /*meta*/) const
    -> tuples::TaggedTuple<RadiationTransport::M1Grey::Tags::TildeE<
        Frame::Inertial, NeutrinoSpecies>> {
  const DataVector r = sqrt(radius_squared(x));

  const double inner_energy = 1.0;
  const double outer_energy = 1.0e-12;

  return rounded_step_function(r, inner_energy, outer_energy, radius_,
                               boundary_roundness_);
}

template <typename NeutrinoSpecies>
auto HomogeneousSphere::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<RadiationTransport::M1Grey::Tags::TildeS<
        Frame::Inertial, NeutrinoSpecies>> /*meta*/) const
    -> tuples::TaggedTuple<RadiationTransport::M1Grey::Tags::TildeS<
        Frame::Inertial, NeutrinoSpecies>> {
  return {make_with_value<tnsr::i<DataVector, 3, Frame::Inertial>>(x, 0.0)};
}

template <typename NeutrinoSpecies>
auto HomogeneousSphere::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<RadiationTransport::M1Grey::Tags::GreyEmissivity<
        NeutrinoSpecies>> /*meta*/) const
    -> tuples::TaggedTuple<
        RadiationTransport::M1Grey::Tags::GreyEmissivity<NeutrinoSpecies>> {
  const DataVector r = sqrt(radius_squared(x));

  const double inner_emissivity = emissivity_and_opacity_;
  const double outer_emissivity = 0.0;

  return rounded_step_function(r, inner_emissivity, outer_emissivity, radius_,
                               boundary_roundness_);
}

template <typename NeutrinoSpecies>
auto HomogeneousSphere::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<RadiationTransport::M1Grey::Tags::GreyAbsorptionOpacity<
        NeutrinoSpecies>> /*meta*/) const
    -> tuples::TaggedTuple<RadiationTransport::M1Grey::Tags::
                               GreyAbsorptionOpacity<NeutrinoSpecies>> {
  const DataVector r = sqrt(radius_squared(x));

  return rounded_step_function(r, emissivity_and_opacity_, outer_opacity_,
                               radius_, boundary_roundness_);
}

template <typename NeutrinoSpecies>
auto HomogeneousSphere::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<RadiationTransport::M1Grey::Tags::GreyScatteringOpacity<
        NeutrinoSpecies>> /*meta*/) const
    -> tuples::TaggedTuple<RadiationTransport::M1Grey::Tags::
                               GreyScatteringOpacity<NeutrinoSpecies>> {
  return {make_with_value<Scalar<DataVector>>(x, 0.0)};
}

auto HomogeneousSphere::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataVector>> /*meta*/)
    -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataVector>> {
  return {make_with_value<Scalar<DataVector>>(x, 1.0)};
}

auto HomogeneousSphere::variables(
    const tnsr::I<DataVector, 3>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3>> /*meta*/)
    -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataVector, 3>> {
  return {make_with_value<tnsr::I<DataVector, 3, Frame::Inertial>>(x, 0.0)};
}

std::unique_ptr<evolution::initial_data::InitialData>
HomogeneousSphere::get_clone() const {
  return std::make_unique<HomogeneousSphere>(*this);
}

void HomogeneousSphere::pup(PUP::er& p) {
  evolution::initial_data::InitialData::pup(p);
  p | radius_;
  p | emissivity_and_opacity_;
  p | outer_opacity_;
  p | boundary_roundness_;
}
#ifndef __CUDA_ARCH__
// NOLINTNEXTLINE
PUP::able::PUP_ID HomogeneousSphere::my_PUP_ID = 0;
#endif  // __CUDA_ARCH__

bool operator!=(const HomogeneousSphere& lhs, const HomogeneousSphere& rhs) {
  return not(lhs == rhs);
}

#define DERIVED_CLASSES (HomogeneousSphere)

#define DERIVED(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)
#define NTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)
#define EBIN(data) BOOST_PP_TUPLE_ELEM(3, data)
#define GENERATE_LIST(z, n, _) BOOST_PP_COMMA_IF(n) n

#define INSTANTIATE_M1_FUNCTION_WITH_FRAME(_, data)                            \
  template tuples::TaggedTuple<TAG(data) < Frame::Inertial,                    \
                               NTYPE(data) < EBIN(data)> >>                    \
      DERIVED(data)::variables(                                                \
          const tnsr::I<DataVector, 3>& x,                                     \
          tmpl::list<TAG(data) < Frame::Inertial, NTYPE(data) < EBIN(data)> >> \
          /*meta*/) const;

#define TEMP_LIST \
  (BOOST_PP_REPEAT(MAX_NUMBER_OF_NEUTRINO_ENERGY_BINS, GENERATE_LIST, _))

GENERATE_INSTANTIATIONS(INSTANTIATE_M1_FUNCTION_WITH_FRAME, DERIVED_CLASSES,
                        (RadiationTransport::M1Grey::Tags::TildeE,
                         RadiationTransport::M1Grey::Tags::TildeS),
                        (neutrinos::ElectronNeutrinos,
                         neutrinos::ElectronAntiNeutrinos,
                         neutrinos::HeavyLeptonNeutrinos),
                        TEMP_LIST)

#undef TEMP_LIST
#undef INSTANTIATE_M1_FUNCTION_WITH_FRAME
#undef DERIVED
#undef TAG
#undef NTYPE
#undef EBIN
#undef GENERATE_LIST

#define DERIVED(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)
#define NTYPE(data) BOOST_PP_TUPLE_ELEM(2, data)
#define EBIN(data) BOOST_PP_TUPLE_ELEM(3, data)
#define GENERATE_LIST(z, n, _) BOOST_PP_COMMA_IF(n) n

#define INSTANTIATE_M1_FUNCTION(_, data)                                \
  template tuples::TaggedTuple<TAG(data) < NTYPE(data) < EBIN(data)> >> \
      DERIVED(data)::variables(                                         \
          const tnsr::I<DataVector, 3>& x,                              \
          tmpl::list<TAG(data) < NTYPE(data) < EBIN(data)> >>           \
          /*meta*/) const;

#define TEMP_LIST \
  (BOOST_PP_REPEAT(MAX_NUMBER_OF_NEUTRINO_ENERGY_BINS, GENERATE_LIST, _))

GENERATE_INSTANTIATIONS(
    INSTANTIATE_M1_FUNCTION, DERIVED_CLASSES,
    (RadiationTransport::M1Grey::Tags::GreyEmissivity,
     RadiationTransport::M1Grey::Tags::GreyAbsorptionOpacity,
     RadiationTransport::M1Grey::Tags::GreyScatteringOpacity),
    (neutrinos::ElectronNeutrinos, neutrinos::ElectronAntiNeutrinos,
     neutrinos::HeavyLeptonNeutrinos),
    TEMP_LIST)

#undef INSTANTIATE_M1_FUNCTION
#undef TEMP_LIST
#undef DERIVED
#undef TAG
#undef NTYPE
#undef EBIN
#undef GENERATE_LIST

}  // namespace RadiationTransport::M1Grey::AnalyticData
