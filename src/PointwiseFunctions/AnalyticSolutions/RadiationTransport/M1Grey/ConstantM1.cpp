// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey/ConstantM1.hpp"

#include <algorithm>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <cmath>
#include <numeric>

#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma:  no_include "DataStructures/Tensor/TypeAliases.hpp"
// IWYU pragma: no_forward_declare Tensor

namespace RadiationTransport::M1Grey::Solutions {

ConstantM1::ConstantM1(const std::array<double, 3>& mean_velocity,
                       const double comoving_energy_density) noexcept
    :  // clang-tidy: do not std::move trivial types.
      mean_velocity_(std::move(mean_velocity)),  // NOLINT
      comoving_energy_density_(comoving_energy_density) {}

void ConstantM1::pup(PUP::er& p) noexcept {
  p | mean_velocity_;
  p | comoving_energy_density_;
  p | background_spacetime_;
}

// Variables templated on neutrino species.
template <typename NeutrinoSpecies>
tuples::TaggedTuple<
    RadiationTransport::M1Grey::Tags::TildeE<Frame::Inertial, NeutrinoSpecies>>
ConstantM1::variables(const tnsr::I<DataVector, 3>& x, double /*t*/,
                      tmpl::list<RadiationTransport::M1Grey::Tags::TildeE<
                          Frame::Inertial, NeutrinoSpecies>> /*meta*/) const
    noexcept {
  const double W_sqr =
      1. /
      (1. - std::inner_product(mean_velocity_.begin(), mean_velocity_.end(),
                               mean_velocity_.begin(), 0.));
  return {Scalar<DataVector>{DataVector(
      get<0>(x).size(), comoving_energy_density_ / 3. * (4. * W_sqr - 1.))}};
}

template <typename NeutrinoSpecies>
tuples::TaggedTuple<
    RadiationTransport::M1Grey::Tags::TildeS<Frame::Inertial, NeutrinoSpecies>>
ConstantM1::variables(const tnsr::I<DataVector, 3>& x, double /*t*/,
                      tmpl::list<RadiationTransport::M1Grey::Tags::TildeS<
                          Frame::Inertial, NeutrinoSpecies>> /*meta*/) const
    noexcept {
  const double W_sqr =
      1. /
      (1. - std::inner_product(mean_velocity_.begin(), mean_velocity_.end(),
                               mean_velocity_.begin(), 0.));
  const double prefactor = 4. / 3. * comoving_energy_density_ * W_sqr;
  auto result =
      make_with_value<tnsr::i<DataVector, 3>>(x, mean_velocity_[0] * prefactor);
  get<1>(result) = mean_velocity_[1] * prefactor;
  get<2>(result) = mean_velocity_[2] * prefactor;
  return {std::move(result)};
}

template <typename NeutrinoSpecies>
tuples::TaggedTuple<
    RadiationTransport::M1Grey::Tags::GreyEmissivity<NeutrinoSpecies>>
ConstantM1::variables(
    const tnsr::I<DataVector, 3>& x, double /*t*/,
    tmpl::list<RadiationTransport::M1Grey::Tags::GreyEmissivity<
        NeutrinoSpecies>> /*meta*/) const noexcept {
  return {Scalar<DataVector>{DataVector(get<0>(x).size(), 0.)}};
}

template <typename NeutrinoSpecies>
tuples::TaggedTuple<
    RadiationTransport::M1Grey::Tags::GreyAbsorptionOpacity<NeutrinoSpecies>>
ConstantM1::variables(
    const tnsr::I<DataVector, 3>& x, double /*t*/,
    tmpl::list<RadiationTransport::M1Grey::Tags::GreyAbsorptionOpacity<
        NeutrinoSpecies>> /*meta*/) const noexcept {
  return {Scalar<DataVector>{DataVector(get<0>(x).size(), 0.)}};
}

template <typename NeutrinoSpecies>
tuples::TaggedTuple<
    RadiationTransport::M1Grey::Tags::GreyScatteringOpacity<NeutrinoSpecies>>
ConstantM1::variables(
    const tnsr::I<DataVector, 3>& x, double /*t*/,
    tmpl::list<RadiationTransport::M1Grey::Tags::GreyScatteringOpacity<
        NeutrinoSpecies>> /*meta*/) const noexcept {
  return {Scalar<DataVector>{DataVector(get<0>(x).size(), 0.)}};
}

// Variables not templated on neutrino species.
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataVector>>
ConstantM1::variables(
    const tnsr::I<DataVector, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::LorentzFactor<DataVector>> /*meta*/) const
    noexcept {
  const double W =
      1. /
      sqrt(1. - std::inner_product(mean_velocity_.begin(), mean_velocity_.end(),
                                   mean_velocity_.begin(), 0.));
  return {Scalar<DataVector>{DataVector(get<0>(x).size(), W)}};
}

tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataVector, 3>>
ConstantM1::variables(
    const tnsr::I<DataVector, 3>& x, double /*t*/,
    tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3>> /*meta*/) const
    noexcept {
  auto result = make_with_value<tnsr::I<DataVector, 3>>(x, mean_velocity_[0]);
  get<1>(result) = mean_velocity_[1];
  get<2>(result) = mean_velocity_[2];
  return {std::move(result)};
}

bool operator==(const ConstantM1& lhs, const ConstantM1& rhs) noexcept {
  return lhs.mean_velocity_ == rhs.mean_velocity_ and
         lhs.comoving_energy_density_ == rhs.comoving_energy_density_ and
         lhs.background_spacetime_ == rhs.background_spacetime_;
}

bool operator!=(const ConstantM1& lhs, const ConstantM1& rhs) noexcept {
  return not(lhs == rhs);
}

#define TAG(data) BOOST_PP_TUPLE_ELEM(0, data)
#define NTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define EBIN(data) BOOST_PP_TUPLE_ELEM(2, data)
#define GENERATE_LIST(z, n, _) BOOST_PP_COMMA_IF(n) n

#define INSTANTIATE_M1_FUNCTION_WITH_FRAME(_, data)                           \
  template tuples::TaggedTuple<TAG(data) < Frame::Inertial,                   \
                               NTYPE(data) < EBIN(data)>>>                    \
      ConstantM1::variables(                                                  \
          const tnsr::I<DataVector, 3>& x, double t,                          \
          tmpl::list<TAG(data) < Frame::Inertial, NTYPE(data) < EBIN(data)>>> \
          /*meta*/) const noexcept;

#define temp_list \
  (BOOST_PP_REPEAT(MAX_NUMBER_OF_NEUTRINO_ENERGY_BINS, GENERATE_LIST, _))

GENERATE_INSTANTIATIONS(INSTANTIATE_M1_FUNCTION_WITH_FRAME,
                        (RadiationTransport::M1Grey::Tags::TildeE,
                         RadiationTransport::M1Grey::Tags::TildeS),
                        (neutrinos::ElectronNeutrinos,
                         neutrinos::ElectronAntiNeutrinos,
                         neutrinos::HeavyLeptonNeutrinos),
                        temp_list)

#undef temp_list
#undef INSTANTIATE_M1_FUNCTION_WITH_FRAME
#undef TAG
#undef NTYPE
#undef EBIN
#undef GENERATE_LIST

#define TAG(data) BOOST_PP_TUPLE_ELEM(0, data)
#define NTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define EBIN(data) BOOST_PP_TUPLE_ELEM(2, data)
#define GENERATE_LIST(z, n, _) BOOST_PP_COMMA_IF(n) n

#define INSTANTIATE_M1_FUNCTION(_, data)                                       \
  template tuples::TaggedTuple<TAG(data) < NTYPE(data) < EBIN(data)>>>         \
      ConstantM1::variables(const tnsr::I<DataVector, 3>& x, double t,         \
                            tmpl::list<TAG(data) < NTYPE(data) < EBIN(data)>>> \
                            /*meta*/) const noexcept;

#define temp_list \
  (BOOST_PP_REPEAT(MAX_NUMBER_OF_NEUTRINO_ENERGY_BINS, GENERATE_LIST, _))

GENERATE_INSTANTIATIONS(
    INSTANTIATE_M1_FUNCTION,
    (RadiationTransport::M1Grey::Tags::GreyEmissivity,
     RadiationTransport::M1Grey::Tags::GreyAbsorptionOpacity,
     RadiationTransport::M1Grey::Tags::GreyScatteringOpacity),
    (neutrinos::ElectronNeutrinos, neutrinos::ElectronAntiNeutrinos,
     neutrinos::HeavyLeptonNeutrinos),
    temp_list)

#undef INSTANTIATE_M1_FUNCTION
#undef temp_list
#undef TAG
#undef NTYPE
#undef EBIN
#undef GENERATE_LIST

}  // namespace RadiationTransport::M1Grey::Solutions
