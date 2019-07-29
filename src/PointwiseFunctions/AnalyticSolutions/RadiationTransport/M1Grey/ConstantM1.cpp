// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/RadiationTransport/M1Grey/ConstantM1.hpp"

#include <algorithm>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <cmath>
#include <numeric>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"     // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "ParallelBackend/PupStlCpp11.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

// IWYU pragma:  no_include "DataStructures/Tensor/TypeAliases.hpp"
// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace RadiationTransport {
namespace M1Grey {
namespace Solutions {

ConstantM1::ConstantM1(
    MeanVelocity::type mean_velocity,
    ComovingEnergyDensity::type comoving_energy_density) noexcept
    :  // clang-tidy: do not std::move trivial types.
      mean_velocity_(std::move(mean_velocity)),  // NOLINT
      comoving_energy_density_(comoving_energy_density) {}

void ConstantM1::pup(PUP::er& p) noexcept {
  p | mean_velocity_;
  p | comoving_energy_density_;
  p | background_spacetime_;
}

// M1 variables.
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
      make_with_value<db::item_type<RadiationTransport::M1Grey::Tags::TildeS<
          Frame::Inertial, NeutrinoSpecies>>>(x, mean_velocity_[0] * prefactor);
  get<1>(result) = mean_velocity_[1] * prefactor;
  get<2>(result) = mean_velocity_[2] * prefactor;
  return {std::move(result)};
}

// Hydro variables.
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

tuples::TaggedTuple<
    hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>
ConstantM1::variables(const tnsr::I<DataVector, 3>& x, double /*t*/,
                      tmpl::list<hydro::Tags::SpatialVelocity<
                          DataVector, 3, Frame::Inertial>> /*meta*/) const
    noexcept {
  auto result = make_with_value<db::item_type<
      hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>>>(
      x, mean_velocity_[0]);
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

#define INSTANTIATE_M1_FUNCTION(_, data)                                      \
  template tuples::TaggedTuple<TAG(data) < Frame::Inertial,                   \
                               NTYPE(data) < EBIN(data)>>>                    \
      ConstantM1::variables(                                                  \
          const tnsr::I<DataVector, 3>& x, double t,                          \
          tmpl::list<TAG(data) < Frame::Inertial, NTYPE(data) < EBIN(data)>>> \
          /*meta*/) const noexcept;

#define temp_list \
  (BOOST_PP_REPEAT(MAX_NUMBER_OF_NEUTRINO_ENERGY_BINS, GENERATE_LIST, _))

GENERATE_INSTANTIATIONS(INSTANTIATE_M1_FUNCTION,
                        (RadiationTransport::M1Grey::Tags::TildeE,
                         RadiationTransport::M1Grey::Tags::TildeS),
                        (neutrinos::ElectronNeutrinos,
                         neutrinos::ElectronAntiNeutrinos,
                         neutrinos::HeavyLeptonNeutrinos),
                        temp_list)

#undef TAG
#undef NTYPE
#undef EBIN
#undef GENERATE_LIST

}  // namespace Solutions
}  // namespace M1Grey
}  // namespace RadiationTransport
/// \endcond
