// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/StandingWave.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace ScalarWave::Solutions {

template <size_t Dim>
StandingWave<Dim>::StandingWave(std::array<double, Dim> wave_vector,
                                std::array<double, Dim> center,
                                const double amplitude)
    : wave_vector_(std::move(wave_vector)),
      center_(std::move(center)),
      amplitude_(amplitude),
      omega_(magnitude(wave_vector_)) {}

template <size_t Dim>
std::unique_ptr<evolution::initial_data::InitialData>
StandingWave<Dim>::get_clone() const {
  return std::make_unique<StandingWave<Dim>>(*this);
}

template <size_t Dim>
StandingWave<Dim>::StandingWave(CkMigrateMessage* msg) : InitialData(msg) {}

template <size_t Dim>
tuples::TaggedTuple<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>
StandingWave<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x, const double t,
    const tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>> /*meta*/) const {
  // u = k . (x - x0)
  auto u = make_with_value<DataVector>(get<0>(x), 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    u += gsl::at(wave_vector_, d) * (x.get(d) - gsl::at(center_, d));
  }

  const DataVector sin_u = sin(u);
  const DataVector cos_u = cos(u);
  const double cos_wt = cos(omega_ * t);
  const double sin_wt = sin(omega_ * t);

  // Psi = A sin(u) cos(wt)
  Scalar<DataVector> psi{amplitude_ * sin_u * cos_wt};

  // Pi = -dt Psi = A w sin(u) sin(wt)
  Scalar<DataVector> pi{amplitude_ * omega_ * sin_u * sin_wt};

  // Phi_i = di Psi = A k_i cos(u) cos(wt)
  auto phi = make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(x, 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    phi.get(d) = amplitude_ * gsl::at(wave_vector_, d) * cos_u * cos_wt;
  }

  return {std::move(psi), std::move(pi), std::move(phi)};
}

template <size_t Dim>
tuples::TaggedTuple<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                    ::Tags::dt<Tags::Phi<Dim>>>
StandingWave<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x, const double t,
    const tmpl::list<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                     ::Tags::dt<Tags::Phi<Dim>>> /*meta*/) const {
  // u = k . (x - x0)
  auto u = make_with_value<DataVector>(get<0>(x), 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    u += gsl::at(wave_vector_, d) * (x.get(d) - gsl::at(center_, d));
  }

  const DataVector sin_u = sin(u);
  const DataVector cos_u = cos(u);
  const double cos_wt = cos(omega_ * t);
  const double sin_wt = sin(omega_ * t);

  // dt Psi = -A w sin(u) sin(wt)
  Scalar<DataVector> dt_psi{-amplitude_ * omega_ * sin_u * sin_wt};

  // dt Pi = -dt^2 Psi = A w^2 sin(u) cos(wt)
  Scalar<DataVector> dt_pi{amplitude_ * omega_ * omega_ * sin_u * cos_wt};

  // dt Phi_i = di(dt Psi) = -A w k_i cos(u) sin(wt)
  auto dt_phi =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(x, 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    dt_phi.get(d) =
        -amplitude_ * omega_ * gsl::at(wave_vector_, d) * cos_u * sin_wt;
  }

  return {std::move(dt_psi), std::move(dt_pi), std::move(dt_phi)};
}

template <size_t Dim>
void StandingWave<Dim>::pup(PUP::er& p) {
  InitialData::pup(p);
  p | wave_vector_;
  p | center_;
  p | amplitude_;
  p | omega_;
}

template <size_t Dim>
bool operator==(const StandingWave<Dim>& lhs, const StandingWave<Dim>& rhs) {
  return (lhs.wave_vector_ == rhs.wave_vector_) and
         (lhs.center_ == rhs.center_) and (lhs.amplitude_ == rhs.amplitude_) and
         (lhs.omega_ == rhs.omega_);
}

template <size_t Dim>
bool operator!=(const StandingWave<Dim>& lhs, const StandingWave<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
PUP::able::PUP_ID StandingWave<Dim>::my_PUP_ID = 0;  // NOLINT
}  // namespace ScalarWave::Solutions

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                      \
  template class ScalarWave::Solutions::StandingWave<DIM(data)>;  \
  template bool ScalarWave::Solutions::operator==(                \
      const ScalarWave::Solutions::StandingWave<DIM(data)>& lhs,  \
      const ScalarWave::Solutions::StandingWave<DIM(data)>& rhs); \
  template bool ScalarWave::Solutions::operator!=(                \
      const ScalarWave::Solutions::StandingWave<DIM(data)>& lhs,  \
      const ScalarWave::Solutions::StandingWave<DIM(data)>& rhs);
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
