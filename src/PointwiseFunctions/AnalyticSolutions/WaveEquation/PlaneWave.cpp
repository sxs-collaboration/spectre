// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"

#include <algorithm>

#include "DataStructures/DataBox/Prefixes.hpp"    // IWYU pragma: keep
#include "DataStructures/Variables.hpp"           // IWYU pragma: keep
#include "Evolution/Systems/ScalarWave/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace ScalarWave::Solutions {

template <size_t Dim>
PlaneWave<Dim>::PlaneWave(
    std::array<double, Dim> wave_vector, std::array<double, Dim> center,
    std::unique_ptr<MathFunction<1, Frame::Inertial>> profile)
    : wave_vector_(std::move(wave_vector)),
      center_(std::move(center)),
      profile_(std::move(profile)),
      omega_(magnitude(wave_vector_)) {}

template <size_t Dim>
PlaneWave<Dim>::PlaneWave(const PlaneWave& other)
    : evolution::initial_data::InitialData(other),
      wave_vector_(other.wave_vector_),
      center_(other.center_),
      profile_(other.profile_->get_clone()),
      omega_(magnitude(wave_vector_)) {}

template <size_t Dim>
PlaneWave<Dim>& PlaneWave<Dim>::operator=(const PlaneWave& other) {
  wave_vector_ = other.wave_vector_;
  center_ = other.center_;
  omega_ = magnitude(wave_vector_);
  profile_ = other.profile_->get_clone();
  return *this;
}

template <size_t Dim>
PlaneWave<Dim>::PlaneWave(CkMigrateMessage* msg) : InitialData(msg) {}

template <size_t Dim>
template <typename T>
Scalar<T> PlaneWave<Dim>::psi(const tnsr::I<T, Dim>& x, const double t) const {
  return Scalar<T>(profile_->operator()(u(x, t)));
}

template <size_t Dim>
template <typename T>
Scalar<T> PlaneWave<Dim>::dpsi_dt(const tnsr::I<T, Dim>& x,
                                  const double t) const {
  return Scalar<T>(-omega_ * profile_->first_deriv(u(x, t)));
}

template <size_t Dim>
template <typename T>
tnsr::i<T, Dim> PlaneWave<Dim>::dpsi_dx(const tnsr::I<T, Dim>& x,
                                        const double t) const {
  auto result = make_with_value<tnsr::i<T, Dim>>(x, 0.0);
  const auto du = profile_->first_deriv(u(x, t));
  for (size_t i = 0; i < Dim; ++i) {
    result.get(i) = gsl::at(wave_vector_, i) * du;
  }
  return result;
}

template <size_t Dim>
template <typename T>
Scalar<T> PlaneWave<Dim>::d2psi_dt2(const tnsr::I<T, Dim>& x,
                                    const double t) const {
  return Scalar<T>(square(omega_) * profile_->second_deriv(u(x, t)));
}

template <size_t Dim>
template <typename T>
tnsr::i<T, Dim> PlaneWave<Dim>::d2psi_dtdx(const tnsr::I<T, Dim>& x,
                                           const double t) const {
  auto result = make_with_value<tnsr::i<T, Dim>>(x, 0.0);
  const auto d2u = profile_->second_deriv(u(x, t));
  for (size_t i = 0; i < Dim; ++i) {
    result.get(i) = -omega_ * gsl::at(wave_vector_, i) * d2u;
  }
  return result;
}

template <size_t Dim>
template <typename T>
tnsr::ii<T, Dim> PlaneWave<Dim>::d2psi_dxdx(const tnsr::I<T, Dim>& x,
                                            const double t) const {
  auto result = make_with_value<tnsr::ii<T, Dim>>(x, 0.0);
  const auto d2u = profile_->second_deriv(u(x, t));
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      result.get(i, j) =
          gsl::at(wave_vector_, i) * gsl::at(wave_vector_, j) * d2u;
    }
  }
  return result;
}

template <size_t Dim>
tuples::TaggedTuple<Tags::Psi, Tags::Pi, Tags::Phi<Dim>>
PlaneWave<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x, double t,
    const tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>> /*meta*/) const {
  tuples::TaggedTuple<Tags::Psi, Tags::Pi, Tags::Phi<Dim>> variables{
      psi(x, t), dpsi_dt(x, t), dpsi_dx(x, t)};
  get<Tags::Pi>(variables).get() *= -1.0;
  return variables;
}

template <size_t Dim>
tuples::TaggedTuple<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                    ::Tags::dt<Tags::Phi<Dim>>>
PlaneWave<Dim>::variables(
    const tnsr::I<DataVector, Dim>& x, double t,
    const tmpl::list<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                     ::Tags::dt<Tags::Phi<Dim>>> /*meta*/) const {
  tuples::TaggedTuple<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                      ::Tags::dt<Tags::Phi<Dim>>>
      dt_variables{dpsi_dt(x, t), d2psi_dt2(x, t), d2psi_dtdx(x, t)};
  get<::Tags::dt<Tags::Pi>>(dt_variables).get() *= -1.0;
  return dt_variables;
}

template <size_t Dim>
void PlaneWave<Dim>::pup(PUP::er& p) {
  InitialData::pup(p);
  p | wave_vector_;
  p | center_;
  p | profile_;
  p | omega_;
}
template <size_t Dim>
bool operator==(const PlaneWave<Dim>& lhs, const PlaneWave<Dim>& rhs) {
  return (lhs.wave_vector_ == rhs.wave_vector_) and
         (lhs.center_ == rhs.center_) and
         (*(lhs.profile_) == *(rhs.profile_)) and (lhs.omega_ == rhs.omega_);
}

template <size_t Dim>
bool operator!=(const PlaneWave<Dim>& lhs, const PlaneWave<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
template <typename T>
T PlaneWave<Dim>::u(const tnsr::I<T, Dim>& x, const double t) const {
  auto result = make_with_value<T>(x, -omega_ * t);
  for (size_t d = 0; d < Dim; ++d) {
    result += gsl::at(wave_vector_, d) * (x.get(d) - gsl::at(center_, d));
  }
  return result;
}

template <size_t Dim>
PUP::able::PUP_ID PlaneWave<Dim>::my_PUP_ID = 0;
}  // namespace ScalarWave::Solutions

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                   \
  template class ScalarWave::Solutions::PlaneWave<DIM(data)>;  \
  template bool ScalarWave::Solutions::operator==(             \
      const ScalarWave::Solutions::PlaneWave<DIM(data)>& lhs,  \
      const ScalarWave::Solutions::PlaneWave<DIM(data)>& rhs); \
  template bool ScalarWave::Solutions::operator!=(             \
      const ScalarWave::Solutions::PlaneWave<DIM(data)>& lhs,  \
      const ScalarWave::Solutions::PlaneWave<DIM(data)>& rhs);
GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                           \
  template Scalar<DTYPE(data)>                                         \
  ScalarWave::Solutions::PlaneWave<DIM(data)>::psi(                    \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double t) const;       \
  template Scalar<DTYPE(data)>                                         \
  ScalarWave::Solutions::PlaneWave<DIM(data)>::dpsi_dt(                \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double t) const;       \
  template tnsr::i<DTYPE(data), DIM(data)>                             \
  ScalarWave::Solutions::PlaneWave<DIM(data)>::dpsi_dx(                \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double t) const; \
  template Scalar<DTYPE(data)>                                         \
  ScalarWave::Solutions::PlaneWave<DIM(data)>::d2psi_dt2(              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double t) const; \
  template tnsr::i<DTYPE(data), DIM(data)>                             \
  ScalarWave::Solutions::PlaneWave<DIM(data)>::d2psi_dtdx(             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double t) const; \
  template tnsr::ii<DTYPE(data), DIM(data)>                            \
  ScalarWave::Solutions::PlaneWave<DIM(data)>::d2psi_dxdx(             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double t) const; \
  template DTYPE(data) ScalarWave::Solutions::PlaneWave<DIM(data)>::u( \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double t) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))

#undef DIM
#undef DTYPE
#undef INSTANTIATE
