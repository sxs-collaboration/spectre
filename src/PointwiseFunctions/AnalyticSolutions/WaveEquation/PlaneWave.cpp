// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"

#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"

namespace ScalarWave {
namespace Solutions {

template <size_t Dim>
PlaneWave<Dim>::PlaneWave(std::array<double, Dim> wave_vector,
                          std::array<double, Dim> center,
                          std::unique_ptr<MathFunction<1>> profile) noexcept
    : wave_vector_(std::move(wave_vector)),
      center_(std::move(center)),
      profile_(std::move(profile)),
      omega_(magnitude(wave_vector_)) {}

template <size_t Dim>
template <typename T>
Scalar<T> PlaneWave<Dim>::psi(const tnsr::I<T, Dim>& x, const double t) const
    noexcept {
  return Scalar<T>(profile_->operator()(u(x, t)));
}

template <size_t Dim>
template <typename T>
Scalar<T> PlaneWave<Dim>::dpsi_dt(const tnsr::I<T, Dim>& x,
                                  const double t) const noexcept {
  return Scalar<T>(-omega_ * profile_->first_deriv(u(x, t)));
}

template <size_t Dim>
template <typename T>
tnsr::i<T, Dim> PlaneWave<Dim>::dpsi_dx(const tnsr::I<T, Dim>& x,
                                        const double t) const noexcept {
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
                                    const double t) const noexcept {
  return Scalar<T>(square(omega_) * profile_->second_deriv(u(x, t)));
}

template <size_t Dim>
template <typename T>
tnsr::i<T, Dim> PlaneWave<Dim>::d2psi_dtdx(const tnsr::I<T, Dim>& x,
                                           const double t) const noexcept {
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
                                            const double t) const noexcept {
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
template <typename T>
T PlaneWave<Dim>::u(const tnsr::I<T, Dim>& x, const double t) const noexcept {
  auto result = make_with_value<T>(x, -omega_ * t);
  for (size_t d = 0; d < Dim; ++d) {
    result += gsl::at(wave_vector_, d) * (x.get(d) - gsl::at(center_, d));
  }
  return result;
}

}  // namespace Solutions
}  // namespace ScalarWave

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                              \
  template Scalar<DTYPE(data)>                                            \
  ScalarWave::Solutions::PlaneWave<DIM(data)>::psi(                       \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double t) const noexcept; \
  template Scalar<DTYPE(data)>                                            \
  ScalarWave::Solutions::PlaneWave<DIM(data)>::dpsi_dt(                   \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double t) const noexcept; \
  template tnsr::i<DTYPE(data), DIM(data)>                                \
  ScalarWave::Solutions::PlaneWave<DIM(data)>::dpsi_dx(                   \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double t)           \
      const noexcept;                                                     \
  template Scalar<DTYPE(data)>                                            \
  ScalarWave::Solutions::PlaneWave<DIM(data)>::d2psi_dt2(                 \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double t)           \
      const noexcept;                                                     \
  template tnsr::i<DTYPE(data), DIM(data)>                                \
  ScalarWave::Solutions::PlaneWave<DIM(data)>::d2psi_dtdx(                \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double t)           \
      const noexcept;                                                     \
  template tnsr::ii<DTYPE(data), DIM(data)>                               \
  ScalarWave::Solutions::PlaneWave<DIM(data)>::d2psi_dxdx(                \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double t)           \
      const noexcept;                                                     \
  template DTYPE(data) ScalarWave::Solutions::PlaneWave<DIM(data)>::u(    \
      const tnsr::I<DTYPE(data), DIM(data)>& x, const double t)           \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))

#undef DIM
#undef DTYPE
#undef INSTANTIATE

template class ScalarWave::Solutions::PlaneWave<1>;
template class ScalarWave::Solutions::PlaneWave<2>;
template class ScalarWave::Solutions::PlaneWave<3>;
/// \endcond
