// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticSolutions/EinsteinSolutions/Minkowski.hpp"

#include "Utilities/GenerateInstantiations.hpp"

namespace EinsteinSolutions {
template<size_t Dim>
template<typename T>
Scalar<T> Minkowski<Dim>::lapse(const tnsr::I<T, Dim> &x,
                                const double /*t*/) const noexcept {
  return Scalar<T>(make_with_value<T>(x, 1.));
}

template<size_t Dim>
template<typename T>
Scalar<T> Minkowski<Dim>::dt_lapse(const tnsr::I<T, Dim> &x,
                                   const double /*t*/) const noexcept {
  return Scalar<T>(make_with_value<T>(x, 0.));
}

template<size_t Dim>
template<typename T>
tnsr::i<T, Dim> Minkowski<Dim>::deriv_lapse(const tnsr::I<T, Dim> &x,
                                            const double /*t*/) const noexcept {
  return tnsr::i<T, Dim>(make_with_value<T>(x, 0.));
}

template<size_t Dim>
template<typename T>
Scalar<T> Minkowski<Dim>::sqrt_determinant_of_spatial_metric(
    const tnsr::I<T, Dim> &x, const double /*t*/) const noexcept {
  return Scalar<T>(make_with_value<T>(x, 1.));
}

template<size_t Dim>
template<typename T>
Scalar<T> Minkowski<Dim>::dt_sqrt_determinant_of_spatial_metric(
    const tnsr::I<T, Dim> &x, const double /*t*/) const noexcept {
  return Scalar<T>(make_with_value<T>(x, 0.));
}

template<size_t Dim>
template<typename T>
tnsr::I<T, Dim> Minkowski<Dim>::shift(const tnsr::I<T, Dim> &x,
                                      const double /*t*/) const noexcept {
  return tnsr::I<T, Dim>(make_with_value<T>(x, 0.));
}

template<size_t Dim>
template<typename T>
tnsr::iJ<T, Dim> Minkowski<Dim>::deriv_shift(const tnsr::I<T, Dim> &x,
                                             const double /*t*/) const
noexcept {
  return tnsr::iJ<T, Dim>(make_with_value<T>(x, 0.));
}

template<size_t Dim>
template<typename T>
tnsr::ii<T, Dim> Minkowski<Dim>::spatial_metric(const tnsr::I<T, Dim> &x,
                                                const double /*t*/) const
noexcept {
  tnsr::ii<T, Dim> lower_metric(make_with_value<T>(x, 0.));
  for (size_t i = 0; i < Dim; ++i) {
    lower_metric.get(i, i) = 1.;
  }
  return lower_metric;
}

template<size_t Dim>
template<typename T>
tnsr::ii<T, Dim> Minkowski<Dim>::dt_spatial_metric(const tnsr::I<T, Dim> &x,
                                                   const double /*t*/) const
noexcept {
  return tnsr::ii<T, Dim>(make_with_value<T>(x, 0.));
}

template<size_t Dim>
template<typename T>
tnsr::ijj<T, Dim> Minkowski<Dim>::deriv_spatial_metric(const tnsr::I<T, Dim> &x,
                                                       const double /*t*/) const
noexcept {
  return tnsr::ijj<T, Dim>(make_with_value<T>(x, 0.));
}

template<size_t Dim>
template<typename T>
tnsr::II<T, Dim> Minkowski<Dim>::inverse_spatial_metric(
    const tnsr::I<T, Dim> &x, const double /*t*/
) const noexcept {
  tnsr::II<T, Dim> upper_metric(make_with_value<T>(x, 0.));
  for (size_t i = 0; i < Dim; ++i) {
    upper_metric.get(i, i) = 1.;
  }
  return upper_metric;
}

template<size_t Dim>
template<typename T>
tnsr::ii<T, Dim> Minkowski<Dim>::extrinsic_curvature(const tnsr::I<T, Dim> &x,
                                                     const double /*t*/) const
noexcept {
  return tnsr::ii<T, Dim>(make_with_value<T>(x, 0.));
}
} // namespace EinsteinSolutions

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                   \
  template Scalar<DTYPE(data)> EinsteinSolutions::Minkowski<DIM(data)>::lapse( \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/) const noexcept;  \
  template Scalar<DTYPE(data)>                                                 \
  EinsteinSolutions::Minkowski<DIM(data)>::dt_lapse(                           \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/) const noexcept;  \
  template tnsr::i<DTYPE(data), DIM(data)>                                     \
  EinsteinSolutions::Minkowski<DIM(data)>::deriv_lapse(                        \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/) const noexcept;  \
  template Scalar<DTYPE(data)>                                                 \
  EinsteinSolutions::Minkowski<DIM(data)>::sqrt_determinant_of_spatial_metric( \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/) const noexcept;  \
  template Scalar<DTYPE(data)> EinsteinSolutions::Minkowski<DIM(data)>::       \
      dt_sqrt_determinant_of_spatial_metric(                                   \
          const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/)              \
          const noexcept;                                                      \
  template tnsr::I<DTYPE(data), DIM(data)>                                     \
  EinsteinSolutions::Minkowski<DIM(data)>::shift(                              \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/) const noexcept;  \
  template tnsr::iJ<DTYPE(data), DIM(data)>                                    \
  EinsteinSolutions::Minkowski<DIM(data)>::deriv_shift(                        \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/) const noexcept;  \
  template tnsr::ii<DTYPE(data), DIM(data)>                                    \
  EinsteinSolutions::Minkowski<DIM(data)>::spatial_metric(                     \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/) const noexcept;  \
  template tnsr::ii<DTYPE(data), DIM(data)>                                    \
  EinsteinSolutions::Minkowski<DIM(data)>::dt_spatial_metric(                  \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/) const noexcept;  \
  template tnsr::ijj<DTYPE(data), DIM(data)>                                   \
  EinsteinSolutions::Minkowski<DIM(data)>::deriv_spatial_metric(               \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/) const noexcept;  \
  template tnsr::II<DTYPE(data), DIM(data)>                                    \
  EinsteinSolutions::Minkowski<DIM(data)>::inverse_spatial_metric(             \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/                   \
      ) const noexcept;                                                        \
  template tnsr::ii<DTYPE(data), DIM(data)>                                    \
  EinsteinSolutions::Minkowski<DIM(data)>::extrinsic_curvature(                \
      const tnsr::I<DTYPE(data), DIM(data)>& x, double /*t*/) const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))

#undef DIM
#undef DTYPE
#undef INSTANTIATE
/// \endcond
