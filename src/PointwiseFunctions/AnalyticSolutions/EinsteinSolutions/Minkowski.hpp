// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/MakeWithValue.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Options/MakeCreatableFromYaml.hpp"
#include "Options/Options.hpp"

namespace EinsteinSolutions {

/*!
 * \ingroup EinsteinSolutions
 * \brief The Minkowski solution for flat space in Dim spatial dimensions.
 *
 * \details Flat space has lapse \f$N(x,t)= 1 \f$, shift \f$N^i(x,t) = 0
 * \f$ and the identity as the spatial metric: \f$g_{ii} = 1 \f$
 */
template <size_t Dim>
class Minkowski {
 public:
  using options = tmpl::list<>;
  static constexpr OptionString_t help{
      "Minkowski solution to Einstein's Equations"};

  Minkowski() = default;
  Minkowski(const Minkowski& /*rhs*/) = delete;
  Minkowski& operator=(const Minkowski& /*rhs*/) = delete;
  Minkowski(Minkowski&& /*rhs*/) noexcept = default;
  Minkowski& operator=(Minkowski&& /*rhs*/) noexcept = default;
  ~Minkowski() = default;

  template <typename T>
  Scalar<T> lapse(const tnsr::I<T, Dim>& x, double /*t*/) const noexcept;

  template <typename T>
  Scalar<T> dt_lapse(const tnsr::I<T, Dim>& x, double /*t*/) const noexcept;

  template <typename T>
  tnsr::i<T, Dim> deriv_lapse(const tnsr::I<T, Dim>& x, double /*t*/) const
      noexcept;

  template <typename T>
  Scalar<T> sqrt_determinant_of_spatial_metric(const tnsr::I<T, Dim>& x,
                                               double /*t*/) const noexcept;

  template <typename T>
  Scalar<T> dt_sqrt_determinant_of_spatial_metric(const tnsr::I<T, Dim>& x,
                                                  double /*t*/) const noexcept;

  template <typename T>
  tnsr::I<T, Dim> shift(const tnsr::I<T, Dim>& x, double /*t*/) const noexcept;

  template <typename T>
  tnsr::iJ<T, Dim> deriv_shift(const tnsr::I<T, Dim>& x, double /*t*/) const
      noexcept;

  template <typename T>
  tnsr::ii<T, Dim> spatial_metric(const tnsr::I<T, Dim>& x, double /*t*/) const
      noexcept;

  template <typename T>
  tnsr::ii<T, Dim> dt_spatial_metric(const tnsr::I<T, Dim>& x,
                                     double /*t*/) const noexcept;

  template <typename T>
  tnsr::ijj<T, Dim> deriv_spatial_metric(const tnsr::I<T, Dim>& x,
                                         double /*t*/) const noexcept;

  template <typename T>
  tnsr::II<T, Dim> inverse_spatial_metric(const tnsr::I<T, Dim>& x, double /*t*/
                                          ) const noexcept;
  template <typename T>
  tnsr::ii<T, Dim> extrinsic_curvature(const tnsr::I<T, Dim>& x,
                                       double /*t*/) const noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& /*p*/){};  // NOLINT
};
}  // namespace EinsteinSolutions

MAKE_CREATABLE_FROM_YAML(size_t Dim, EinsteinSolutions::Minkowski<Dim>)

// ================================================================
// Minkowski Definitions
// ================================================================

namespace EinsteinSolutions {
template <size_t Dim>
template <typename T>
Scalar<T> Minkowski<Dim>::lapse(const tnsr::I<T, Dim>& x,
                                const double /*t*/) const noexcept {
  return Scalar<T>(make_with_value<T>(x, 1.));
}

template <size_t Dim>
template <typename T>
Scalar<T> Minkowski<Dim>::dt_lapse(const tnsr::I<T, Dim>& x,
                                   const double /*t*/) const noexcept {
  return Scalar<T>(make_with_value<T>(x, 0.));
}

template <size_t Dim>
template <typename T>
tnsr::i<T, Dim> Minkowski<Dim>::deriv_lapse(const tnsr::I<T, Dim>& x,
                                            const double /*t*/) const noexcept {
  return tnsr::i<T, Dim>(make_with_value<T>(x, 0.));
}

template <size_t Dim>
template <typename T>
Scalar<T> Minkowski<Dim>::sqrt_determinant_of_spatial_metric(
    const tnsr::I<T, Dim>& x, const double /*t*/) const noexcept {
  return Scalar<T>(make_with_value<T>(x, 1.));
}

template <size_t Dim>
template <typename T>
Scalar<T> Minkowski<Dim>::dt_sqrt_determinant_of_spatial_metric(
    const tnsr::I<T, Dim>& x, const double /*t*/) const noexcept {
  return Scalar<T>(make_with_value<T>(x, 0.));
}

template <size_t Dim>
template <typename T>
tnsr::I<T, Dim> Minkowski<Dim>::shift(const tnsr::I<T, Dim>& x,
                                      const double /*t*/) const noexcept {
  return tnsr::I<T, Dim>(make_with_value<T>(x, 0.));
}

template <size_t Dim>
template <typename T>
tnsr::iJ<T, Dim> Minkowski<Dim>::deriv_shift(const tnsr::I<T, Dim>& x,
                                             const double /*t*/) const
    noexcept {
  return tnsr::iJ<T, Dim>(make_with_value<T>(x, 0.));
}

template <size_t Dim>
template <typename T>
tnsr::ii<T, Dim> Minkowski<Dim>::spatial_metric(const tnsr::I<T, Dim>& x,
                                                const double /*t*/) const
    noexcept {
  tnsr::ii<T, Dim> lower_metric(make_with_value<T>(x, 0.));
  for (size_t i = 0.; i < Dim; ++i) {
    lower_metric.get(i, i) = make_with_value<T>(x, 1.);
  }
  return lower_metric;
}

template <size_t Dim>
template <typename T>
tnsr::ii<T, Dim> Minkowski<Dim>::dt_spatial_metric(const tnsr::I<T, Dim>& x,
                                                   const double /*t*/) const
    noexcept {
  return tnsr::ii<T, Dim>(make_with_value<T>(x, 0.));
}

template <size_t Dim>
template <typename T>
tnsr::ijj<T, Dim> Minkowski<Dim>::deriv_spatial_metric(const tnsr::I<T, Dim>& x,
                                                       const double /*t*/) const
    noexcept {
  return tnsr::ijj<T, Dim>(make_with_value<T>(x, 0.));
}

template <size_t Dim>
template <typename T>
tnsr::II<T, Dim> Minkowski<Dim>::inverse_spatial_metric(
    const tnsr::I<T, Dim>& x, const double /*t*/
    ) const noexcept {
  tnsr::II<T, Dim> upper_metric(make_with_value<T>(x, 0.));
  for (size_t i = 0; i < Dim; ++i) {
    upper_metric.get(i, i) = make_with_value<T>(x, 1.);
  }
  return upper_metric;
}

template <size_t Dim>
template <typename T>
tnsr::ii<T, Dim> Minkowski<Dim>::extrinsic_curvature(const tnsr::I<T, Dim>& x,
                                                     const double /*t*/) const
    noexcept {
  return tnsr::ii<T, Dim>(make_with_value<T>(x, 0.));
}
}  // namespace EinsteinSolutions
