// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines WaveEquationSolutions::PlaneWave

#pragma once

#include <array>
#include <memory>

#include "DataStructures/MakeWithValue.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/OptionsDetails.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/MakeArray.hpp"

namespace ScalarWave {
namespace Solutions {
/*!
 * \brief A plane wave solution to the Euclidean wave equation
 *
 * The solution is given by \f$\Psi(\vec{x},t) = F(u(\vec{x},t))\f$
 * where the profile \f$F\f$ of the plane wave is an arbitrary one-dimensional
 * function of \f$u = \vec{k} \cdot (\vec{x} - \vec{x_o}) - \omega t\f$
 * with the wave vector \f$\vec{k}\f$, the frequency \f$\omega = ||\vec{k}||\f$
 * and initial center of the profile \f$\vec{x_o}\f$.
 *
 * \tparam Dim the spatial dimension of the solution
 */
template <size_t Dim>
class PlaneWave {
 public:
  struct WaveVector {
    using type = std::array<double, Dim>;
    static constexpr OptionString_t help = {
        "The direction of propagation of the wave."};
  };

  struct Center {
    using type = std::array<double, Dim>;
    static constexpr OptionString_t help = {
        "The initial center of the profile of the wave."};
    static type default_value() { return make_array<Dim>(0.0); }
  };

  struct Profile {
    using type = std::unique_ptr<MathFunction<1>>;
    static constexpr OptionString_t help = {"The profile of the wave."};
  };

  using options = tmpl::list<WaveVector, Center, Profile>;

  static constexpr OptionString_t help = {
      "A plane wave solution of the Euclidean wave equation"};

  PlaneWave(std::array<double, Dim> wave_vector, std::array<double, Dim> center,
            std::unique_ptr<MathFunction<1>>&& profile) noexcept;
  PlaneWave(const PlaneWave&) noexcept = delete;
  PlaneWave& operator=(const PlaneWave&) noexcept = delete;
  PlaneWave(PlaneWave&&) noexcept = default;
  PlaneWave& operator=(PlaneWave&&) noexcept = default;
  ~PlaneWave() noexcept = default;

  /// The value of the scalar field
  template <typename T>
  Scalar<T> psi(const tnsr::I<T, Dim>& x, double t) const noexcept;

  /// The time derivative of the scalar field
  template <typename T>
  Scalar<T> dpsi_dt(const tnsr::I<T, Dim>& x, double t) const noexcept;

  /// The spatial derivatives of the scalar field
  template <typename T>
  tnsr::i<T, Dim> dpsi_dx(const tnsr::I<T, Dim>& x, double t) const noexcept;

  /// The second time derivative of the scalar field
  template <typename T>
  Scalar<T> d2psi_dt2(const tnsr::I<T, Dim>& x, double t) const noexcept;

  /// The second mixed derivatives of the scalar field
  template <typename T>
  tnsr::i<T, Dim> d2psi_dtdx(const tnsr::I<T, Dim>& x, double t) const noexcept;

  /// The second spatial derivatives of the scalar field
  template <typename T>
  tnsr::ii<T, Dim> d2psi_dxdx(const tnsr::I<T, Dim>& x, double t) const
      noexcept;

 private:
  template <typename T>
  T u(const tnsr::I<T, Dim>& x, double t) const noexcept;

  std::array<double, Dim> wave_vector_;
  std::array<double, Dim> center_;
  std::unique_ptr<MathFunction<1>> profile_;
  double omega_;
};

// ======================================================================
// Template Definitions
// ======================================================================

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
