// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines WaveEquationSolutions::PlaneWave

#pragma once

#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
namespace ScalarWave {
struct Pi;
struct Psi;
template <size_t Dim>
struct Phi;
}  // namespace ScalarWave
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
template <size_t VolumeDim>
class MathFunction;

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

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
    static constexpr OptionString help = {
        "The direction of propagation of the wave."};
  };

  struct Center {
    using type = std::array<double, Dim>;
    static constexpr OptionString help = {
        "The initial center of the profile of the wave."};
    static type default_value() noexcept { return make_array<Dim>(0.0); }
  };

  struct Profile {
    using type = std::unique_ptr<MathFunction<1>>;
    static constexpr OptionString help = {"The profile of the wave."};
  };

  using options = tmpl::list<WaveVector, Center, Profile>;

  static constexpr OptionString help = {
      "A plane wave solution of the Euclidean wave equation"};

  PlaneWave() = default;
  PlaneWave(std::array<double, Dim> wave_vector, std::array<double, Dim> center,
            std::unique_ptr<MathFunction<1>> profile) noexcept;
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

  /// Retrieve the evolution variables at time `t` and spatial coordinates `x`
  tuples::TaggedTuple<ScalarWave::Pi, ScalarWave::Phi<Dim>, ScalarWave::Psi>
  variables(const tnsr::I<DataVector, Dim>& x, double t,
            tmpl::list<ScalarWave::Pi, ScalarWave::Phi<Dim>,
                       ScalarWave::Psi> /*meta*/) const noexcept;

  /// Retrieve the time derivative of the evolution variables at time `t` and
  /// spatial coordinates `x`
  ///
  /// \note This function's expected use case is setting the past time
  /// derivative values for Adams-Bashforth-like steppers.
  tuples::TaggedTuple<Tags::dt<ScalarWave::Pi>, Tags::dt<ScalarWave::Phi<Dim>>,
                      Tags::dt<ScalarWave::Psi>>
  variables(const tnsr::I<DataVector, Dim>& x, double t,
            tmpl::list<Tags::dt<ScalarWave::Pi>, Tags::dt<ScalarWave::Phi<Dim>>,
                       Tags::dt<ScalarWave::Psi>> /*meta*/) const noexcept;

  // clang-tidy: no pass by reference
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  template <typename T>
  T u(const tnsr::I<T, Dim>& x, double t) const noexcept;

  std::array<double, Dim> wave_vector_{};
  std::array<double, Dim> center_{};
  std::unique_ptr<MathFunction<1>> profile_;
  double omega_{};
};
}  // namespace Solutions
}  // namespace ScalarWave
