// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines WaveEquationSolutions::PlaneWave

#pragma once

#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
namespace ScalarWave::Tags {
struct Psi;
struct Pi;
template <size_t Dim>
struct Phi;
}  // namespace ScalarWave::Tags
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
template <size_t VolumeDim, typename Fr>
class MathFunction;

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ScalarWave::Solutions {
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
class PlaneWave : public evolution::initial_data::InitialData,
                  public MarkAsAnalyticSolution {
 public:
  static constexpr size_t volume_dim = Dim;
  struct WaveVector {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = {
        "The direction of propagation of the wave."};
  };

  struct Center {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = {
        "The initial center of the profile of the wave."};
  };

  struct Profile {
    using type = std::unique_ptr<MathFunction<1, Frame::Inertial>>;
    static constexpr Options::String help = {"The profile of the wave."};
  };

  using options = tmpl::list<WaveVector, Center, Profile>;

  static constexpr Options::String help = {
      "A plane wave solution of the Euclidean wave equation"};
  using tags =
      tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<3>, ::Tags::dt<Tags::Psi>,
                 ::Tags::dt<Tags::Pi>, ::Tags::dt<Tags::Phi<Dim>>>;

  PlaneWave() = default;
  PlaneWave(std::array<double, Dim> wave_vector, std::array<double, Dim> center,
            std::unique_ptr<MathFunction<1, Frame::Inertial>> profile);
  PlaneWave(const PlaneWave&);
  PlaneWave& operator=(const PlaneWave&);
  PlaneWave(PlaneWave&&) = default;
  PlaneWave& operator=(PlaneWave&&) = default;
  ~PlaneWave() override = default;

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit PlaneWave(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(PlaneWave);
  /// \endcond

  /// The value of the scalar field
  template <typename T>
  Scalar<T> psi(const tnsr::I<T, Dim>& x, double t) const;

  /// The time derivative of the scalar field
  template <typename T>
  Scalar<T> dpsi_dt(const tnsr::I<T, Dim>& x, double t) const;

  /// The spatial derivatives of the scalar field
  template <typename T>
  tnsr::i<T, Dim> dpsi_dx(const tnsr::I<T, Dim>& x, double t) const;

  /// The second time derivative of the scalar field
  template <typename T>
  Scalar<T> d2psi_dt2(const tnsr::I<T, Dim>& x, double t) const;

  /// The second mixed derivatives of the scalar field
  template <typename T>
  tnsr::i<T, Dim> d2psi_dtdx(const tnsr::I<T, Dim>& x, double t) const;

  /// The second spatial derivatives of the scalar field
  template <typename T>
  tnsr::ii<T, Dim> d2psi_dxdx(const tnsr::I<T, Dim>& x, double t) const;

  /// Retrieve the evolution variables at time `t` and spatial coordinates `x`
  tuples::TaggedTuple<Tags::Psi, Tags::Pi, Tags::Phi<Dim>> variables(
      const tnsr::I<DataVector, Dim>& x, double t,
      tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim>> /*meta*/) const;

  /// Retrieve the time derivative of the evolution variables at time `t` and
  /// spatial coordinates `x`
  ///
  /// \note This function's expected use case is setting the past time
  /// derivative values for Adams-Bashforth-like steppers.
  tuples::TaggedTuple<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                      ::Tags::dt<Tags::Phi<Dim>>>
  variables(const tnsr::I<DataVector, Dim>& x, double t,
            tmpl::list<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                       ::Tags::dt<Tags::Phi<Dim>>> /*meta*/) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const PlaneWave<LocalDim>& lhs,
                         const PlaneWave<LocalDim>& rhs);
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator!=(const PlaneWave<LocalDim>& lhs,
                         const PlaneWave<LocalDim>& rhs);

  template <typename T>
  T u(const tnsr::I<T, Dim>& x, double t) const;

  std::array<double, Dim> wave_vector_{};
  std::array<double, Dim> center_{};
  std::unique_ptr<MathFunction<1, Frame::Inertial>> profile_;
  double omega_{};
};
}  // namespace ScalarWave::Solutions
