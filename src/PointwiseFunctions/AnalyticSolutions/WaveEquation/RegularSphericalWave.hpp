// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ScalarWave::Solutions::RegularSphericalWave

#pragma once

#include <cstddef>
#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
// IWYU pragma: no_forward_declare Tensor

/// \cond
class DataVector;
namespace ScalarWave::Tags {
struct Pi;
struct Psi;
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

namespace ScalarWave {
namespace Solutions {
/*!
 * \brief A 3D spherical wave solution to the Euclidean wave equation that is
 * regular at the origin
 *
 * The solution is given by \f$\Psi(\vec{x},t) = \Psi(r,t) =
 * \frac{F(r-t)-F(-r-t)}{r}\f$ describing an outgoing and an ingoing wave
 * with profile \f$F(u)\f$. For small \f$r\f$ the solution is approximated by
 * its Taylor expansion \f$\Psi(r,t)=2 F^\prime(-t) + \mathcal{O}(r^2)\f$. The
 * outgoing and ingoing waves meet at the origin (and cancel each other) when
 * \f$F^\prime(-t)=0\f$.
 *
 * The expansion is employed where \f$r\f$ lies within the cubic root of the
 * machine epsilon. Inside this radius we expect the error due to the truncation
 * of the Taylor expansion to be smaller than the numerical error made when
 * evaluating the full \f$\Psi(r,t)\f$. This is because the truncation error
 * scales as \f$r^2\f$ (since we keep the zeroth order, and the linear order
 * vanishes as all odd orders do) and the numerical error scales as
 * \f$\frac{\epsilon}{r}\f$, so they are comparable at
 * \f$r\propto\epsilon^\frac{1}{3}\f$.
 *
 * \requires the profile \f$F(u)\f$ to have a length scale of order unity so
 * that "small" \f$r\f$ means \f$r\ll 1\f$. This is without loss of generality
 * because of the scale invariance of the wave equation. The profile could be a
 * Gausssian centered at 0 with width 1, for instance.
 */
class RegularSphericalWave : public MarkAsAnalyticSolution {
 public:
  static constexpr size_t volume_dim = 3;
  struct Profile {
    using type = std::unique_ptr<MathFunction<1, Frame::Inertial>>;
    static constexpr Options::String help = {
        "The radial profile of the spherical wave."};
  };

  using options = tmpl::list<Profile>;

  static constexpr Options::String help = {
      "A spherical wave solution of the Euclidean wave equation that is "
      "regular at the origin"};

  RegularSphericalWave() = default;
  explicit RegularSphericalWave(
      std::unique_ptr<MathFunction<1, Frame::Inertial>> profile);
  RegularSphericalWave(const RegularSphericalWave&) = delete;
  RegularSphericalWave& operator=(const RegularSphericalWave&) = delete;
  RegularSphericalWave(RegularSphericalWave&&) = default;
  RegularSphericalWave& operator=(RegularSphericalWave&&) = default;
  ~RegularSphericalWave() = default;

  tuples::TaggedTuple<Tags::Psi, Tags::Pi, Tags::Phi<3>> variables(
      const tnsr::I<DataVector, 3>& x, double t,
      tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<3>> /*meta*/) const;

  tuples::TaggedTuple<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                      ::Tags::dt<Tags::Phi<3>>>
  variables(const tnsr::I<DataVector, 3>& x, double t,
            tmpl::list<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                       ::Tags::dt<Tags::Phi<3>>> /*meta*/) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  std::unique_ptr<MathFunction<1, Frame::Inertial>> profile_;
};
}  // namespace Solutions
}  // namespace ScalarWave
