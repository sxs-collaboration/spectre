// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines ScalarWave::Solutions::StandingWave

#pragma once

#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace ScalarWave::Tags {
struct Psi;
struct Pi;
template <size_t Dim, typename Frame>
struct Phi;
}  // namespace ScalarWave::Tags
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ScalarWave::Solutions {
/*!
 * \brief A standing wave solution to the Euclidean wave equation
 *
 * The solution is given by
 * \f$\Psi(\vec{x},t) = A \sin(\vec{k} \cdot (\vec{x} - \vec{x_0}))
 * \cos(\omega t)\f$
 * with the wave vector \f$\vec{k}\f$, frequency
 * \f$\omega = ||\vec{k}||\f$, amplitude \f$A\f$, and center
 * \f$\vec{x_0}\f$. The first-order variables follow the ScalarWave
 * conventions \f$\Pi = -\partial_t \Psi\f$ and \f$\Phi_i = \partial_i \Psi\f$.
 *
 * At \f$t = 0\f$ this gives \f$\Pi = 0\f$, meaning the initial data
 * decomposes into equal left-moving and right-moving components.
 *
 * \tparam Dim the spatial dimension of the solution
 */
template <size_t Dim>
class StandingWave : public evolution::initial_data::InitialData,
                     public MarkAsAnalyticSolution {
 public:
  static constexpr size_t volume_dim = Dim;

  struct WaveVector {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = {
        "The wave vector of the standing wave."};
  };

  struct Center {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = {
        "The center of the spatial profile."};
  };

  struct Amplitude {
    using type = double;
    static constexpr Options::String help = {
        "The amplitude of the standing wave."};
  };

  using options = tmpl::list<WaveVector, Center, Amplitude>;

  static constexpr Options::String help = {
      "A standing wave solution of the Euclidean wave equation. "
      "Psi = A sin(k.(x-x0)) cos(omega t), with omega = |k|. "
      "At t=0, Pi=0 so the wave has equal left- and right-moving components."};

  using tags = tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<Dim, Frame::Inertial>,
                          ::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                          ::Tags::dt<Tags::Phi<Dim, Frame::Inertial>>>;

  StandingWave() = default;
  StandingWave(std::array<double, Dim> wave_vector,
               std::array<double, Dim> center, double amplitude);
  StandingWave(const StandingWave&) = default;
  StandingWave& operator=(const StandingWave&) = default;
  StandingWave(StandingWave&&) = default;
  StandingWave& operator=(StandingWave&&) = default;
  ~StandingWave() override = default;

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit StandingWave(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(StandingWave);
  /// \endcond

  /// Retrieve the evolution variables at time `t` and spatial coordinates `x`
  tuples::TaggedTuple<Tags::Psi, Tags::Pi, Tags::Phi<Dim, Frame::Inertial>>
  variables(const tnsr::I<DataVector, Dim>& x, double t,
            tmpl::list<Tags::Psi, Tags::Pi,
                       Tags::Phi<Dim, Frame::Inertial>> /*meta*/) const;

  /// Retrieve the time derivatives of the evolution variables
  tuples::TaggedTuple<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                      ::Tags::dt<Tags::Phi<Dim, Frame::Inertial>>>
  variables(
      const tnsr::I<DataVector, Dim>& x, double t,
      tmpl::list<::Tags::dt<Tags::Psi>, ::Tags::dt<Tags::Pi>,
                 ::Tags::dt<Tags::Phi<Dim, Frame::Inertial>>> /*meta*/) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const StandingWave<LocalDim>& lhs,
                         const StandingWave<LocalDim>& rhs);
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator!=(const StandingWave<LocalDim>& lhs,
                         const StandingWave<LocalDim>& rhs);

  std::array<double, Dim> wave_vector_{};
  std::array<double, Dim> center_{};
  double amplitude_{};
  double omega_{};
};
}  // namespace ScalarWave::Solutions
