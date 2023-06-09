// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"  // IWYU pragma: keep
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
// IWYU pragma: no_forward_declare Tensor
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ScalarWave::Solutions {
/*!
 * \brief An exact solution to the semidiscretized DG ScalarWave
 * system with an upwind flux
 *
 * This solution takes into account the spatial discretization error,
 * and so should show convergence in time integration accuracy to
 * roundoff at any resolution.
 *
 * \warning This is not really a pointwise function, as the solution
 * depends on the spatial discretization.  It will only work on a
 * periodic domain of length \f$2 \pi\f$ (or an integer multiple) with
 * equally sized linear elements.
 */
class SemidiscretizedDg : public evolution::initial_data::InitialData,
                          public MarkAsAnalyticSolution {
 public:
  using tags = tmpl::list<Tags::Pi, Tags::Phi<1>, Tags::Psi>;

  struct Harmonic {
    using type = int;
    static constexpr Options::String help =
        "Number of wave periods across the domain";
  };

  struct Amplitudes {
    using type = std::array<double, 4>;
    static constexpr Options::String help =
        "Amplitudes of the independent modes of the harmonic";
  };

  using options = tmpl::list<Harmonic, Amplitudes>;

  static constexpr Options::String help =
      "A solution of the semidiscretized DG system on linear elements\n"
      "with spatial period 2 pi.";

  SemidiscretizedDg(int harmonic, const std::array<double, 4>& amplitudes);

  SemidiscretizedDg() = default;
  ~SemidiscretizedDg() override = default;

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit SemidiscretizedDg(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SemidiscretizedDg);
  /// \endcond

  /// Retrieve the evolution variables at time `t` and spatial coordinates `x`
  template <typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataVector, 1>& x,
                                         double t,
                                         tmpl::list<Tags...> /*meta*/) const {
    static_assert(
        tmpl2::flat_all_v<tmpl::list_contains_v<tags, Tags>...>,
        "At least one of the requested tags is not supported. The requested "
        "tags are listed as template parameters of the `variables` function.");
    return {get<Tags>(variables(x, t, tmpl::list<Tags>{}))...};
  }

  /// \cond
  tuples::TaggedTuple<Tags::Pi> variables(const tnsr::I<DataVector, 1>& x,
                                          double t,
                                          tmpl::list<Tags::Pi> /*meta*/) const;

  tuples::TaggedTuple<Tags::Phi<1>> variables(
      const tnsr::I<DataVector, 1>& x, double t,
      tmpl::list<Tags::Phi<1>> /*meta*/) const;

  tuples::TaggedTuple<Tags::Psi> variables(
      const tnsr::I<DataVector, 1>& x, double t,
      tmpl::list<Tags::Psi> /*meta*/) const;
  /// \endcond

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  int harmonic_{std::numeric_limits<int>::max()};
  std::array<double, 4> amplitudes_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
};
}  // namespace ScalarWave::Solutions
