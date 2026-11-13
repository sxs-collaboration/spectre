// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <pup.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Type.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class ElementId;
template <size_t Dim>
class Mesh;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace gh::AmrCriteria {
/*!
 * \brief p-refine or p-coarsen an element towards a target truncation error of
 * the generalized harmonic variables, as SpEC's adaptive mesh refinement does.
 *
 * \details In each logical direction the spectral coefficients are monitored
 * in two groups, as in SpEC: the spacetime metric, and its derivatives
 * \f$\Pi_{ab}\f$ and \f$\Phi_{iab}\f$ together. The power monitor of a group is
 * the root mean square over the tensor components of
 * `PowerMonitors::directional_power_monitors`. The number of filtered modes is
 * measured as in SpEC: the filter is applied to a function whose modes in that
 * direction are all one, and the amounts that it removes from the modes are
 * added up and rounded. From the remaining, unfiltered modes of each group,
 * `PowerMonitors::relative_truncation_error` computes the truncation error
 * \f$E\f$ of the current expansion as well as the truncation error
 * \f$E^-\f$ given by one fewer mode. Each direction of each element is then:
 *
 * - p-refined if \f$E\f$ of either group is larger than
 *   `RelativeTruncationTarget`,
 * - p-coarsened if \f$10^{0.1} E^-\f$ of both groups is at most
 *   `RelativeTruncationTarget`,
 * - left unchanged otherwise.
 *
 * This is the truncation error part of the p-refinement criterion of
 * \cite Szilagyi2014fna as implemented by SpEC's `ChangeSpectralGrid`. The
 * following parts of that criterion are not implemented yet: the limits from
 * pile-up modes and from the projected constraints, the adjustment of the
 * target, and the waiting times after a change. Only meshes with Legendre or
 * Chebyshev bases and filters that do not need Jacobians are supported so far.
 *
 * \note The truncation error differs from SpEC's where the unfiltered modes
 * reach roundoff: `PowerMonitors::relative_truncation_error` ignores modes
 * below roundoff and treats a spectrum whose two highest modes are below
 * roundoff as exactly represented, whereas SpEC averages over all modes.
 */
class SpecPRefinement : public amr::Criterion {
 public:
  struct RelativeTruncationTarget {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The target relative truncation error."};
  };

  using options = tmpl::list<RelativeTruncationTarget>;

  static constexpr Options::String help = {
      "Refine the grid towards the target truncation error of the generalized "
      "harmonic variables, following SpEC."};

  SpecPRefinement() = default;

  explicit SpecPRefinement(double relative_truncation_target);

  /// \cond
  explicit SpecPRefinement(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SpecPRefinement);  // NOLINT
  /// \endcond

  amr::Criteria::Type type() override { return amr::Criteria::Type::p; }

  std::string observation_name() override { return "SpecPRefinement"; }

  using compute_tags_for_observation_box = tmpl::list<>;

  using argument_tags = tmpl::list<
      gr::Tags::SpacetimeMetric<DataVector, 3>, Tags::Pi<DataVector, 3>,
      Tags::Phi<DataVector, 3>, domain::Tags::Mesh<3>,
      Filters::Tags::SpectralFilter<3, System<3>::variables_tag::tags_list>>;

  template <typename Metavariables>
  std::array<amr::Flag, 3> operator()(
      const tnsr::aa<DataVector, 3>& spacetime_metric,
      const tnsr::aa<DataVector, 3>& pi, const tnsr::iaa<DataVector, 3>& phi,
      const Mesh<3>& mesh,
      const Filters::Filter<3, System<3>::variables_tag::tags_list>& filter,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<3>& /*element_id*/) const {
    return impl(spacetime_metric, pi, phi, mesh, filter);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  std::array<amr::Flag, 3> impl(
      const tnsr::aa<DataVector, 3>& spacetime_metric,
      const tnsr::aa<DataVector, 3>& pi, const tnsr::iaa<DataVector, 3>& phi,
      const Mesh<3>& mesh,
      const Filters::Filter<3, System<3>::variables_tag::tags_list>& filter)
      const;

  double relative_truncation_target_{
      std::numeric_limits<double>::signaling_NaN()};
};
}  // namespace gh::AmrCriteria
