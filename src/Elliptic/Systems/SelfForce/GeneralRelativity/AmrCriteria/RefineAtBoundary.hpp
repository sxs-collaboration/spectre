// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Tags.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Type.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace GrSelfForce::AmrCriteria {
/*!
 * \brief Split boundary elements in the direction of the boundary
 *
 * For `DimToRefine == 0` (radial), only the lower boundary (near the horizion)
 * is refined. For `DimToRefine == 1` (angular), both boundaries are refined
 * since they are poles like $\theta=0,\pi$.
 */
template <size_t Dim, size_t DimToRefine>
class RefineAtBoundary : public amr::Criterion {
  static_assert(Dim == 2,
                "This specific implementation is tailored for Dim == 2");
  static_assert(DimToRefine < Dim);

 public:
  using options = tmpl::list<>;

  static constexpr Options::String help = {
      "Split boundary elements in the direction of the boundary. Useful to "
      "refine poles like theta=0,pi."};

  RefineAtBoundary() = default;

  /// \cond
  explicit RefineAtBoundary(CkMigrateMessage* msg) : amr::Criterion(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(RefineAtBoundary);  // NOLINT
  /// \endcond

  amr::Criteria::Type type() override { return amr::Criteria::Type::h; }

  static std::string name() {
    if constexpr (DimToRefine == 0) {
      return "RefineAtBoundaryX";
    } else if constexpr (DimToRefine == 1) {
      return "RefineAtBoundaryY";
    } else {
      return "RefineAtBoundaryZ";
    }
  }

  std::string observation_name() override { return name(); }

  using compute_tags_for_observation_box = tmpl::list<>;
  using argument_tags = tmpl::list<domain::Tags::Element<Dim>>;

  template <typename Metavariables>
  std::array<amr::Flag, Dim> operator()(
      const Element<Dim>& element,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*element_id*/) const {
    const auto& external_boundaries = element.external_boundaries();
    auto result = make_array<Dim>(amr::Flag::DoNothing);
    if constexpr (DimToRefine == 0) {
      // Radial direction: only refine near the horizon (lower boundary)
      if (external_boundaries.count(Direction<Dim>{DimToRefine, Side::Lower}) ==
          1) {
        get<DimToRefine>(result) = amr::Flag::Split;
      }
    } else {
      // Angular direction: refine near both poles
      if (external_boundaries.count(Direction<Dim>{DimToRefine, Side::Lower}) ==
              1 or
          external_boundaries.count(Direction<Dim>{DimToRefine, Side::Upper}) ==
              1) {
        get<DimToRefine>(result) = amr::Flag::Split;
      }
    }
    return result;
  }
};

/// \cond
template <size_t Dim, size_t DimToRefine>
PUP::able::PUP_ID RefineAtBoundary<Dim, DimToRefine>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace GrSelfForce::AmrCriteria
