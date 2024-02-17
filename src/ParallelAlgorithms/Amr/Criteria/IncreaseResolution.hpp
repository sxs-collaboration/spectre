// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <pup.h>

#include "Domain/Amr/Flag.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace amr::Criteria {
/*!
 * \brief Uniformly increases the number of grid points by one
 *
 * Useful to do uniform p-refinement, possibly alongside a nontrivial
 * h-refinement criterion.
 */
template <size_t Dim>
class IncreaseResolution : public Criterion {
 public:
  using options = tmpl::list<>;

  static constexpr Options::String help = {
      "Uniformly increases the number of grid points by one."};

  IncreaseResolution() = default;

  /// \cond
  explicit IncreaseResolution(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(IncreaseResolution);  // NOLINT
  /// \endcond

  using compute_tags_for_observation_box = tmpl::list<>;
  using argument_tags = tmpl::list<>;

  template <typename Metavariables>
  std::array<Flag, Dim> operator()(
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*element_id*/) const {
    return make_array<Dim>(Flag::IncreaseResolution);
  }
};

/// \cond
template <size_t Dim>
PUP::able::PUP_ID IncreaseResolution<Dim>::my_PUP_ID = 0;  // NOLINT
/// \endcond

}  // namespace amr::Criteria
