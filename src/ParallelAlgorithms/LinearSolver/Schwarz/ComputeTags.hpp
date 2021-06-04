// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver::Schwarz::Tags {

/*!
 * \brief A diagnostic quantity to check that weights are conserved
 *
 * \see `LinearSolver::Schwarz::Tags::SummedIntrudingOverlapWeights`
 */
template <size_t Dim, typename OptionsGroup>
struct SummedIntrudingOverlapWeightsCompute
    : db::ComputeTag,
      SummedIntrudingOverlapWeights<OptionsGroup> {
  using base = SummedIntrudingOverlapWeights<OptionsGroup>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                                         Weight<OptionsGroup>>,
                 domain::Tags::Mesh<Dim>, IntrudingExtents<Dim, OptionsGroup>>;
  static void function(
      const gsl::not_null<return_type*> summed_intruding_overlap_weights,
      const std::unordered_map<Direction<Dim>, Scalar<DataVector>>&
          all_intruding_weights,
      const Mesh<Dim>& mesh,
      const std::array<size_t, Dim>& all_intruding_extents) noexcept {
    destructive_resize_components(summed_intruding_overlap_weights,
                                  mesh.number_of_grid_points());
    get(*summed_intruding_overlap_weights) = 0.;
    for (const auto& [direction, intruding_weight] : all_intruding_weights) {
      // Extend intruding weight to full extents
      // There's not function to extend a single tensor, so we create a
      // temporary Variables. This is only a diagnostic quantity that won't be
      // used in production code, so it doesn't need to be particularly
      // efficient and we don't need to add an overload of
      // `LinearSolver::Schwarz::extended_overlap_data` just for this purpose.
      using temp_tag = ::Tags::TempScalar<0>;
      Variables<tmpl::list<temp_tag>> temp_vars{get(intruding_weight).size()};
      get<temp_tag>(temp_vars) = intruding_weight;
      temp_vars = LinearSolver::Schwarz::extended_overlap_data(
          temp_vars, mesh.extents(),
          gsl::at(all_intruding_extents, direction.dimension()), direction);
      // Contribute to conserved weight
      get(*summed_intruding_overlap_weights) += get(get<temp_tag>(temp_vars));
    }
  }
};

}  // namespace LinearSolver::Schwarz::Tags
