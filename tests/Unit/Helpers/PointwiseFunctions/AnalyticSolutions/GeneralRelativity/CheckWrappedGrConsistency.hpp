// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TMPL.hpp"

template <typename WrappedGrSolution, typename ArgumentSolution, size_t Dim>
void check_wrapped_gr_solution_consistency(
    const WrappedGrSolution& wrapped_solution,
    const ArgumentSolution& argument_solution,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& x, const double t) {
  using argument_solution_tags =
      typename ArgumentSolution::template tags<DataVector>;
  const auto wrapped_vars =
      wrapped_solution.variables(x, t, argument_solution_tags{});
  const auto argument_vars =
      argument_solution.variables(x, t, argument_solution_tags{});
  tmpl::for_each<argument_solution_tags>(
      [&wrapped_vars, &argument_vars](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        CHECK(get<tag>(wrapped_vars) == get<tag>(argument_vars));
      });
}
