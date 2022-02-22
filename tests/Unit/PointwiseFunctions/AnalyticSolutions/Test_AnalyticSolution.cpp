// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"

namespace {
struct Solution : public MarkAsAnalyticSolution {};
struct SolutionDependentAnalyticData : public MarkAsAnalyticData,
                                       private Solution {};

static_assert(is_analytic_solution_v<Solution>,
              "Failed testing is_analytic_solution_v");
static_assert(is_analytic_solution<Solution>::value,
              "Failed testing is_analytic_solution");
static_assert(not is_analytic_solution_v<SolutionDependentAnalyticData>,
              "Failed testing is_solution_data_v");
static_assert(not is_analytic_solution<SolutionDependentAnalyticData>::value,
              "Failed testing is_solution_data");
}  // namespace
