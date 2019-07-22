// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/TypeTraits.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"

namespace {
struct AnalyticData : public MarkAsAnalyticData {};
struct Solution : public MarkAsAnalyticSolution {};
struct SolutionDependentAnalyticData : public MarkAsAnalyticData,
                                       private Solution {};

static_assert(evolution::is_analytic_solution_v<Solution>,
              "Failed testing evolution::is_analytic_solution_v");
static_assert(evolution::is_analytic_data_v<AnalyticData>,
              "Failed testing evolution::is_analytic_data_v");
static_assert(evolution::is_analytic_data_v<SolutionDependentAnalyticData>,
              "Failed testing evolution::is_analytic_data_v");
static_assert(
    not evolution::is_analytic_solution_v<SolutionDependentAnalyticData>,
    "Failed testing evolution::is_solution_data_v");

static_assert(evolution::is_analytic_solution<Solution>::value,
              "Failed testing evolution::is_analytic_solution");
static_assert(evolution::is_analytic_data<AnalyticData>::value,
              "Failed testing evolution::is_analytic_data");
static_assert(evolution::is_analytic_data<SolutionDependentAnalyticData>::value,
              "Failed testing evolution::is_analytic_data");
static_assert(
    not evolution::is_analytic_solution<SolutionDependentAnalyticData>::value,
    "Failed testing evolution::is_solution_data");
}  // namespace
