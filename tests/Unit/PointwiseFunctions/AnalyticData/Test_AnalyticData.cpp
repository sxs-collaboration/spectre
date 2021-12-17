// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"

namespace {
struct AnalyticData : public MarkAsAnalyticData {};
struct Solution : public MarkAsAnalyticSolution {};
struct SolutionDependentAnalyticData : public MarkAsAnalyticData,
                                       private Solution {};

static_assert(is_analytic_data_v<AnalyticData>,
              "Failed testing is_analytic_data_v");
static_assert(is_analytic_data_v<SolutionDependentAnalyticData>,
              "Failed testing is_analytic_data_v");

static_assert(is_analytic_data<AnalyticData>::value,
              "Failed testing is_analytic_data");
static_assert(is_analytic_data<SolutionDependentAnalyticData>::value,
              "Failed testing is_analytic_data");
}  // namespace
