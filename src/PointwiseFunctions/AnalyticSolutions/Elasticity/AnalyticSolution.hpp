// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticData/Elasticity/AnalyticData.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Elasticity {
/// Analytic solutions of the linear Elasticity equations
namespace Solutions {

/*!
 * \brief Base class for analytic solutions of the linear Elasticity equations
 *
 * This abstract base class allows factory-creating derived classes from
 * input-file options (see `Registration`). The list of `Registrars` may also
 * contain classes that aren't analytic solutions but only derive off
 * `Elasticity::AnalyticData::AnalyticData`. This allows factory-creating any
 * choice of analytic solution _or_ analytic data from input-file options, but
 * you must use the `Elasticity::AnalyticData::AnalyticData` class for the
 * factory-creation. Note that analytic solutions provide more variables than
 * analytic data classes, so to request variables that only analytic solutions
 * provide you must perform a cast to `AnalyticSolution`.
 */
template <size_t Dim, typename Registrars>
class AnalyticSolution
    : public Elasticity::AnalyticData::AnalyticData<Dim, Registrars> {
 private:
  using Base = Elasticity::AnalyticData::AnalyticData<Dim, Registrars>;

 protected:
  /// \cond
  AnalyticSolution() = default;
  AnalyticSolution(const AnalyticSolution&) = default;
  AnalyticSolution(AnalyticSolution&&) = default;
  AnalyticSolution& operator=(const AnalyticSolution&) = default;
  AnalyticSolution& operator=(AnalyticSolution&&) = default;
  /// \endcond

 public:
  ~AnalyticSolution() override = default;

  using registrars = Registrars;
  using creatable_classes = Registration::registrants<registrars>;

  /// \cond
  explicit AnalyticSolution(CkMigrateMessage* m) : Base(m) {}
  WRAPPED_PUPable_abstract(AnalyticSolution);  // NOLINT
  /// \endcond

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, Dim, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return call_with_dynamic_type<
        tuples::TaggedTuple<RequestedTags...>,
        tmpl::filter<
            creatable_classes,
            std::is_base_of<AnalyticSolution<Dim, Registrars>, tmpl::_1>>>(
        this, [&x](auto* const derived) {
          return derived->variables(x, tmpl::list<RequestedTags...>{});
        });
  }
};

}  // namespace Solutions
}  // namespace Elasticity
