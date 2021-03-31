// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts {
/// Analytic solutions of the \ref Xcts "XCTS" equations
namespace Solutions {

/*!
 * \brief Base class for analytic solutions of the \ref Xcts "XCTS" equations
 *
 * This abstract base class allows factory-creating derived classes from
 * input-file options (see `Registration`). The list of `Registrars` may also
 * contain classes that aren't analytic solutions but only derive off
 * `::AnalyticData`. This allows factory-creating any choice of analytic
 * solution _or_ analytic data from input-file options, but you must use the
 * `::AnalyticData` class for the factory-creation. Note that analytic solutions
 * provide more variables than analytic data classes, so to request variables
 * that only analytic solutions provide you must perform a cast to
 * `AnalyticSolution`.
 */
template <typename Registrars>
class AnalyticSolution : public ::AnalyticData<3, Registrars> {
 private:
  using Base = ::AnalyticData<3, Registrars>;

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
  explicit AnalyticSolution(CkMigrateMessage* m) noexcept : Base(m) {}
  WRAPPED_PUPable_abstract(AnalyticSolution);  // NOLINT
  /// \endcond

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const noexcept {
    return call_with_dynamic_type<
        tuples::TaggedTuple<RequestedTags...>,
        tmpl::filter<creatable_classes,
                     std::is_base_of<tmpl::pin<AnalyticSolution<Registrars>>,
                                     tmpl::_1>>>(
        this, [&x](auto* const derived) noexcept {
          return derived->variables(x, tmpl::list<RequestedTags...>{});
        });
  }
};

}  // namespace Solutions
}  // namespace Xcts
