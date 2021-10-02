// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/Elasticity/ConstitutiveRelations/ConstitutiveRelation.hpp"
#include "Utilities/FakeVirtual.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Elasticity {
/// Analytic data for the Elasticity system
namespace AnalyticData {

/// Base class for the background of the Elasticity system, i.e. its
/// variable-independent quantities. Derived classes must provide a constitutive
/// relation.
template <size_t Dim, typename Registrars>
class AnalyticData : public ::AnalyticData<Dim, Registrars> {
 private:
  using Base = ::AnalyticData<Dim, Registrars>;

 protected:
  /// \cond
  AnalyticData() = default;
  AnalyticData(const AnalyticData&) = default;
  AnalyticData(AnalyticData&&) = default;
  AnalyticData& operator=(const AnalyticData&) = default;
  AnalyticData& operator=(AnalyticData&&) = default;
  /// \endcond

 public:
  ~AnalyticData() override = default;

  /// \cond
  explicit AnalyticData(CkMigrateMessage* m) : Base(m) {}
  WRAPPED_PUPable_abstract(AnalyticData);  // NOLINT
  /// \endcond

  /// A constitutive relation that represents the properties of the elastic
  /// material
  virtual const ConstitutiveRelations::ConstitutiveRelation<Dim>&
  constitutive_relation() const = 0;
};
}  // namespace AnalyticData
}  // namespace Elasticity
