// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <pup.h>
#include <type_traits>

#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/TMPL.hpp"

namespace amr {
/// \ingroup AmrGroup
/// \brief Base class for something that determines how an adaptive mesh should
/// be changed
///
/// \details When AMR criteria are evaluated for each element, they should
/// return a std::aray<amr::domain::Flag, Dim> containing the recommended
/// refinement choice in each logical dimension of the Element.
class Criterion : public PUP::able {
 protected:
  /// \cond
  Criterion() = default;
  Criterion(const Criterion&) = default;
  Criterion(Criterion&&) = default;
  Criterion& operator=(const Criterion&) = default;
  Criterion& operator=(Criterion&&) = default;
  /// \endcond

 public:
  ~Criterion() override = default;
  explicit Criterion(CkMigrateMessage* msg) : PUP::able(msg) {}

  WRAPPED_PUPable_abstract(Criterion);  // NOLINT

  template <typename ComputeTagsList, typename DataBoxType,
            typename Metavariables, typename ArrayIndex>
  auto evaluate(const ObservationBox<ComputeTagsList, DataBoxType>& box,
                Parallel::GlobalCache<Metavariables>& cache,
                const ArrayIndex& array_index) const {
    using factory_classes =
        typename std::decay_t<Metavariables>::factory_creation::factory_classes;
    return call_with_dynamic_type<
        std::array<amr::domain::Flag, Metavariables::volume_dim>,
        tmpl::at<factory_classes, Criterion>>(
        this, [&box, &cache, &array_index](auto* const criterion) {
          return apply(*criterion, box, cache, array_index);
        });
  }
};
}  // namespace amr
