// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <pup.h>
#include <type_traits>

#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace amr {
/// \ingroup AmrGroup
/// \brief Base class for something that determines how an adaptive mesh should
/// be changed
///
/// \details Each class derived from this class should (see the examples below):
/// - Be option-creatable
/// - Be serializable
/// - Define a call operator that returns a std::array<amr::Flag, Dim>
///   containing the recommended refinement choice in each logical dimension of
///   the Element.
/// - Define the type aliases `argument_tags` and
///   `compute_tags_for_observation_box` that are type lists of tags used in the
///   call operator.
/// The call operator should take as arguments the values corresponding to each
/// tag in `argument_tags` (in order), followed by the Parallel::GlobalCache,
/// and the ElementId.  The tags listed in `argument_tags` should either be tags
/// in the DataBox of the array component, or listed in
/// `compute_tags_for_observation_box`.
///
/// \example
/// \snippet Test_Criterion.cpp criterion_examples
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

  /// Evaluates the AMR criteria by selecting the appropriate derived class
  /// and forwarding its `argument_tags` from the ObservationBox (along with the
  /// GlobalCache and ArrayIndex) to the call operator of the derived class
  ///
  /// \note In order to be available, a derived Criterion must be listed in
  /// the entry for Criterion in
  /// Metavarialbes::factory_creation::factory_classes
  ///
  /// \note The ComputeTagsList of the ObservationBox should contain the union
  /// of the tags listed in `compute_tags_for_observation_box` for each derived
  /// Criterion listed in the `factory_classes`.
  template <typename ComputeTagsList, typename DataBoxType,
            typename Metavariables>
  auto evaluate(const ObservationBox<ComputeTagsList, DataBoxType>& box,
                Parallel::GlobalCache<Metavariables>& cache,
                const ElementId<Metavariables::volume_dim>& element_id) const {
    using factory_classes =
        typename std::decay_t<Metavariables>::factory_creation::factory_classes;
    return call_with_dynamic_type<
        std::array<amr::Flag, Metavariables::volume_dim>,
        tmpl::at<factory_classes, Criterion>>(
        this, [&box, &cache, &element_id](auto* const criterion) {
          return apply(*criterion, box, cache, element_id);
        });
  }
};
}  // namespace amr
