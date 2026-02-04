// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

namespace ah {
/*!
 * \ingroup SurfacesGroup
 * \brief Base class for criteria that determine how the resolution of an
 * apparent horizon should be changed
 *
 * \details Each class derived from this class should
 * - Be option-creatable
 * - Be serializable
 * - Define the type aliases `argument_tags` and
 *   `compute_tags_for_observation_box` that are type lists of tags used in the
 *   call operator
 * - Define a call operator that returns a `size_t` that is the recommended
 *   resolution $L$ for the Strahlkorper representing the apparent horizon
 *
 * The call operator should take the following arguments, in order:
 * - An argument for each tag in `argument_tags`
 * - The Parallel::GlobalCache
 * - The Strahlkorper representing the apparent horizon
 * - A FastFlow::IterInfo corresponding to the horizon find that found the
 *   Strahlkorper
 */
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

  WRAPPED_PUPable_abstract(Criterion);

  virtual std::string observation_name() = 0;

  virtual bool is_equal(const Criterion& other) const = 0;

  /// Evaluates the apparent horizon criteria by selecting the appropriate
  /// derived class and forwarding its `argument_tags` from the ObservationBox
  /// (along with the GlobalCache) to the call operator of the
  /// derived class
  ///
  /// \note In order to be available, a derived Criterion must be listed in
  /// the entry for Criterion in
  /// Metavarialbes::factory_creation::factory_classes
  ///
  /// \note The ComputeTagsList of the ObservationBox should contain the union
  /// of the tags listed in `compute_tags_for_observation_box` for each derived
  /// Criterion listed in the `factory_classes`.
  template <typename ComputeTagsList, typename DataBoxType,
            typename Metavariables, typename Frame>
  auto evaluate(const ObservationBox<ComputeTagsList, DataBoxType>& box,
                Parallel::GlobalCache<Metavariables>& cache,
                const ylm::Strahlkorper<Frame>& strahlkorper,
                const FastFlow::IterInfo& info) const {
    using factory_classes =
        typename std::decay_t<Metavariables>::factory_creation::factory_classes;
    return call_with_dynamic_type<size_t, tmpl::at<factory_classes, Criterion>>(
        this, [&box, &cache, &strahlkorper, &info](auto* const criterion) {
          return apply(*criterion, box, cache, strahlkorper, info);
        });
  }
};
}  // namespace ah
