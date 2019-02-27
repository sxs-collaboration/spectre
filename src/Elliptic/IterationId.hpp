// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <functional>
#include <iosfwd>

#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
// #include "NumericalAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Elliptic {

/*!
 * \brief The factor to divide the `Elliptic::IterationId::value()` by to
 * recover the `ComponentTag`
 *
 * \details To identify observations we must encode the `Elliptic::IterationId`
 * in a single number. To this end we multiply each component by this factor
 * and sum them. Then the individual component can be recovered by dividing by
 * this factor and performing a `floor`.
 */
template <typename ComponentTag>
constexpr size_t iteration_id_value_factor = 0.;
template <>
constexpr size_t iteration_id_value_factor<LinearSolver::Tags::IterationId> = 1;
// The nonlinear solver is not merged yet
// template <>
// constexpr size_t
// iteration_id_value_factor<NonlinearSolver::Tags::IterationId> = 1e6;

/*!
 * \brief Identifies a step in an elliptic solve
 *
 * \details The elliptic solve is composed of nested iterative algorithms,
 * each of which keeps track of its own iteration id. This type combines the
 * individual iteration ids into a unique identifier within the full elliptic
 * solve.
 */
template <typename... ComponentTags>
struct IterationId : tuples::TaggedTuple<ComponentTags...> {
  using tuples::TaggedTuple<ComponentTags...>::TaggedTuple;

  using component_tags_list = tmpl::list<ComponentTags...>;

  /*!
   * \brief Encodes all components in a single number
   *
   * \details In this implementation we simply multiply each component iteration
   * id by a particular factor (see `iteration_id_value_factor`) and sum the
   * results. Therefore, the individual component may be recovered by dividing
   * by this factor and performing a `floor`.
   *
   * This implementation implies that we assume a maximum number of iterations
   * for each component. The maximum is the quotient between the
   * `iteration_id_value_factor` of the component and the next. In case we ever
   * need to relax this assumption we can instead represent the
   * `Elliptic::IterationId` by a single `size_t`. This number we would
   * construct from each component's number of completed sub-steps, which must
   * then be kept track of. For instance, if the elliptic solve is composed of
   * an iterative nonlinear solve, which performs a linear solve at each step,
   * we would need to keep track of the number of linear solver steps for
   * each nonlinear solver step. The unique `Elliptic::IterationId` would then
   * be the sum of those linear solver steps.
   */
  double value() const noexcept {
    size_t v = 0;
    tmpl::for_each<tmpl::list<ComponentTags...>>([&v, this ](
        auto component_tag) noexcept {
      using ComponentTag = tmpl::type_from<decltype(component_tag)>;
      v += iteration_id_value_factor<ComponentTag> * get<ComponentTag>(*this);
    });
    return v;
  }

  /*!
   * \brief Increment the specified component by 1, resetting all following
   * components to 0
   */
  template <typename IncrementComponentTag>
  IterationId<ComponentTags...> increment() noexcept {
    get<IncrementComponentTag>(*this)++;
    bool reset_minor_components = false;
    tmpl::for_each<tmpl::list<ComponentTags...>>(
        [&reset_minor_components, this ](auto component_tag) noexcept {
          using ComponentTag = tmpl::type_from<decltype(component_tag)>;
          if (reset_minor_components) {
            get<ComponentTag>(*this) = 0;
          } else if (IncrementComponentTag* derived =
                         dynamic_cast<IncrementComponentTag*>(component_tag)) {
            reset_minor_components = true;
          }
        });
  }
};

template <typename... ComponentTags>
size_t hash_value(const IterationId<ComponentTags...>& id) noexcept {
  size_t h = 0;
  tmpl::for_each<tmpl::list<ComponentTags...>>(
      [&h, &id ](auto component_tag) noexcept {
        using ComponentTag = tmpl::type_from<decltype(component_tag)>;
        boost::hash_combine(h, get<ComponentTag>(id));
      });
  return h;
}

}  // namespace Elliptic

namespace std {
template <typename... ComponentTags>
struct hash<Elliptic::IterationId<ComponentTags...>> {
  size_t operator()(const Elliptic::IterationId<ComponentTags...>& id) const
      noexcept {
    return boost::hash<Elliptic::IterationId<ComponentTags...>>{}(id);
  }
};
}  // namespace std
