// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/EventsAndDenseTriggers/EventsAndDenseTriggers.hpp"
#include "Evolution/EventsAndDenseTriggers/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/EvolutionOrdering.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
class DataVector;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
struct TimeStep;
}  // namespace Tags
/// \endcond

namespace evolution::Actions {
/// \ingroup ActionsGroup
/// \ingroup EventsAndTriggersGroup
/// \brief Run the events and dense triggers
///
/// If dense output is required, each `postprocessor` in the \p
/// Postprocessors list will be called as
/// `postprocessor::is_ready(make_not_null(&box),
/// make_not_null(&inboxes), cache, array_index, component)`.  If it
/// returns false, the algorithm will be stopped to wait for more
/// data.  After performing dense output, each of the \p
/// Postprocessors will be passed to `db::mutate_apply` on the
/// DataBox.  The wrapper struct `AlwaysReadyPostprocessor` is
/// provided for convenience to provide an `is_ready` function when a
/// pure mutate-apply is desired.
///
/// At the end of the action, the values of the time, evolved
/// variables, and anything appearing in the `return_tags` of the \p
/// Postprocessors will be restored to their initial values.
///
/// Uses:
/// - DataBox: EventsAndDenseTriggers and as required by events,
///   triggers, and postprocessors
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: as performed by the postprocessor `is_ready` functions
template <typename Postprocessors>
struct RunEventsAndDenseTriggers {
 private:
  static_assert(tt::is_a_v<tmpl::list, Postprocessors>);

  // RAII object to restore the time and variables changed by dense
  // output.
  template <typename DbTags, typename Tags>
  class StateRestorer {
    template <typename Tag, bool IsVariables>
    struct expand_variables_impl {
      using type = typename Tag::tags_list;
    };

    template <typename Tag>
    struct expand_variables_impl<Tag, false> {
      using type = tmpl::list<Tag>;
    };

    template <typename Tag>
    struct expand_variables
        : expand_variables_impl<Tag,
                                tt::is_a_v<Variables, typename Tag::type>> {};

    using expanded_tags = tmpl::remove_duplicates<
        tmpl::join<tmpl::transform<Tags, expand_variables<tmpl::_1>>>>;
    using tensors_and_non_tensors = tmpl::partition<
        expanded_tags,
        tmpl::bind<
            tmpl::apply,
            tmpl::if_<tt::is_a<Tensor, tmpl::bind<tmpl::type_from, tmpl::_1>>,
                      tmpl::defer<tmpl::parent<std::is_same<
                          tmpl::bind<tmpl::type_from,
                                     tmpl::bind<tmpl::type_from, tmpl::_1>>,
                          DataVector>>>,
                      std::false_type>>>;
    using tensor_tags = tmpl::front<tensors_and_non_tensors>;
    using non_tensor_tags = tmpl::back<tensors_and_non_tensors>;

   public:
    StateRestorer(const gsl::not_null<db::DataBox<DbTags>*> box) : box_(box) {}

    void save() {
      // Only store the value the first time, because after that we
      // are seeing the value after the previous change instead of the
      // original.
      if (not non_tensors_.has_value()) {
        if constexpr (not std::is_same_v<tensor_tags, tmpl::list<>>) {
          tensors_.initialize(
              db::get<tmpl::front<tensor_tags>>(*box_).begin()->size());
          tmpl::for_each<tensor_tags>([this](auto tag_v) {
            using tag = tmpl::type_from<decltype(tag_v)>;
            get<tag>(tensors_) = db::get<tag>(*box_);
          });
        }
        tmpl::as_pack<non_tensor_tags>([this](auto... tags_v) {
          non_tensors_.emplace(
              db::get<tmpl::type_from<decltype(tags_v)>>(*box_)...);
        });
      }
    }

    // WARNING: Manually calling this if there are non_tensor_tags
    // will cause use-after-moves.
    void restore() {
      if (non_tensors_.has_value()) {
        tmpl::for_each<tensor_tags>([this](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          db::mutate<tag>(
              box_, [this](const gsl::not_null<typename tag::type*> value) {
                *value = get<tag>(tensors_);
              });
        });
        tmpl::for_each<non_tensor_tags>([this](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          db::mutate<tag>(
              box_, [this](const gsl::not_null<typename tag::type*> value) {
                *value = std::move(tuples::get<tag>(*non_tensors_));
              });
        });
      }
    }

    ~StateRestorer() { restore(); }

   private:
    gsl::not_null<db::DataBox<DbTags>*> box_ = nullptr;
    // Store all tensors in a single allocation.
    Variables<tensor_tags> tensors_{};
    std::optional<tuples::tagged_tuple_from_typelist<non_tensor_tags>>
        non_tensors_;
  };

  template <typename T>
  struct get_return_tags {
    using type = typename T::return_tags;
  };

 public:
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const component) {
    using system = typename Metavariables::system;
    using variables_tag = typename system::variables_tag;

    const auto& time_step_id = db::get<::Tags::TimeStepId>(box);
    if (time_step_id.slab_number() < 0) {
      // Skip dense output during self-start
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    auto& events_and_dense_triggers =
        db::get_mutable_reference<::evolution::Tags::EventsAndDenseTriggers>(
            make_not_null(&box));

    const auto step_end =
        time_step_id.step_time() + db::get<::Tags::TimeStep>(box);
    const evolution_less<double> before{time_step_id.time_runs_forward()};

    using postprocessor_return_tags =
        tmpl::join<tmpl::transform<Postprocessors, get_return_tags<tmpl::_1>>>;
    // The evolved variables will be restored anyway, so no reason to
    // copy them twice.
    using postprocessor_restore_tags =
        tmpl::list_difference<postprocessor_return_tags,
                              typename variables_tag::tags_list>;

    StateRestorer<DbTags, tmpl::list<::Tags::Time>> time_restorer(
        make_not_null(&box));
    StateRestorer<DbTags, tmpl::list<variables_tag>> variables_restorer(
        make_not_null(&box));
    StateRestorer<DbTags, postprocessor_restore_tags> postprocessor_restorer(
        make_not_null(&box));

    for (;;) {
      const double next_trigger = events_and_dense_triggers.next_trigger(box);
      if (before(step_end.value(), next_trigger)) {
        return {Parallel::AlgorithmExecution::Continue, std::nullopt};
      }

      // This can only be true the first time through the loop,
      // because triggers are not allowed to reschedule for the time
      // they just triggered at.  This check is primarily to avoid
      // special-case bookkeeping for the initial simulation time.
      const bool already_at_correct_time =
          db::get<::Tags::Time>(box) == next_trigger;
      if (not already_at_correct_time) {
        time_restorer.save();
        db::mutate<::Tags::Time>(
            make_not_null(&box),
            [&next_trigger](const gsl::not_null<double*> time) {
              *time = next_trigger;
            });
      }

      const auto triggered = events_and_dense_triggers.is_ready(
          box, cache, array_index, component);
      using TriggeringState = std::decay_t<decltype(triggered)>;
      switch (triggered) {
        case TriggeringState::NotReady:
          return {Parallel::AlgorithmExecution::Retry, std::nullopt};
        case TriggeringState::NeedsEvolvedVariables:
          if (not already_at_correct_time) {
            bool ready = true;
            tmpl::for_each<Postprocessors>([&](auto postprocessor_v) {
              using postprocessor = tmpl::type_from<decltype(postprocessor_v)>;
              if (ready) {
                if (not postprocessor::is_ready(make_not_null(&box),
                                                make_not_null(&inboxes), cache,
                                                array_index, component)) {
                  ready = false;
                }
              }
            });
            if (not ready) {
              return {Parallel::AlgorithmExecution::Retry, std::nullopt};
            }

            using history_tag = ::Tags::HistoryEvolvedVariables<variables_tag>;
            bool dense_output_succeeded = false;
            variables_restorer.save();
            db::mutate<variables_tag>(
                make_not_null(&box),
                [&dense_output_succeeded, &next_trigger](
                    gsl::not_null<typename variables_tag::type*> vars,
                    const TimeStepper& stepper,
                    const typename history_tag::type& history) {
                  dense_output_succeeded =
                      stepper.dense_update_u(vars, history, next_trigger);
                },
                db::get<::Tags::TimeStepper<>>(box), db::get<history_tag>(box));
            if (not dense_output_succeeded) {
              // Need to take another time step
              return {Parallel::AlgorithmExecution::Continue, std::nullopt};
            }

            postprocessor_restorer.save();
            tmpl::for_each<Postprocessors>([&box](auto postprocessor_v) {
              using postprocessor = tmpl::type_from<decltype(postprocessor_v)>;
              db::mutate_apply<postprocessor>(make_not_null(&box));
            });
          }
          [[fallthrough]];
        default:
          break;
      }

      events_and_dense_triggers.run_events(box, cache, array_index, component);
      if (not events_and_dense_triggers.reschedule(box, cache, array_index,
                                                   component)) {
        return {Parallel::AlgorithmExecution::Retry, std::nullopt};
      }
    }
  }
};

struct InitializeRunEventsAndDenseTriggers {
  using simple_tags_from_options =
      tmpl::list<evolution::Tags::EventsAndDenseTriggers>;
  using simple_tags = tmpl::list<Tags::PreviousTriggerTime>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*component*/) {
    Initialization::mutate_assign<simple_tags>(make_not_null(&box),
                                               std::nullopt);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace evolution::Actions

/// A wrapper adding an always-true `is_ready` function for a
/// `RunEventsAndDenseTriggers` postprocessor.  This allows structs
/// designed as mutate_apply arguments to be used without
/// modification.
template <typename T>
struct AlwaysReadyPostprocessor : T {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ParallelComponent>
  static bool is_ready(
      const gsl::not_null<db::DataBox<DbTagsList>*> /*box*/,
      const gsl::not_null<tuples::TaggedTuple<InboxTags...>*> /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const ParallelComponent* const /*component*/) {
    return true;
  }
};
