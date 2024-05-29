// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <type_traits>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
class LtsTimeStepper;
class TimeStepper;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace Tags {
template <typename StepperInterface>
struct TimeStepper;
template <typename Tag>
struct HistoryEvolvedVariables;
}  // namespace Tags
namespace evolution::dg::Tags {
template <size_t Dim, typename CouplingResult>
struct MortarDataHistory;
}  // namespace evolution::dg::Tags
namespace imex::Tags {
template <typename ImplicitSector>
struct ImplicitHistory;
}  // namespace imex::Tags
namespace imex::protocols {
struct ImexSystem;
}  // namespace imex::protocols
namespace tuples {
template <class... Tags>
class TaggedTuple;
}  // namespace tuples
/// \endcond

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup TimeGroup
/// \brief Clean time stepper history after a substep
///
/// Uses:
/// - DataBox:
///   - Tags::TimeStepper<TimeStepper>
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - Tags::HistoryEvolvedVariables<variables_tag>
///   - imex::Tags::ImplicitHistory<sector> for each IMEX sector
///   - evolution::dg::Tags::MortarDataHistory<...> if CleanBoundaryHistory
template <typename System, bool CleanBoundaryHistory>
struct CleanHistory {
 private:
  template <typename T>
  struct is_mortar_data_history : std::false_type {};

  template <size_t Dim, typename CouplingResult>
  struct is_mortar_data_history<
      evolution::dg::Tags::MortarDataHistory<Dim, CouplingResult>>
      : std::true_type {};

 public:
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {  // NOLINT const
    const auto& time_stepper = db::get<Tags::TimeStepper<TimeStepper>>(box);

    using variables_tags = tmpl::conditional_t<
        tt::is_a_v<tmpl::list, typename System::variables_tag>,
        typename System::variables_tag,
        tmpl::list<typename System::variables_tag>>;
    using history_tags =
        tmpl::transform<variables_tags,
                        tmpl::bind<Tags::HistoryEvolvedVariables, tmpl::_1>>;
    db::mutate_apply<history_tags, tmpl::list<>>(
        [&time_stepper](const auto... histories) {
          expand_pack((time_stepper.clean_history(histories), 0)...);
        },
        make_not_null(&box));

    // This action can't depend on Evolution, but most of the time
    // stepping structures are in Evolution, so everything past here
    // has to only depend on Evolution classes through the DataBox
    // type.

    if constexpr (tt::conforms_to_v<System, imex::protocols::ImexSystem>) {
      using implicit_history_tags =
          tmpl::transform<typename System::implicit_sectors,
                          tmpl::bind<imex::Tags::ImplicitHistory, tmpl::_1>>;
      db::mutate_apply<implicit_history_tags, tmpl::list<>>(
          [&time_stepper](const auto... histories) {
            expand_pack((time_stepper.clean_history(histories), 0)...);
          },
          make_not_null(&box));
    }

    if constexpr (CleanBoundaryHistory) {
      using boundary_history_tags =
          tmpl::filter<DbTags, is_mortar_data_history<tmpl::_1>>;
      // First type is expanded by some compilers in the error
      // message.
      static_assert(((void)boundary_history_tags{},
                     tmpl::size<boundary_history_tags>::value == 1));
      db::mutate_apply<boundary_history_tags,
                       tmpl::list<Tags::TimeStepper<LtsTimeStepper>>>(
          [](const auto history, const auto& lts_time_stepper) {
            for (auto& mortar : *history) {
              lts_time_stepper.clean_boundary_history(
                  make_not_null(&mortar.second));
            }
          },
          make_not_null(&box));
    }

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
