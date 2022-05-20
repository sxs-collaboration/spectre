// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/IO/Observers/MockH5.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::observers {
/*!
 * Tag that holds a MockH5File object for the MockObserverWriter
 */
struct MockReductionFileTag : ::db::SimpleTag {
  using type = MockH5File;
};

/*!
 * \brief Action meant to mock WriteReductionDataRow.
 *
 * \details Instead of writing to disk, this action will mutate the
 * MockReductionFileTag stored in the DataBox of the MockObserverWriter with the
 * values passed in. We want to avoid writing to disk in the testing framework
 * because IO is slow and we don't really care that things were written to disk,
 * just that the correct values were recorded.
 *
 * To check what values were written, get the MockDat file:
 *
 * \snippet Test_MockWriteReductionDataRow.cpp check_mock_writer_data
 */
struct MockWriteReductionDataRow {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex, typename... Ts>
  static void apply(::db::DataBox<DbTagsList>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const std::string& subfile_name,
                    std::vector<std::string>&& legend,
                    std::tuple<Ts...>&& in_reduction_data) {
    if constexpr (::db::tag_is_retrievable_v<MockReductionFileTag,
                                             ::db::DataBox<DbTagsList>>) {
      ::db::mutate<MockReductionFileTag>(
          make_not_null(&box),
          [subfile_name, legend,
           in_reduction_data](const gsl::not_null<MockH5File*> mock_h5_file) {
            auto& dat_file = (*mock_h5_file).try_insert(subfile_name);

            std::vector<double> data{};
            tmpl::for_each<tmpl::range<size_t, 0, sizeof...(Ts)>>(
                [&data, &in_reduction_data](auto size_holder) {
                  constexpr size_t index =
                      std::decay_t<decltype(size_holder)>::type::value;
                  ::observers::ThreadedActions::ReductionActions_detail::
                      append_to_reduction_data(
                          make_not_null(&data),
                          std::get<index>(in_reduction_data));
                });

            dat_file.append(legend, data);
          });
    } else {
      (void)subfile_name;
      (void)legend;
      (void)in_reduction_data;
      ERROR(
          "Wrong DataBox. Cannot retrieve the MockReductionFileTag. Expecting "
          "DataBox for MockObserverWriter.");
    }
  }
};

/*!
 * \brief Component that mocks the ObserverWriter.
 *
 * \details The only tag that is added to the DataBox is the
 * MockReductionFileTag. To initialize this component do
 *
 * \snippet Test_MockWriteReductionDataRow.cpp initialize_component
 *
 * This component replaces the WriteReductionDataRow threaded action with the
 * MockWriteReductionDataRow threaded action.
 */
template <typename Metavariables>
struct MockObserverWriter {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<
          ActionTesting::InitializeDataBox<tmpl::list<MockReductionFileTag>>>>>;
  using component_being_mocked = ::observers::ObserverWriter<Metavariables>;

  using replace_these_threaded_actions =
      tmpl::list<::observers::ThreadedActions::WriteReductionDataRow>;
  using with_these_threaded_actions = tmpl::list<MockWriteReductionDataRow>;
};
}  // namespace TestHelpers::observers
