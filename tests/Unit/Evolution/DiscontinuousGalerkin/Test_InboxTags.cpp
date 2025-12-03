// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <unordered_set>
#include <utility>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Time/Slab.hpp"
#include "Time/TimeStepId.hpp"

namespace {
enum class SendType { GhostData, DgData, AllData, SplitDgData };

template <size_t Dim>
evolution::dg::BoundaryData<Dim> make_boundary_data(const int label,
                                                    const TimeStepId& next_step,
                                                    const SendType type) {
  return {Mesh<Dim>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
          (type != SendType::DgData)
              ? std::optional{Mesh<Dim>{3, Spectral::Basis::Legendre,
                                        Spectral::Quadrature::Gauss}}
              : std::nullopt,
          (type != SendType::GhostData)
              ? std::optional{Mesh<Dim - 1>{5, Spectral::Basis::Legendre,
                                            Spectral::Quadrature::Gauss}}
              : std::nullopt,
          (type == SendType::GhostData or type == SendType::AllData)
              ? std::optional{DataVector{static_cast<double>(label)}}
              : std::nullopt,
          (type != SendType::GhostData)
              ? std::optional{DataVector{static_cast<double>(-label)}}
              : std::nullopt,
          next_step,
          0,
          3,
          std::nullopt};
}

template <size_t Dim, bool UseNodegroupDgElements>
void test() {
  CAPTURE(Dim);
  CAPTURE(UseNodegroupDgElements);

  const Slab slab(1.2, 3.4);
  const TimeStepId time_step_1(true, 5, slab.start());
  const TimeStepId time_step_2 =
      time_step_1.next_substep(slab.duration() / 2, 0.3);
  const TimeStepId time_step_3 = time_step_2.next_step(slab.duration() / 2);
  const TimeStepId time_step_4 =
      time_step_3.next_substep(slab.duration() / 2, 0.3);

  const ElementId<Dim> element_upper(2);
  const ElementId<Dim> element_lower(4);
  const DirectionalId<Dim> mortar_upper{Direction<Dim>::upper_xi(),
                                        element_upper};
  const DirectionalId<Dim> mortar_lower{Direction<Dim>::lower_xi(),
                                        element_lower};

  using Inbox = evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
      Dim, UseNodegroupDgElements>;

  typename Inbox::type inbox{};

  const auto data_upper_1 =
      make_boundary_data<Dim>(0, time_step_2, SendType::AllData);
  CHECK(not Inbox::insert_into_inbox(&inbox, time_step_1,
                                     std::pair{mortar_upper, data_upper_1}));
  inbox.collect_messages();
  const auto data_lower_2 =
      make_boundary_data<Dim>(1, time_step_3, SendType::DgData);
  CHECK(not Inbox::insert_into_inbox(&inbox, time_step_2,
                                     std::pair{mortar_lower, data_lower_2}));

  CHECK(not inbox.set_missing_messages(3));

  const auto data_lower_3 =
      make_boundary_data<Dim>(2, time_step_4, SendType::DgData);
  CHECK(not Inbox::insert_into_inbox(&inbox, time_step_3,
                                     std::pair{mortar_lower, data_lower_3}));
  const auto data_lower_1 =
      make_boundary_data<Dim>(3, time_step_2, SendType::DgData);
  CHECK(Inbox::insert_into_inbox(&inbox, time_step_1,
                                 std::pair{mortar_lower, data_lower_1}));
  const auto data_upper_3 =
      make_boundary_data<Dim>(4, time_step_4, SendType::GhostData);
  CHECK(not Inbox::insert_into_inbox(&inbox, time_step_3,
                                     std::pair{mortar_upper, data_upper_3}));

  inbox.collect_messages();

  CHECK(inbox.messages.size() == 3);
  {
    const auto& time1_messages = inbox.messages.at(time_step_1);
    CHECK(time1_messages.size() == 2);
    CHECK(time1_messages.at(mortar_upper) == data_upper_1);
    CHECK(time1_messages.at(mortar_lower) == data_lower_1);
  }
  {
    const auto& time2_messages = inbox.messages.at(time_step_2);
    CHECK(time2_messages.size() == 1);
    CHECK(time2_messages.at(mortar_lower) == data_lower_2);
  }
  {
    const auto& time3_messages = inbox.messages.at(time_step_3);
    CHECK(time3_messages.size() == 2);
    CHECK(time3_messages.at(mortar_upper) == data_upper_3);
    CHECK(time3_messages.at(mortar_lower) == data_lower_3);
  }

  CHECK(not inbox.set_missing_messages(1));

  CHECK(Inbox::insert_into_inbox(
      &inbox, time_step_3,
      std::pair{mortar_upper, make_boundary_data<Dim>(4, time_step_4,
                                                      SendType::SplitDgData)}));

  inbox.collect_messages();

  CHECK(inbox.messages.size() == 3);
  CHECK(inbox.messages.at(time_step_1).size() == 2);
  CHECK(inbox.messages.at(time_step_2).size() == 1);
  CHECK(inbox.messages.at(time_step_3).size() == 2);

  CHECK(inbox.messages.at(time_step_3).at(mortar_upper) ==
        make_boundary_data<Dim>(4, time_step_4, SendType::AllData));

  const auto data_upper_2 =
      make_boundary_data<Dim>(5, time_step_2, SendType::GhostData);
  CHECK(not Inbox::insert_into_inbox(&inbox, time_step_2,
                                     std::pair{mortar_upper, data_upper_3}));

  CHECK(inbox.set_missing_messages(1));
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.InboxTags", "[Unit][Evolution]") {
  test<1, false>();
  test<2, false>();
  test<3, false>();
  test<1, true>();
  test<2, true>();
  test<3, true>();
}
}  // namespace
