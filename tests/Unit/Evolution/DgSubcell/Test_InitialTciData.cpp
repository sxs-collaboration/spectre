// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "Evolution/DgSubcell/InitialTciData.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Evolution/DgSubcell/Tags/InitialTciData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/MakeString.hpp"

namespace evolution::dg::subcell {
namespace {
template <size_t Dim>
void test() {
  using Tag = Tags::InitialTciData<Dim>;
  using Inbox = typename Tag::type;
  Inbox inbox{};

  const std::pair id0{Direction<Dim>::lower_xi(), ElementId<Dim>{0}};
  const std::pair id1{Direction<Dim>::lower_xi(), ElementId<Dim>{1}};
  const std::pair const_initial_tci_data{
      id0, InitialTciData{{100}, {RdmpTciData{{1.0}, {-1.0}}}}};
  Tag::insert_into_inbox(&inbox, 1, const_initial_tci_data);
  std::pair initial_tci_data{
      id1, InitialTciData{{200}, {RdmpTciData{{2.0}, {-2.0}}}}};
  Tag::insert_into_inbox(&inbox, 1, std::move(initial_tci_data));

  CHECK(inbox.at(1).at(id0).tci_status.value() == 100);
  CHECK(inbox.at(1).at(id0).initial_rdmp_data.value() ==
        RdmpTciData{{1.0}, {-1.0}});

  CHECK(inbox.at(1).at(id1).tci_status.value() == 200);
  CHECK(inbox.at(1).at(id1).initial_rdmp_data.value() ==
        RdmpTciData{{2.0}, {-2.0}});

  const std::string inbox_output = Tag::output_inbox(inbox, 1_st);
  const std::string expected_inbox_output_v1 =
      MakeString{} << " InitialTciDataInbox:\n"
                   << "  Action number: 1\n"
                   << "   Key: " << id0 << ", TCI: 100\n"
                   << "   Key: " << id1 << ", TCI: 200\n";
  const std::string expected_inbox_output_v2 =
      MakeString{} << " InitialTciDataInbox:\n"
                   << "  Action number: 1\n"
                   << "   Key: " << id1 << ", TCI: 200\n"
                   << "   Key: " << id0 << ", TCI: 100\n";
  // The inner map between key and TCI is unordered so we have to check both
  // possibilities
  CHECK((inbox_output == expected_inbox_output_v1 or
         inbox_output == expected_inbox_output_v2));
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.InitialTciStatus",
                  "[Evolution][Unit]") {
  {
    const InitialTciData initial_tci_data{{}, {}};
    const auto deserialize = serialize_and_deserialize(initial_tci_data);
    CHECK_FALSE(deserialize.tci_status.has_value());
    CHECK_FALSE(deserialize.initial_rdmp_data.has_value());
  }
  {
    const InitialTciData initial_tci_data{{100}, {}};
    const auto deserialize = serialize_and_deserialize(initial_tci_data);
    CHECK(deserialize.tci_status.value() == 100);
    CHECK_FALSE(deserialize.initial_rdmp_data.has_value());
  }
  {
    const InitialTciData initial_tci_data{{}, {RdmpTciData{{1.0}, {-1.0}}}};
    const auto deserialize = serialize_and_deserialize(initial_tci_data);
    CHECK_FALSE(deserialize.tci_status.has_value());
    CHECK(deserialize.initial_rdmp_data.value() == RdmpTciData{{1.0}, {-1.0}});
  }
  {
    const InitialTciData initial_tci_data{{100}, {RdmpTciData{{1.0}, {-1.0}}}};
    const auto deserialize = serialize_and_deserialize(initial_tci_data);
    CHECK(deserialize.tci_status.value() == 100);
    CHECK(deserialize.initial_rdmp_data.value() == RdmpTciData{{1.0}, {-1.0}});
  }

  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
}  // namespace evolution::dg::subcell
