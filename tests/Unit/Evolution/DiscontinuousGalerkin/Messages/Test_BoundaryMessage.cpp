// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <sstream>
#include <string>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg {
namespace {
template <size_t Dim, typename Generator>
void test_boundary_message(const gsl::not_null<Generator*> generator,
                           const size_t subcell_size, const size_t dg_size) {
  CAPTURE(Dim);
  CAPTURE(subcell_size);
  CAPTURE(dg_size);

  const size_t total_size_with_data =
      BoundaryMessage<Dim>::total_bytes_with_data(subcell_size, dg_size);
  CHECK(total_size_with_data == sizeof(BoundaryMessage<Dim>) +
                                    (subcell_size + dg_size) * sizeof(double));

  const bool owning = false;
  const bool enable_if_disabled = false;
  const size_t sender_node = 2;
  const size_t sender_core = 15;
  const int tci_status = -3;

  const Slab current_slab{0.1, 0.5};
  const Time current_time{current_slab, {0, 1}};
  const TimeStepId current_time_id{true, 0, current_time};
  const Slab next_slab{0.5, 0.9};
  const Time next_time{next_slab, {0, 1}};
  const TimeStepId next_time_id{true, 0, next_time};
  const Direction<Dim> neighbor_direction{0, Side::Upper};
  const ElementId<Dim> element_id{0};

  const size_t extents = 4;
  const Mesh<Dim> volume_mesh{extents, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim - 1> interface_mesh = volume_mesh.slice_away(0);

  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  auto subcell_data = make_with_random_values<DataVector>(
      generator, make_not_null(&dist), subcell_size);
  auto dg_data = make_with_random_values<DataVector>(
      generator, make_not_null(&dist), dg_size);
  // Have to copy data so we have different pointers inside BoundaryMessage to
  // different data that isn't deleted inside pack
  DataVector copied_subcell_data = subcell_data;
  DataVector copied_dg_data = dg_data;

  BoundaryMessage<Dim>* boundary_message = new BoundaryMessage<Dim>(
      subcell_size, dg_size, owning, enable_if_disabled, sender_node,
      sender_core, tci_status, current_time_id, next_time_id,
      neighbor_direction, element_id, volume_mesh, interface_mesh,
      subcell_size != 0 ? subcell_data.data() : nullptr,
      dg_size != 0 ? dg_data.data() : nullptr);
  // Since we expect the copied message to have owning = true because that's set
  // in the pack() function, we set owning = true here
  BoundaryMessage<Dim>* copied_boundary_message = new BoundaryMessage<Dim>(
      subcell_size, dg_size, true, enable_if_disabled, sender_node, sender_core,
      tci_status, current_time_id, next_time_id, neighbor_direction, element_id,
      volume_mesh, interface_mesh,
      subcell_size != 0 ? copied_subcell_data.data() : nullptr,
      dg_size != 0 ? copied_dg_data.data() : nullptr);

  CHECK(subcell_data.size() == subcell_size);
  CHECK(dg_data.size() == dg_size);

  void* packed_message = BoundaryMessage<Dim>::pack(boundary_message);

  BoundaryMessage<Dim>* unpacked_message =
      BoundaryMessage<Dim>::unpack(packed_message);

  CHECK(unpacked_message->owning);
  CHECK(*copied_boundary_message == *unpacked_message);
  CHECK_FALSE(*copied_boundary_message != *unpacked_message);

  BoundaryMessage<Dim>* repacked_unpacked_message =
      BoundaryMessage<Dim>::unpack(
          BoundaryMessage<Dim>::pack(unpacked_message));

  // Technically unpacked_message is now invalidated because we went through
  // pack/unpack, but we are only concerned that the pointers are the same. We
  // aren't using any data. These should be the same because packing an owning
  // message doesn't do any new allocations, and the unpack function also
  // doesn't do any new allocations, so the data shouldn't have moved
  CHECK(unpacked_message == repacked_unpacked_message);
}

void test_output() {
  const size_t subcell_size = 4;
  const size_t dg_size = 3;

  const bool owning = true;
  const bool enable_if_disabled = false;
  const size_t sender_node = 2;
  const size_t sender_core = 15;
  const int tci_status = -3;

  const Slab current_slab{0.1, 0.5};
  const Time current_time{current_slab, {0, 1}};
  const TimeStepId current_time_id{true, 0, current_time};
  const Slab next_slab{0.5, 0.9};
  const Time next_time{next_slab, {0, 1}};
  const TimeStepId next_time_id{true, 0, next_time};
  const Direction<2> neighbor_direction{0, Side::Upper};
  const ElementId<2> element_id{0};

  const size_t extents = 4;
  const Mesh<2> volume_mesh{extents, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
  const Mesh<1> interface_mesh = volume_mesh.slice_away(0);

  DataVector subcell_data{0.1, 0.2, 0.3, 0.4};
  DataVector dg_data{-0.3, -0.2, -0.1};

  BoundaryMessage<2> message{subcell_size,   dg_size,
                             owning,         enable_if_disabled,
                             sender_node,    sender_core,
                             tci_status,     current_time_id,
                             next_time_id,   neighbor_direction,
                             element_id,     volume_mesh,
                             interface_mesh, subcell_data.data(),
                             dg_data.data()};

  const std::string message_str = get_output(message);

  std::stringstream ss;
  ss << "subcell_ghost_data_size = 4\n"
     << "dg_flux_data_size = 3\n"
     << "owning = true\n"
     << "enable_if_disabled = false\n"
     << "sender_node = 2\n"
     << "sender_core = 15\n"
     << "tci_status = -3\n"
     // TimeStepIds have complicated output so don't try and hard code it, just
     // use get_output
     << "current_time_ste_id = " << get_output(current_time_id) << "\n"
     << "next_time_ste_id = " << get_output(next_time_id) << "\n"
     << "neighbor_direction = +0\n"
     << "element_id = [B0,(L0I0,L0I0)]\n"
     << "volume_or_ghost_mesh = "
        "[(4,4),(Legendre,Legendre),(GaussLobatto,GaussLobatto)]\n"
     << "interface_mesh = [(4),(Legendre),(GaussLobatto)]\n"
     << "subcell_ghost_data = (0.1,0.2,0.3,0.4)\n"
     << "dg_flux_data = (-0.3,-0.2,-0.1)";

  CHECK(message_str == ss.str());
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.BoundaryMessage", "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);

  test_output();

  std::uniform_int_distribution<size_t> size_dist{1, 10};
  tmpl::for_each<tmpl::integral_list<size_t, 1, 2, 3>>(
      [&generator, &size_dist](auto dim_t) {
        constexpr size_t Dim =
            tmpl::type_from<std::decay_t<decltype(dim_t)>>::value;
        // Only subcell data
        test_boundary_message<Dim>(make_not_null(&generator),
                                   size_dist(generator), 0);
        // Only dg data
        test_boundary_message<Dim>(make_not_null(&generator), 0,
                                   size_dist(generator));
        // Both subcell and dg data
        test_boundary_message<Dim>(make_not_null(&generator),
                                   size_dist(generator), size_dist(generator));
        // Neither subcell nor dg data. This isn't currently a use case, but we
        // test it for completeness to ensure pack/unpack are doing the correct
        // thing
        test_boundary_message<Dim>(make_not_null(&generator), 0, 0);
      });
}
}  // namespace
}  // namespace evolution::dg
