// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <sstream>
#include <string>

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

  const size_t total_size_without_data =
      BoundaryMessage<Dim>::total_bytes_without_data();
  // Mesh<0> == 8, Mesh<1> == 16, Mesh<2> == 32, Mesh<3> == 48
  CHECK(total_size_without_data == (Dim == 1 ? 256 : (Dim == 2 ? 280 : 312)));
  const size_t total_size_with_data =
      BoundaryMessage<Dim>::total_bytes_with_data(subcell_size, dg_size);
  CHECK(total_size_with_data ==
        total_size_without_data + (subcell_size + dg_size) * sizeof(double));

  const bool sent_across_nodes = true;
  const size_t sender_node = 2;
  const size_t sender_core = 15;

  const Slab current_slab{0.1, 0.5};
  const Time current_time{current_slab, {0, 1}};
  const TimeStepId current_time_id{true, 0, current_time};
  const Slab next_slab{0.5, 0.9};
  const Time next_time{next_slab, {0, 1}};
  const TimeStepId next_time_id{true, 0, next_time};

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
      subcell_size, dg_size, sent_across_nodes, sender_node, sender_core,
      current_time_id, next_time_id, volume_mesh, interface_mesh,
      subcell_size != 0 ? subcell_data.data() : nullptr,
      dg_size != 0 ? dg_data.data() : nullptr);
  BoundaryMessage<Dim>* copied_boundary_message = new BoundaryMessage<Dim>(
      subcell_size, dg_size, sent_across_nodes, sender_node, sender_core,
      current_time_id, next_time_id, volume_mesh, interface_mesh,
      subcell_size != 0 ? copied_subcell_data.data() : nullptr,
      dg_size != 0 ? copied_dg_data.data() : nullptr);

  CHECK(subcell_data.size() == subcell_size);
  CHECK(dg_data.size() == dg_size);

  void* packed_message = boundary_message->pack(boundary_message);

  BoundaryMessage<Dim>* unpacked_message =
      boundary_message->unpack(packed_message);

  CHECK(*copied_boundary_message == *unpacked_message);
  CHECK_FALSE(*copied_boundary_message != *unpacked_message);
}

void test_output() {
  const size_t subcell_size = 4;
  const size_t dg_size = 3;

  const bool sent_across_nodes = true;
  const size_t sender_node = 2;
  const size_t sender_core = 15;

  const Slab current_slab{0.1, 0.5};
  const Time current_time{current_slab, {0, 1}};
  const TimeStepId current_time_id{true, 0, current_time};
  const Slab next_slab{0.5, 0.9};
  const Time next_time{next_slab, {0, 1}};
  const TimeStepId next_time_id{true, 0, next_time};

  const size_t extents = 4;
  const Mesh<2> volume_mesh{extents, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
  const Mesh<1> interface_mesh = volume_mesh.slice_away(0);

  DataVector subcell_data{0.1, 0.2, 0.3, 0.4};
  DataVector dg_data{-0.3, -0.2, -0.1};

  BoundaryMessage<2> message{
      subcell_size,        dg_size,       sent_across_nodes,
      sender_node,         sender_core,   current_time_id,
      next_time_id,        volume_mesh,   interface_mesh,
      subcell_data.data(), dg_data.data()};

  const std::string message_str = get_output(message);

  std::stringstream ss;
  ss << "subcell_ghost_data_size = 4\n"
     << "dg_flux_data_size = 3\n"
     << "sent_across_nodes = true\n"
     << "sender_node = 2\n"
     << "sender_core = 15\n"
     // TimeStepIds have complicated output so don't try and hard code it, just
     // use get_output
     << "current_time_ste_id = " << get_output(current_time_id) << "\n"
     << "next_time_ste_id = " << get_output(next_time_id) << "\n"
     << "volume_or_ghost_mesh = "
        "[(4,4),(Legendre,Legendre),(GaussLobatto,GaussLobatto)]\n"
     << "interface_mesh = [(4),(Legendre),(GaussLobatto)]\n"
     << "subcell_ghost_data = (0.1,0.2,0.3,0.4)\n"
     << "dg_flux_data = (-0.3,-0.2,-0.1)";

  CHECK(message_str == ss.str());
}

void test_offset() {
  CHECK(detail::offset<size_t>() == 8);
  CHECK(detail::offset<bool>() == 8);
  CHECK(detail::offset<TimeStepId>() == 88);
  CHECK(detail::offset<Mesh<0>>() == 8);
  CHECK(detail::offset<Mesh<1>>() == 16);
  CHECK(detail::offset<Mesh<2>>() == 32);
  CHECK(detail::offset<Mesh<3>>() == 48);
  CHECK(detail::offset<double*>() == 8);

  CHECK_THROWS_WITH(
      detail::offset<int>(),
      Catch::Contains(
          "Cannot calculate offset for 'int' in a BoundaryMessage"));
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.BoundaryMessage", "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);

  test_output();
  test_offset();

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
