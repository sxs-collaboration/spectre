// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DiscontinuousGalerkin/InboxTags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg {
namespace {

template <size_t Dim>
BoundaryMessage<Dim>* create_boundary_message(
    const TimeStepId& current_time_step_id, const TimeStepId& next_time_step_id,
    const std::pair<Direction<Dim>, ElementId<Dim>>& key,
    const Mesh<Dim>& volume_mesh, const Mesh<Dim - 1>& interface_mesh,
    std::optional<DataVector>& ghost_data, std::optional<DataVector>& dg_data,
    const int tci_status) {
  return new BoundaryMessage<Dim>(
      ghost_data.value_or(DataVector{}).size(),  // subcell_ghost_data_size
      dg_data.value_or(DataVector{}).size(),     // dg_flux_data_size
      true,                                      // owning
      false,                                     // enable_if_disabled
      2,                                         // sender_node
      12,                                        // sender_core
      tci_status,                                // tci_status
      current_time_step_id,                      // current_time_step_id
      next_time_step_id,                         // next_time_step_id
      key.first,                                 // neighbor_direction
      key.second,                                // element_id
      volume_mesh,                               // volume_or_ghost_mesh
      interface_mesh,                            // interface_mesh
      ghost_data.has_value() ? ghost_data.value().data()
                             : nullptr,  // subcell_ghost_data
      dg_data.has_value() ? dg_data.value().data() : nullptr  // dg_flux_data
  );
}

template <size_t Dim>
void test_no_ghost_cells() {
  static constexpr size_t number_of_components = 1 + Dim;
  using bc_tag = Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>;
  using bm_tag = Tags::BoundaryMessageInbox<Dim>;
  using BcType = std::tuple<Mesh<Dim>, Mesh<Dim - 1>, std::optional<DataVector>,
                            std::optional<DataVector>, ::TimeStepId, int>;
  using BcInbox = typename bc_tag::type;
  using BmInbox = typename bm_tag::type;

  std::uniform_real_distribution<double> dist(-1.0, 2.3);
  MAKE_GENERATOR(gen);
  std::optional<DataVector> nullopt = std::nullopt;

  const TimeStepId time_step_id_a{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};
  const TimeStepId time_step_id_b{true, 4, Time{Slab{3.4, 5.4}, {13, 100}}};
  const TimeStepId time_step_id_c{true, 5, Time{Slab{5.4, 6.4}, {17, 100}}};
  const std::pair nhbr_key{Direction<Dim>::lower_xi(), ElementId<Dim>{1}};

  BcInbox bc_inbox{};
  BmInbox bm_inbox{};

  BcType send_data_a{};
  const Mesh<Dim> volume_mesh_a{5, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim - 1> mesh_a{5, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
  get<0>(send_data_a) = volume_mesh_a;
  get<1>(send_data_a) = mesh_a;
  get<3>(send_data_a) =
      DataVector{mesh_a.number_of_grid_points() * number_of_components, 0.0};
  get<4>(send_data_a) = time_step_id_a;
  get<5>(send_data_a) = 5;
  fill_with_random_values(make_not_null(&*get<3>(send_data_a)),
                          make_not_null(&gen), make_not_null(&dist));

  BoundaryMessage<Dim>* boundary_message_a = create_boundary_message(
      time_step_id_a, time_step_id_a, nhbr_key, volume_mesh_a, mesh_a, nullopt,
      get<3>(send_data_a), get<5>(send_data_a));
  BoundaryMessage<Dim>* boundary_message_a_compare = boundary_message_a;

  bc_tag::insert_into_inbox(make_not_null(&bc_inbox), time_step_id_a,
                            std::make_pair(nhbr_key, send_data_a));
  bm_tag::insert_into_inbox(make_not_null(&bm_inbox), boundary_message_a);

  CHECK((bc_inbox.at(time_step_id_a).at(nhbr_key) == send_data_a));
  // Check the values, not the pointers
  CHECK(*(bm_inbox.at(time_step_id_a).at(nhbr_key).get()) ==
        *boundary_message_a_compare);

  BcType send_data_b{};
  const Mesh<Dim> volume_mesh_b{7, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim - 1> mesh_b{7, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
  get<0>(send_data_b) = volume_mesh_b;
  get<1>(send_data_b) = mesh_b;

  get<3>(send_data_b) =
      DataVector{mesh_b.number_of_grid_points() * number_of_components, 0.0};
  // Set the future time step to make sure the implementation doesn't mix the
  // receive time ID and the validity range time ID
  get<4>(send_data_b) = time_step_id_c;
  get<5>(send_data_a) = 5;
  fill_with_random_values(make_not_null(&*get<3>(send_data_b)),
                          make_not_null(&gen), make_not_null(&dist));

  BoundaryMessage<Dim>* boundary_message_b = create_boundary_message(
      time_step_id_b, time_step_id_c, nhbr_key, volume_mesh_b, mesh_b, nullopt,
      get<3>(send_data_b), get<5>(send_data_b));
  BoundaryMessage<Dim>* boundary_message_b_compare = boundary_message_b;

  bc_tag::insert_into_inbox(make_not_null(&bc_inbox), time_step_id_b,
                            std::make_pair(nhbr_key, send_data_b));
  bm_tag::insert_into_inbox(make_not_null(&bm_inbox),
                            boundary_message_b_compare);

  CHECK((bc_inbox.at(time_step_id_a).at(nhbr_key) == send_data_a));
  CHECK((bc_inbox.at(time_step_id_b).at(nhbr_key) == send_data_b));
  CHECK(*(bm_inbox.at(time_step_id_a).at(nhbr_key).get()) ==
        *boundary_message_a_compare);
  CHECK(*(bm_inbox.at(time_step_id_b).at(nhbr_key).get()) ==
        *boundary_message_b_compare);

  bc_inbox.erase(time_step_id_a);
  bm_inbox.erase(time_step_id_a);
  CHECK(bc_inbox.count(time_step_id_a) == 0);
  CHECK(bm_inbox.count(time_step_id_a) == 0);

  CHECK((bc_inbox.at(time_step_id_b).at(nhbr_key) == send_data_b));
  CHECK(*(bm_inbox.at(time_step_id_b).at(nhbr_key).get()) ==
        *boundary_message_b_compare);
  bc_inbox.erase(time_step_id_b);
  bm_inbox.erase(time_step_id_b);
  CHECK(bc_inbox.count(time_step_id_b) == 0);
  CHECK(bm_inbox.count(time_step_id_b) == 0);
}

template <size_t Dim>
void test_with_ghost_cells() {
  static constexpr size_t number_of_components = 1 + Dim;
  using bc_tag = Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>;
  using bm_tag = Tags::BoundaryMessageInbox<Dim>;
  using BcType = std::tuple<Mesh<Dim>, Mesh<Dim - 1>, std::optional<DataVector>,
                            std::optional<DataVector>, ::TimeStepId, int>;
  using BcInbox = typename bc_tag::type;
  using BmInbox = typename bm_tag::type;

  std::uniform_real_distribution<double> dist(-1.0, 2.3);
  MAKE_GENERATOR(gen);
  std::optional<DataVector> nullopt = std::nullopt;

  const TimeStepId time_step_id_a{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};
  const TimeStepId time_step_id_b{true, 4, Time{Slab{3.4, 5.4}, {13, 100}}};
  const TimeStepId time_step_id_c{true, 5, Time{Slab{5.4, 6.4}, {17, 100}}};
  const std::pair nhbr_key{Direction<Dim>::lower_xi(), ElementId<Dim>{1}};

  BcInbox bc_inbox{};
  BmInbox bm_inbox{};

  // Send ghost cells first
  BcType send_data_a{};
  const Mesh<Dim> volume_mesh_a{5, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim - 1> mesh_a{5, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
  get<0>(send_data_a) = volume_mesh_a;
  get<1>(send_data_a) = mesh_a;
  get<2>(send_data_a) =
      DataVector{mesh_a.number_of_grid_points() * number_of_components, 0.0};
  get<4>(send_data_a) = time_step_id_a;
  get<5>(send_data_a) = 5;
  fill_with_random_values(make_not_null(&*get<2>(send_data_a)),
                          make_not_null(&gen), make_not_null(&dist));

  BoundaryMessage<Dim>* boundary_message_a = create_boundary_message(
      time_step_id_a, time_step_id_a, nhbr_key, volume_mesh_a, mesh_a,
      get<2>(send_data_a), nullopt, get<5>(send_data_a));
  BoundaryMessage<Dim>* boundary_message_a_compare = boundary_message_a;

  bc_tag::insert_into_inbox(make_not_null(&bc_inbox), time_step_id_a,
                            std::make_pair(nhbr_key, send_data_a));
  bm_tag::insert_into_inbox(make_not_null(&bm_inbox), boundary_message_a);

  CHECK((bc_inbox.at(time_step_id_a).at(nhbr_key) == send_data_a));
  // Check the values, not the pointers
  CHECK(*(bm_inbox.at(time_step_id_a).at(nhbr_key).get()) ==
        *boundary_message_a_compare);

  BcType send_data_b{};
  const Mesh<Dim> volume_mesh_b{7, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto};
  const Mesh<Dim - 1> mesh_b{7, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
  get<0>(send_data_b) = volume_mesh_b;
  get<1>(send_data_b) = mesh_b;
  get<2>(send_data_b) = DataVector{
      get<1>(send_data_b).number_of_grid_points() * number_of_components, 0.0};
  get<4>(send_data_b) = time_step_id_b;
  get<5>(send_data_b) = 6;
  fill_with_random_values(make_not_null(&*get<2>(send_data_b)),
                          make_not_null(&gen), make_not_null(&dist));

  BoundaryMessage<Dim>* boundary_message_b = create_boundary_message(
      time_step_id_b, time_step_id_b, nhbr_key, volume_mesh_b, mesh_b,
      get<2>(send_data_b), nullopt, get<5>(send_data_b));
  BoundaryMessage<Dim>* boundary_message_b_compare = boundary_message_b;

  bc_tag::insert_into_inbox(make_not_null(&bc_inbox), time_step_id_b,
                            std::make_pair(nhbr_key, send_data_b));
  bm_tag::insert_into_inbox(make_not_null(&bm_inbox), boundary_message_b);

  CHECK((bc_inbox.at(time_step_id_a).at(nhbr_key) == send_data_a));
  CHECK((bc_inbox.at(time_step_id_b).at(nhbr_key) == send_data_b));
  CHECK(*(bm_inbox.at(time_step_id_a).at(nhbr_key).get()) ==
        *boundary_message_a_compare);
  CHECK(*(bm_inbox.at(time_step_id_b).at(nhbr_key).get()) ==
        *boundary_message_b_compare);

  bc_inbox.erase(time_step_id_a);
  bm_inbox.erase(time_step_id_a);
  CHECK(bc_inbox.count(time_step_id_a) == 0);
  CHECK(bm_inbox.count(time_step_id_a) == 0);

  CHECK((bc_inbox.at(time_step_id_b).at(nhbr_key) == send_data_b));
  CHECK(*(bm_inbox.at(time_step_id_b).at(nhbr_key).get()) ==
        *boundary_message_b_compare);
  bc_inbox.erase(time_step_id_b);
  bm_inbox.erase(time_step_id_b);
  CHECK(bc_inbox.count(time_step_id_b) == 0);
  CHECK(bm_inbox.count(time_step_id_b) == 0);

  // Now send fluxes separately.
  bc_tag::insert_into_inbox(make_not_null(&bc_inbox), time_step_id_a,
                            std::make_pair(nhbr_key, send_data_a));

  boundary_message_a = create_boundary_message(
      time_step_id_a, time_step_id_a, nhbr_key, volume_mesh_a, mesh_a,
      get<2>(send_data_a), nullopt, get<5>(send_data_a));
  bm_tag::insert_into_inbox(make_not_null(&bm_inbox), boundary_message_a);

  BcType send_flux_data_a;
  get<0>(send_flux_data_a) = get<0>(send_data_a);
  get<1>(send_flux_data_a) = get<1>(send_data_a);
  get<3>(send_flux_data_a) = DataVector{
      get<1>(send_data_a).number_of_grid_points() * number_of_components, 0.0};
  // Verify that when we update the fluxes the validity of the fluxes is also
  // updated correctly
  get<4>(send_flux_data_a) = time_step_id_c;
  get<5>(send_flux_data_a) = 6;
  fill_with_random_values(make_not_null(&*get<3>(send_flux_data_a)),
                          make_not_null(&gen), make_not_null(&dist));

  BoundaryMessage<Dim>* flux_boundary_message_a = create_boundary_message(
      time_step_id_a, time_step_id_c, nhbr_key, volume_mesh_a, mesh_a, nullopt,
      get<3>(send_flux_data_a), get<5>(send_flux_data_a));

  bc_tag::insert_into_inbox(make_not_null(&bc_inbox), time_step_id_a,
                            std::make_pair(nhbr_key, send_flux_data_a));
  bm_tag::insert_into_inbox(make_not_null(&bm_inbox), flux_boundary_message_a);

  BcType send_all_data_a = send_data_a;
  get<3>(send_all_data_a) = get<3>(send_flux_data_a);
  get<4>(send_all_data_a) = get<4>(send_flux_data_a);
  get<5>(send_all_data_a) = get<5>(send_flux_data_a);

  BoundaryMessage<Dim>* all_boundary_message_a =
      create_boundary_message(time_step_id_a, time_step_id_c, nhbr_key,
                              volume_mesh_a, mesh_a, get<2>(send_all_data_a),
                              get<3>(send_all_data_a), get<5>(send_all_data_a));
  BoundaryMessage<Dim>* all_boundary_message_a_compare = all_boundary_message_a;

  CHECK(bc_inbox.at(time_step_id_a).at(nhbr_key) == send_all_data_a);
  CHECK(*(bm_inbox.at(time_step_id_a).at(nhbr_key).get()) ==
        *all_boundary_message_a_compare);

  // Check sending both ghost and flux data at once
  BcType send_all_data_b = send_data_b;
  get<3>(send_all_data_b) =
      DataVector{2 * get<1>(send_all_data_b).number_of_grid_points() *
                     number_of_components,
                 0.0};
  get<4>(send_all_data_b) = time_step_id_c;
  get<5>(send_data_a) = 6;
  fill_with_random_values(make_not_null(&*get<3>(send_all_data_b)),
                          make_not_null(&gen), make_not_null(&dist));

  BoundaryMessage<Dim>* all_boundary_message_b =
      create_boundary_message(time_step_id_b, time_step_id_c, nhbr_key,
                              volume_mesh_b, mesh_b, get<2>(send_all_data_b),
                              get<3>(send_all_data_b), get<5>(send_all_data_b));
  BoundaryMessage<Dim>* all_boundary_message_b_compare = all_boundary_message_b;

  bc_tag::insert_into_inbox(make_not_null(&bc_inbox), time_step_id_b,
                            std::make_pair(nhbr_key, send_all_data_b));
  bm_tag::insert_into_inbox(make_not_null(&bm_inbox), all_boundary_message_b);

  CHECK((bc_inbox.at(time_step_id_b).at(nhbr_key) == send_all_data_b));
  CHECK(*(bm_inbox.at(time_step_id_b).at(nhbr_key).get()) ==
        *all_boundary_message_b_compare);
}

template <size_t Dim>
void test() {
  test_no_ghost_cells<Dim>();
  test_with_ghost_cells<Dim>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.InboxTags", "[Unit][Evolution]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace evolution::dg
