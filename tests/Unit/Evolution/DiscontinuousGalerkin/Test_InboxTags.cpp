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
#include "Time/TimeStepId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg {
namespace {
template <size_t Dim>
void test_no_ghost_cells() {
  static constexpr size_t number_of_components = 1 + Dim;
  using bc_tag = Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>;
  using Type = std::tuple<Mesh<Dim - 1>, std::optional<std::vector<double>>,
                          std::optional<std::vector<double>>, ::TimeStepId>;
  using Inbox = typename bc_tag::type;

  std::uniform_real_distribution<double> dist(-1.0, 2.3);
  MAKE_GENERATOR(gen);

  const TimeStepId time_step_id_a{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};
  const TimeStepId time_step_id_b{true, 4, Time{Slab{3.4, 5.4}, {13, 100}}};
  const TimeStepId time_step_id_c{true, 5, Time{Slab{5.4, 6.4}, {17, 100}}};
  const std::pair nhbr_key{Direction<Dim>::lower_xi(), ElementId<Dim>{1}};

  Inbox inbox{};

  Type send_data_a{};
  const Mesh<Dim - 1> mesh_a{5, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
  get<0>(send_data_a) = mesh_a;
  get<2>(send_data_a) = std::vector<double>(mesh_a.number_of_grid_points() *
                                            number_of_components);
  get<3>(send_data_a) = time_step_id_a;
  fill_with_random_values(make_not_null(&*get<2>(send_data_a)),
                          make_not_null(&gen), make_not_null(&dist));

  bc_tag::insert_into_inbox(make_not_null(&inbox), time_step_id_a,
                            std::make_pair(nhbr_key, send_data_a));

  CHECK((inbox.at(time_step_id_a).at(nhbr_key) == send_data_a));

  Type send_data_b{};
  const Mesh<Dim - 1> mesh_b{7, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
  get<0>(send_data_b) = mesh_b;

  get<2>(send_data_b) = std::vector<double>(mesh_b.number_of_grid_points() *
                                            number_of_components);
  // Set the future time step to make sure the implementation doesn't mix the
  // receive time ID and the validity range time ID
  get<3>(send_data_b) = time_step_id_c;
  fill_with_random_values(make_not_null(&*get<2>(send_data_b)),
                          make_not_null(&gen), make_not_null(&dist));

  bc_tag::insert_into_inbox(make_not_null(&inbox), time_step_id_b,
                            std::make_pair(nhbr_key, send_data_b));

  CHECK((inbox.at(time_step_id_a).at(nhbr_key) == send_data_a));
  CHECK((inbox.at(time_step_id_b).at(nhbr_key) == send_data_b));

  inbox.erase(time_step_id_a);
  CHECK(inbox.count(time_step_id_a) == 0);

  CHECK((inbox.at(time_step_id_b).at(nhbr_key) == send_data_b));
  inbox.erase(time_step_id_b);
  CHECK(inbox.count(time_step_id_b) == 0);
}

template <size_t Dim>
void test_with_ghost_cells() {
  static constexpr size_t number_of_components = 1 + Dim;
  using bc_tag = Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>;
  using Type = std::tuple<Mesh<Dim - 1>, std::optional<std::vector<double>>,
                          std::optional<std::vector<double>>, ::TimeStepId>;
  using Inbox = typename bc_tag::type;

  std::uniform_real_distribution<double> dist(-1.0, 2.3);
  MAKE_GENERATOR(gen);

  const TimeStepId time_step_id_a{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};
  const TimeStepId time_step_id_b{true, 4, Time{Slab{3.4, 5.4}, {13, 100}}};
  const TimeStepId time_step_id_c{true, 5, Time{Slab{5.4, 6.4}, {17, 100}}};
  const std::pair nhbr_key{Direction<Dim>::lower_xi(), ElementId<Dim>{1}};

  Inbox inbox{};

  // Send ghost cells first
  Type send_data_a{};
  const Mesh<Dim - 1> mesh_a{5, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
  get<0>(send_data_a) = mesh_a;
  get<1>(send_data_a) = std::vector<double>(mesh_a.number_of_grid_points() *
                                            number_of_components);
  get<3>(send_data_a) = time_step_id_a;
  fill_with_random_values(make_not_null(&*get<1>(send_data_a)),
                          make_not_null(&gen), make_not_null(&dist));

  bc_tag::insert_into_inbox(make_not_null(&inbox), time_step_id_a,
                            std::make_pair(nhbr_key, send_data_a));

  CHECK((inbox.at(time_step_id_a).at(nhbr_key) == send_data_a));

  Type send_data_b{};
  const Mesh<Dim - 1> mesh_b{7, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
  get<0>(send_data_b) = mesh_b;
  get<1>(send_data_b) = std::vector<double>(
      get<0>(send_data_b).number_of_grid_points() * number_of_components);
  get<3>(send_data_b) = time_step_id_b;
  fill_with_random_values(make_not_null(&*get<1>(send_data_b)),
                          make_not_null(&gen), make_not_null(&dist));

  bc_tag::insert_into_inbox(make_not_null(&inbox), time_step_id_b,
                            std::make_pair(nhbr_key, send_data_b));

  CHECK((inbox.at(time_step_id_a).at(nhbr_key) == send_data_a));
  CHECK((inbox.at(time_step_id_b).at(nhbr_key) == send_data_b));

  inbox.erase(time_step_id_a);
  CHECK(inbox.count(time_step_id_a) == 0);

  CHECK((inbox.at(time_step_id_b).at(nhbr_key) == send_data_b));
  inbox.erase(time_step_id_b);
  CHECK(inbox.count(time_step_id_b) == 0);

  // Now send fluxes separately.
  bc_tag::insert_into_inbox(make_not_null(&inbox), time_step_id_a,
                            std::make_pair(nhbr_key, send_data_a));
  Type send_flux_data_a;
  get<0>(send_flux_data_a) = get<0>(send_data_a);
  get<2>(send_flux_data_a) = std::vector<double>(
      get<0>(send_data_a).number_of_grid_points() * number_of_components);
  // Verify that when we update the fluxes the validity of the fluxes is also
  // updated correctly
  get<3>(send_flux_data_a) = time_step_id_c;
  fill_with_random_values(make_not_null(&*get<2>(send_flux_data_a)),
                          make_not_null(&gen), make_not_null(&dist));
  bc_tag::insert_into_inbox(make_not_null(&inbox), time_step_id_a,
                            std::make_pair(nhbr_key, send_flux_data_a));

  Type send_all_data_a = send_data_a;
  get<2>(send_all_data_a) = get<2>(send_flux_data_a);
  get<3>(send_all_data_a) = get<3>(send_flux_data_a);

  CHECK((inbox.at(time_step_id_a).at(nhbr_key) == send_all_data_a));

  // Check sending both ghost and flux data at once
  Type send_all_data_b = send_data_b;
  get<2>(send_all_data_b) =
      std::vector<double>(2 * get<0>(send_all_data_b).number_of_grid_points() *
                          number_of_components);
  get<3>(send_all_data_b) = time_step_id_c;
  fill_with_random_values(make_not_null(&*get<2>(send_all_data_b)),
                          make_not_null(&gen), make_not_null(&dist));
  bc_tag::insert_into_inbox(make_not_null(&inbox), time_step_id_b,
                            std::make_pair(nhbr_key, send_all_data_b));

  CHECK((inbox.at(time_step_id_b).at(nhbr_key) == send_all_data_b));
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
