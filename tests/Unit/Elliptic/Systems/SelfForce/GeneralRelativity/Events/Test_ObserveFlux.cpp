// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Domain/Tags/SurfaceJacobian.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Events/ObserveFlux.hpp"
#include "Elliptic/Systems/SelfForce/GeneralRelativity/Tags.hpp"
#include "Elliptic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct MockContributeReductionData {
  using ReductionData = GrSelfForce::Events::ObserveFlux<>::ReductionData;

  struct Results {
    observers::ObservationId observation_id{};
    std::string subfile_name{};
    std::vector<std::string> legend{};
    double iteration_id{};
    size_t num_grid_points{};
    double energy_flux{};
    double integral_of_d_cos_theta{};
    // Reduction data combined over all contributing elements
    std::optional<ReductionData> combined_reduction_data{};
  };
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
  static Results results;

  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex, typename... Ts>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationId& observation_id,
                    Parallel::ArrayComponentId /*sender_array_id*/,
                    const std::string& subfile_name,
                    const std::vector<std::string>& legend,
                    Parallel::ReductionData<Ts...>&& local_reduction_data) {
    if (not results.combined_reduction_data.has_value()) {
      results.combined_reduction_data.emplace(std::move(local_reduction_data));
    } else {
      results.combined_reduction_data->combine(std::move(local_reduction_data));
    }
    auto reduction_data = *results.combined_reduction_data;
    reduction_data.finalize();
    results.observation_id = observation_id;
    results.subfile_name = subfile_name;
    results.legend = legend;
    results.iteration_id = std::get<0>(reduction_data.data());
    results.num_grid_points = std::get<1>(reduction_data.data());
    results.energy_flux = std::get<2>(reduction_data.data());
    results.integral_of_d_cos_theta = std::get<3>(reduction_data.data());
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
MockContributeReductionData::Results MockContributeReductionData::results{};

template <typename Metavariables>
struct ElementComponent {
  using component_being_mocked = void;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<2>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

template <typename Metavariables>
struct MockObserverComponent {
  using component_being_mocked = observers::Observer<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::ContributeReductionData>;
  using with_these_simple_actions = tmpl::list<MockContributeReductionData>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockObserverComponent<Metavariables>>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<Event, tmpl::list<GrSelfForce::Events::ObserveFlux<>>>>;
  };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.GrSelfForce.Events.ObserveFlux", "[Unit][Elliptic]") {
  // Schwarzschild, r0 = 10, m = 2, coordinates (r, cos(theta)).
  const double orbital_radius = 10.0;
  const int m_mode = 2;
  const GrSelfForce::AnalyticData::CircularOrbit circular_orbit{
      1.0, 0.0, orbital_radius, m_mode, std::array{5.0, 5.0, 15.0, 15.0},
      true};

  // Two elements in r: [9, 10] and [10, 11]. The outer boundary is r = 11.
  const domain::creators::Rectilinear<2> domain_creator{
      {{9.0, -1.0}}, {{11.0, 1.0}}, {{1, 0}}, {{4, 4}}};
  const auto domain = domain_creator.create_domain();
  const Mesh<2> mesh{4_st, Spectral::Basis::Legendre,
                     Spectral::Quadrature::Gauss};
  const ElementId<2> element_id_a{0, {{{1, 0}, {0, 0}}}};  // left, inner
  const ElementId<2> element_id_b{0, {{{1, 1}, {0, 0}}}};  // right, outer xi
  const std::vector<std::array<size_t, 2>> refinement_levels{{1, 0}};

  using mock_observer_writer = MockObserverComponent<Metavariables>;
  using element_component = ElementComponent<Metavariables>;
  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_group_component<mock_observer_writer>(&runner);
  ActionTesting::emplace_component<element_component>(&runner, element_id_a);
  ActionTesting::emplace_component<element_component>(&runner, element_id_b);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // Complex test field, constant everywhere: psi^k = (1 + i) + (2 + i) k for
  // the 10 components tt, tr, ..., phiphi (k = 0, ..., 9).
  const auto psi_value = [](const size_t k) {
    return std::complex<double>{1.0, 1.0} +
           std::complex<double>{2.0, 1.0} * static_cast<double>(k);
  };
  tnsr::aa<ComplexDataVector, 3> field{mesh.number_of_grid_points()};
  size_t k = 0;
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = a; b < 4; ++b) {
      field.get(a, b) =
          ComplexDataVector{mesh.number_of_grid_points(), psi_value(k)};
      ++k;
    }
  }
  const std::complex<double> psi7 = psi_value(7);  // theta theta
  const std::complex<double> psi8 = psi_value(8);  // theta phi
  const std::complex<double> psi9 = psi_value(9);  // phi phi

  // Expected flux: (m Omega)^2 / 32 * (|psi7 - psi9|^2 + 4 |psi8|^2) * 2
  //              = (0.004 / 32) * (20 + 1480) * 2 = 0.375
  const double omega = 1.0 / std::pow(orbital_radius, 1.5);
  const double expected_energy_flux =
      square(m_mode * omega) / 32.0 *
      (std::norm(psi7 - psi9) + 4.0 * std::norm(psi8)) * 2.0;
  CHECK(expected_energy_flux == approx(0.375));

  const GrSelfForce::Events::ObserveFlux event{};

  using face_jac_tag = domain::Tags::Faces<
      2,
      domain::Tags::DetSurfaceJacobian<Frame::ElementLogical, Frame::Inertial>>;
  using face_coords_tag =
      domain::Tags::Faces<2, domain::Tags::Coordinates<2, Frame::Inertial>>;

  const Direction<2> upper_xi = Direction<2>::upper_xi();
  const Mesh<1> face_mesh = mesh.slice_away(0);
  const auto face_logical_eta = get<0>(logical_coordinates(face_mesh));

  for (const auto& element_id : std::array{element_id_a, element_id_b}) {
    auto element = domain::create_initial_element(element_id, domain.blocks(),
                                                  refinement_levels);
    const bool is_outer = element.external_boundaries().contains(upper_xi);

    // Outer face of element b: r = 11, cos(theta) = eta, Jacobian = 1.
    // Element a has no outer face and must not contribute.
    typename face_jac_tag::type face_jac_map{};
    typename face_coords_tag::type face_coords_map{};
    if (is_outer) {
      tnsr::I<DataVector, 2, Frame::Inertial> face_coords{
          face_mesh.number_of_grid_points()};
      get<0>(face_coords) = 11.0;
      get<1>(face_coords) = face_logical_eta;
      face_coords_map[upper_xi] = std::move(face_coords);
      face_jac_map[upper_xi] =
          Scalar<DataVector>{face_mesh.number_of_grid_points(), 1.0};
    }

    auto box = db::create<tmpl::list<
        elliptic::Tags::Background<elliptic::analytic_data::Background>,
        domain::Tags::Element<2>, domain::Tags::Mesh<2>, face_jac_tag,
        face_coords_tag, GrSelfForce::Tags::MMode>>(
        std::unique_ptr<elliptic::analytic_data::Background>(
            std::make_unique<GrSelfForce::AnalyticData::CircularOrbit>(
                circular_orbit)),
        std::move(element), mesh, std::move(face_jac_map),
        std::move(face_coords_map), field);
    auto obs_box =
        make_observation_box<db::AddComputeTags<>>(make_not_null(&box));
    event.run(make_not_null(&obs_box),
              ActionTesting::cache<element_component>(runner, element_id),
              element_id, std::add_pointer_t<element_component>{},
              {.name = "Iteration", .value = 1.0});
  }

  runner.template invoke_queued_simple_action<mock_observer_writer>(0);
  runner.template invoke_queued_simple_action<mock_observer_writer>(0);
  CHECK(runner.template is_simple_action_queue_empty<mock_observer_writer>(0));

  const auto& results = MockContributeReductionData::results;
  CHECK(results.observation_id.value() == 1.0);
  CHECK(results.observation_id.observation_key() ==
        observers::ObservationKey("/Flux.dat"));
  CHECK(results.subfile_name == "/Flux");
  CHECK(results.legend ==
        std::vector<std::string>{"ObservationValue", "NumberOfPoints",
                                 "EnergyFlux", "IntegralOfDCosTheta"});
  CHECK(results.iteration_id == 1.0);
  CHECK(results.num_grid_points == 2 * mesh.number_of_grid_points());
  CHECK(results.energy_flux == approx(expected_energy_flux));
  // int_{-1}^{1} d(cos(theta)) = 2
  CHECK(results.integral_of_d_cos_theta == approx(2.0));
}
