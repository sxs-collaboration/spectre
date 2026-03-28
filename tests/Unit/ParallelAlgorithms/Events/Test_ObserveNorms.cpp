// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/MetavariablesTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/Observer/Actions/RegisterEvents.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Events/ObserveNorms.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var0 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var1 : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Inertial>;
};

struct Var0TimesTwo : db::SimpleTag {
  using type = std::optional<Scalar<DataVector>>;
};

struct Var0TimesTwoCompute : db::ComputeTag, Var0TimesTwo {
  using base = Var0TimesTwo;
  using return_type = std::optional<Scalar<DataVector>>;
  using argument_tags = tmpl::list<Var0>;
  static void function(
      const gsl::not_null<std::optional<Scalar<DataVector>>*> result,
      const Scalar<DataVector>& scalar_var) {
    *result = Scalar<DataVector>{DataVector{2.0 * get(scalar_var)}};
  }
};

struct Var0TimesThree : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var0TimesThreeCompute : db::ComputeTag,
                               ::Tags::Variables<tmpl::list<Var0TimesThree>> {
  using base = ::Tags::Variables<tmpl::list<Var0TimesThree>>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<Var0>;
  static void function(
      const gsl::not_null<::Variables<tmpl::list<Var0TimesThree>>*> result,
      const Scalar<DataVector>& scalar_var) {
    result->initialize(get(scalar_var).size());
    get(get<Var0TimesThree>(*result)) = 3.0 * get(scalar_var);
  }
};

struct TestSectionIdTag {};

// Name for option parsing
struct ObserveMyNorms {};

struct MockContributeReductionData {
  struct Results {
    observers::ObservationId observation_id;
    std::string subfile_name;
    std::vector<std::string> reduction_names;
    double time;
    size_t number_of_grid_points;
    double volume;
    std::vector<double> max_values;
    std::vector<double> min_values;
    std::vector<double> l1_norm_values;
    std::vector<double> l1_integral_norm_values;
    std::vector<double> l2_norm_values;
    std::vector<double> l2_integral_norm_values;
    std::vector<double> volume_integral_values;
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
                    const std::vector<std::string>& reduction_names,
                    Parallel::ReductionData<Ts...>&& reduction_data) {
    reduction_data.finalize();
    results.observation_id = observation_id;
    results.subfile_name = subfile_name;
    results.reduction_names = reduction_names;
    results.time = std::get<0>(reduction_data.data());
    results.number_of_grid_points = std::get<1>(reduction_data.data());
    results.volume = std::get<2>(reduction_data.data());
    results.max_values = std::get<3>(reduction_data.data());
    results.min_values = std::get<4>(reduction_data.data());
    results.l1_norm_values = std::get<5>(reduction_data.data());
    results.l1_integral_norm_values = std::get<6>(reduction_data.data());
    results.l2_norm_values = std::get<7>(reduction_data.data());
    results.l2_integral_norm_values = std::get<8>(reduction_data.data());
    results.volume_integral_values = std::get<9>(reduction_data.data());
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
MockContributeReductionData::Results MockContributeReductionData::results{};

template <typename Metavariables>
struct ElementComponent {
  using component_being_mocked = void;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;
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

template <typename ArraySectionIdTag, typename OptionName = void>
using ObserveNormsEvent = Events::ObserveNorms<
    tmpl::list<Var0, Var1, Var0TimesTwoCompute, Var0TimesThree>,
    tmpl::list<Var0TimesThreeCompute>, ArraySectionIdTag, OptionName>;

template <size_t Dim, typename ArraySectionIdTag, typename OptionName = void>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockObserverComponent<Metavariables>>;

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        Event, tmpl::list<ObserveNormsEvent<ArraySectionIdTag, OptionName>>>>;
  };
};

template <typename ArraySectionIdTag, typename ObserveEvent>
void test(const std::unique_ptr<ObserveEvent> observe,
          const Spectral::Basis basis, const Spectral::Quadrature quadrature,
          const std::optional<std::string>& section) {
  CAPTURE(pretty_type::name<ArraySectionIdTag>());
  CAPTURE(section);
  using metavariables = Metavariables<3, ArraySectionIdTag>;
  using element_component = ElementComponent<metavariables>;
  using observer_component = MockObserverComponent<metavariables>;
  const typename element_component::array_index array_index(0);
  const Mesh<3> mesh{3, basis, quadrature};
  const size_t num_points = mesh.number_of_grid_points();
  // Jacobian of a cube with side length 1, so expected volume is 1.
  const Scalar<DataVector> det_inv_jacobian(num_points, cube(2.));
  const double expected_volume = 1.;
  const double observation_time = 2.0;
  Variables<tmpl::list<Var0, Var1>> vars(num_points);
  // Fill the variables with some data.  It doesn't matter much what,
  // but integers are nice in that we don't have to worry about
  // roundoff error.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::iota(vars.data(), vars.data() + vars.size(), 1.0);

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
  ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                      array_index);
  ActionTesting::emplace_group_component<observer_component>(&runner);

  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<metavariables>,
                        ::Events::Tags::ObserverMesh<3>,
                        ::Events::Tags::ObserverDetInvJacobian<
                            Frame::ElementLogical, Frame::Inertial>,
                        Tags::Variables<typename decltype(vars)::tags_list>,
                        observers::Tags::ObservationKey<ArraySectionIdTag>>>(
      metavariables{}, mesh, det_inv_jacobian, vars, section);

  const auto ids_to_register =
      observers::get_registration_observation_type_and_key(*observe, box);
  const std::string expected_subfile_name{
      "/reduction0" +
      (std::is_same_v<ArraySectionIdTag, void> ? ""
                                               : section.value_or("Unused"))};
  const observers::ObservationKey expected_observation_key_for_reg(
      expected_subfile_name + ".dat");
  if (std::is_same_v<ArraySectionIdTag, void> or section.has_value()) {
    CHECK(ids_to_register->first == observers::TypeOfObservation::Reduction);
    CHECK(ids_to_register->second == expected_observation_key_for_reg);
  } else {
    CHECK_FALSE(ids_to_register.has_value());
  }

  CHECK(static_cast<const Event&>(*observe).is_ready(
      box, ActionTesting::cache<element_component>(runner, array_index),
      array_index, std::add_pointer_t<element_component>{}));

  auto obs_box = make_observation_box<
      tmpl::filter<typename ObserveNormsEvent<
                       ArraySectionIdTag>::compute_tags_for_observation_box,
                   db::is_compute_tag<tmpl::_1>>>(make_not_null(&box));
  observe->run(make_not_null(&obs_box),
               ActionTesting::cache<element_component>(runner, array_index),
               array_index, std::add_pointer_t<element_component>{},
               {"TimeName", observation_time});

  // Process the data
  runner.template invoke_queued_simple_action<observer_component>(0);
  CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));

  const auto& results = MockContributeReductionData::results;
  CHECK(results.observation_id.value() == observation_time);
  CHECK(results.observation_id.observation_key() ==
        expected_observation_key_for_reg);
  CHECK(results.subfile_name == expected_subfile_name);
  CHECK(results.reduction_names[0] == "TimeName");
  CHECK(results.time == observation_time);
  CHECK(results.reduction_names[1] == "NumberOfPoints");
  CHECK(results.number_of_grid_points == num_points);
  CHECK(results.reduction_names[2] == "Volume");
  CHECK(results.volume == approx(expected_volume));

  // Check max values
  CHECK(results.reduction_names[3] == "Max(Var0)");
  CHECK(results.reduction_names[4] == "Max(Var0)");
  CHECK(results.reduction_names[5] == "Max(Var0TimesTwo)");
  CHECK(results.reduction_names[6] == "Max(Var0TimesThree)");
  CHECK(results.max_values == std::vector<double>{27.0, 27.0, 54.0, 81.0});

  // Check min values
  CHECK(results.reduction_names[7] == "Min(Var1_x)");
  CHECK(results.reduction_names[8] == "Min(Var1_y)");
  CHECK(results.reduction_names[9] == "Min(Var1_z)");
  CHECK(results.reduction_names[10] == "Min(Var1)");
  CHECK(results.min_values == std::vector<double>{28.0, 55.0, 82.0, 28.0});

  // Check L1 norms
  CHECK(results.reduction_names[11] == "L1Norm(Var1)");
  CHECK(results.reduction_names[12] == "L1Norm(Var1_x)");
  CHECK(results.reduction_names[13] == "L1Norm(Var1_y)");
  CHECK(results.reduction_names[14] == "L1Norm(Var1_z)");
  CHECK(results.l1_norm_values[0] == approx(204.0));
  CHECK(results.l1_norm_values[1] == approx(41.0));
  CHECK(results.l1_norm_values[2] == approx(68.0));
  CHECK(results.l1_norm_values[3] == approx(95.0));

  // Check L1 integral norms
  CHECK(results.reduction_names[15] == "L1IntegralNorm(Var1)");
  CHECK(results.reduction_names[16] == "L1IntegralNorm(Var1_x)");
  CHECK(results.reduction_names[17] == "L1IntegralNorm(Var1_y)");
  CHECK(results.reduction_names[18] == "L1IntegralNorm(Var1_z)");
  // All Var1 values are positive, so L1IntegralNorm equals the volume integral
  // divided by the volume (which is 1), giving the same values as L1Norm.
  for (size_t i = 0; i < 4; i++) {
    CHECK(results.l1_integral_norm_values[i] ==
          approx(results.l1_norm_values[i]));
  }

  // Check L2 norms
  CHECK(results.reduction_names[19] == "L2Norm(Var1)");
  CHECK(results.reduction_names[20] == "L2Norm(Var1_x)");
  CHECK(results.reduction_names[21] == "L2Norm(Var1_y)");
  CHECK(results.reduction_names[22] == "L2Norm(Var1_z)");
  CHECK(results.l2_norm_values[0] == approx(124.5471798155221137));
  CHECK(results.l2_norm_values[1] == approx(41.73328008516305232));
  CHECK(results.l2_norm_values[2] == approx(68.44462481938714404));
  CHECK(results.l2_norm_values[3] == approx(95.3187634554008838));

  // Check L2 integral norms
  CHECK(results.reduction_names[23] == "L2IntegralNorm(Var1)");
  CHECK(results.reduction_names[24] == "L2IntegralNorm(Var1_x)");
  CHECK(results.reduction_names[25] == "L2IntegralNorm(Var1_y)");
  CHECK(results.reduction_names[26] == "L2IntegralNorm(Var1_z)");
  if (basis != Spectral::Basis::FiniteDifference) {
    CHECK(results.l2_integral_norm_values[0] == approx(124.18131904598212145));
    CHECK(results.l2_integral_norm_values[1] == approx(41.36826480931165406));
    CHECK(results.l2_integral_norm_values[2] == approx(68.22267462752640199));
    CHECK(results.l2_integral_norm_values[3] == approx(95.15951520123110186));
  } else {
    for (size_t i = 0; i < 4; i++) {
      CHECK(results.l2_integral_norm_values[i] ==
            approx(results.l2_norm_values[i]));
    }
  }

  // Check volume integral norms
  CHECK(results.reduction_names[27] == "VolumeIntegral(Var1)");
  CHECK(results.reduction_names[28] == "VolumeIntegral(Var1_x)");
  CHECK(results.reduction_names[29] == "VolumeIntegral(Var1_y)");
  CHECK(results.reduction_names[30] == "VolumeIntegral(Var1_z)");
  CHECK(results.volume_integral_values[0] == approx(204.0));
  CHECK(results.volume_integral_values[1] == approx(41.0));
  CHECK(results.volume_integral_values[2] == approx(68.0));
  CHECK(results.volume_integral_values[3] == approx(95.0));
}

template <bool Spherical, typename ArraySectionIdTag, typename ObserveEvent>
void test_cartoon(const std::unique_ptr<ObserveEvent> observe,
                  const std::optional<std::string>& section,
                  const double x_inner = 0.0) {
  // We are testing that the correct cartesian to spherical/cylindrical jacobian
  // is being mulitplied
  CAPTURE(pretty_type::name<ArraySectionIdTag>());
  CAPTURE(section);
  CAPTURE(Spherical);
  using metavariables = Metavariables<3, ArraySectionIdTag>;
  using element_component = ElementComponent<metavariables>;
  using observer_component = MockObserverComponent<metavariables>;
  const typename element_component::array_index array_index(0);
  Mesh<3> mesh;
  // DataBox needs inertial coordinates to do cartoon-basis integration
  using Affine = domain::CoordinateMaps::Affine;
  using Identity1D = domain::CoordinateMaps::Identity<1>;
  double expected_volume{};
  tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords;
  Scalar<DataVector> det_inv_jacobian;
  if constexpr (Spherical) {
    mesh = Mesh<3>{{{5, 1, 1}},
                   {{Spectral::Basis::Legendre, Spectral::Basis::Cartoon,
                     Spectral::Basis::Cartoon}},
                   {{Spectral::Quadrature::GaussLobatto,
                     Spectral::Quadrature::SphericalSymmetry,
                     Spectral::Quadrature::SphericalSymmetry}}};
    const domain::CoordinateMap<
        Frame::ElementLogical, Frame::Inertial,
        domain::CoordinateMaps::ProductOf3Maps<Affine, Identity1D, Identity1D>>
        map{{Affine{-1.0, 1.0, x_inner, 2.0}, Identity1D{}, Identity1D{}}};
    inertial_coords = map(logical_coordinates(mesh));
    det_inv_jacobian =
        determinant(map.inv_jacobian(logical_coordinates((mesh))));
    expected_volume = 4.0 * M_PI * (cube(2) - cube(x_inner)) / 3.0;
  } else {
    mesh = Mesh<3>{{{5, 4, 1}},
                   {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
                     Spectral::Basis::Cartoon}},
                   {{Spectral::Quadrature::GaussLobatto,
                     Spectral::Quadrature::GaussLobatto,
                     Spectral::Quadrature::AxialSymmetry}}};
    const domain::CoordinateMap<
        Frame::ElementLogical, Frame::Inertial,
        domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Identity1D>>
        map{{Affine{-1.0, 1.0, x_inner, 2.0}, Affine{-1.0, 1.0, 2.0, 3.0},
             Identity1D{}}};
    inertial_coords = map(logical_coordinates(mesh));
    det_inv_jacobian =
        determinant(map.inv_jacobian(logical_coordinates((mesh))));
    expected_volume = M_PI * (square(2.0) - square(x_inner));
  }
  const size_t num_points = mesh.number_of_grid_points();

  const double observation_time = 2.0;
  Variables<tmpl::list<Var0, Var1>> vars(num_points);

  auto& scalar = get<Var0>(vars);
  get<>(scalar) = get<0>(inertial_coords);

  get<Var1>(vars) = inertial_coords;

  ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
  ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                      array_index);
  ActionTesting::emplace_group_component<observer_component>(&runner);

  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<metavariables>,
                        ::Events::Tags::ObserverMesh<3>,
                        ::Events::Tags::ObserverDetInvJacobian<
                            Frame::ElementLogical, Frame::Inertial>,
                        domain::Tags::Coordinates<3, Frame::Inertial>,
                        Tags::Variables<typename decltype(vars)::tags_list>,
                        observers::Tags::ObservationKey<ArraySectionIdTag>>>(
      metavariables{}, mesh, det_inv_jacobian, inertial_coords, vars, section);

  auto obs_box = make_observation_box<
      tmpl::filter<typename ObserveNormsEvent<
                       ArraySectionIdTag>::compute_tags_for_observation_box,
                   db::is_compute_tag<tmpl::_1>>>(make_not_null(&box));
  observe->run(make_not_null(&obs_box),
               ActionTesting::cache<element_component>(runner, array_index),
               array_index, std::add_pointer_t<element_component>{},
               {"TimeName", observation_time});

  // Process the data
  runner.template invoke_queued_simple_action<observer_component>(0);
  CHECK(runner.template is_simple_action_queue_empty<observer_component>(0));

  const auto& results = MockContributeReductionData::results;

  CHECK(results.reduction_names[0] == "TimeName");
  CHECK(results.time == observation_time);
  CHECK(results.reduction_names[1] == "NumberOfPoints");
  CHECK(results.number_of_grid_points == num_points);
  CHECK(results.reduction_names[2] == "Volume");
  CHECK(results.volume == approx(expected_volume));

  const auto integrate = [&mesh, &det_inv_jacobian](
                             const DataVector& a,
                             const DataVector& coord_jacobian) -> double {
    const DataVector integrand = a * coord_jacobian / get<>(det_inv_jacobian);
    return definite_integral(integrand, mesh);
  };
  const auto normalize_l1 = [&expected_volume](const double a) -> double {
    return a / expected_volume;
  };
  const auto normalize_l2 = [&expected_volume](const double a) -> double {
    return sqrt(a / expected_volume);
  };

  // Check L1 integral norms
  // Var1 = inertial_coords; all non-z components are >= 0 in the test domain,
  // so abs(Var1_x) = Var1_x, abs(Var1_y) = Var1_y, abs(Var1_z) = 0.
  CHECK(results.reduction_names[3] == "L1IntegralNorm(Var1)");
  CHECK(results.reduction_names[4] == "L1IntegralNorm(Var1_x)");
  CHECK(results.reduction_names[5] == "L1IntegralNorm(Var1_y)");
  CHECK(results.reduction_names[6] == "L1IntegralNorm(Var1_z)");
  if constexpr (Spherical) {
    const double l1_result = normalize_l1(
        integrate(get<0>(get<Var1>(vars)), square(get<0>(inertial_coords))));
    CHECK(results.l1_integral_norm_values[0] == approx(l1_result));
    CHECK(results.l1_integral_norm_values[1] == approx(l1_result));
    CHECK(results.l1_integral_norm_values[2] == approx(0.0));
  } else {
    CHECK(results.l1_integral_norm_values[0] ==
          approx(normalize_l1(
              integrate(get<0>(get<Var1>(vars)), get<0>(inertial_coords)) +
              integrate(get<1>(get<Var1>(vars)), get<0>(inertial_coords)))));
    CHECK(results.l1_integral_norm_values[1] ==
          approx(normalize_l1(
              integrate(get<0>(get<Var1>(vars)), get<0>(inertial_coords)))));
    CHECK(results.l1_integral_norm_values[2] ==
          approx(normalize_l1(
              integrate(get<1>(get<Var1>(vars)), get<0>(inertial_coords)))));
  }
  CHECK(results.l1_integral_norm_values[3] == approx(0.0));

  // Check L2 integral norms
  CHECK(results.reduction_names[7] == "L2IntegralNorm(Var1)");
  CHECK(results.reduction_names[8] == "L2IntegralNorm(Var1_x)");
  CHECK(results.reduction_names[9] == "L2IntegralNorm(Var1_y)");
  CHECK(results.reduction_names[10] == "L2IntegralNorm(Var1_z)");
  if constexpr (Spherical) {
    const double result = normalize_l2(integrate(
        square(get<0>(get<Var1>(vars))), square(get<0>(inertial_coords))));
    CHECK(results.l2_integral_norm_values[0] == approx(result));
    CHECK(results.l2_integral_norm_values[1] == approx(result));
    CHECK(results.l2_integral_norm_values[2] == approx(0.0));
  } else {
    CHECK(results.l2_integral_norm_values[0] ==
          approx(normalize_l2(integrate(square(get<0>(get<Var1>(vars))),
                                        get<0>(inertial_coords)) +
                              integrate(square(get<1>(get<Var1>(vars))),
                                        get<0>(inertial_coords)))));
    CHECK(results.l2_integral_norm_values[1] ==
          approx(normalize_l2(integrate(square(get<0>(get<Var1>(vars))),
                                        get<0>(inertial_coords)))));
    CHECK(results.l2_integral_norm_values[2] ==
          approx(normalize_l2(integrate(square(get<1>(get<Var1>(vars))),
                                        get<0>(inertial_coords)))));
  }
  CHECK(results.l2_integral_norm_values[3] == approx(0.0));

  // Check volume integral norms
  CHECK(results.reduction_names[11] == "VolumeIntegral(Var1)");
  CHECK(results.reduction_names[12] == "VolumeIntegral(Var1_x)");
  CHECK(results.reduction_names[13] == "VolumeIntegral(Var1_y)");
  CHECK(results.reduction_names[14] == "VolumeIntegral(Var1_z)");
  if constexpr (Spherical) {
    const double result =
        integrate(get<0>(get<Var1>(vars)), square(get<0>(inertial_coords)));
    CHECK(results.volume_integral_values[0] == approx(result));
    CHECK(results.volume_integral_values[1] == approx(result));
    CHECK(results.volume_integral_values[2] == approx(0.0));
  } else {
    CHECK(results.volume_integral_values[0] ==
          approx(integrate(get<0>(get<Var1>(vars)), get<0>(inertial_coords)) +
                 integrate(get<1>(get<Var1>(vars)), get<0>(inertial_coords))));
    CHECK(results.volume_integral_values[1] ==
          approx(integrate(get<0>(get<Var1>(vars)), get<0>(inertial_coords))));
    CHECK(results.volume_integral_values[2] ==
          approx(integrate(get<1>(get<Var1>(vars)), get<0>(inertial_coords))));
  }
  CHECK(results.volume_integral_values[3] == approx(0.0));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ObserveNorms", "[Unit][Evolution]") {
  test<TestSectionIdTag>(std::make_unique<ObserveNormsEvent<TestSectionIdTag>>(
                             ObserveNormsEvent<TestSectionIdTag>{
                                 "reduction0",
                                 {{"Var0", "Max", "Individual"},
                                  {"Var1", "Min", "Individual"},
                                  {"Var0", "Max", "Sum"},
                                  {"Var0TimesTwo", "Max", "Individual"},
                                  {"Var0TimesThree", "Max", "Individual"},
                                  {"Var1", "L1Norm", "Sum"},
                                  {"Var1", "L1IntegralNorm", "Sum"},
                                  {"Var1", "L2Norm", "Sum"},
                                  {"Var1", "L2IntegralNorm", "Sum"},
                                  {"Var1", "VolumeIntegral", "Sum"},
                                  {"Var1", "L1Norm", "Individual"},
                                  {"Var1", "L1IntegralNorm", "Individual"},
                                  {"Var1", "L2Norm", "Individual"},
                                  {"Var1", "L2IntegralNorm", "Individual"},
                                  {"Var1", "VolumeIntegral", "Individual"},
                                  {"Var1", "Min", "Sum"}}}),
                         Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto, "Section0");

  INFO("create/serialize");
  register_factory_classes_with_charm<Metavariables<3, void>>();
  const auto factory_event = TestHelpers::test_creation<std::unique_ptr<Event>,
                                                        Metavariables<3, void>>(
      // [input_file_examples]
      R"(
  ObserveNorms:
    SubfileName: reduction0
    TensorsToObserve:
    - Name: Var0
      NormType: Max
      Components: Individual
    - Name: Var1
      NormType: Min
      Components: Individual
    - Name: Var0
      NormType: Max
      Components: Sum
    - Name: Var0TimesTwo
      NormType: Max
      Components: Individual
    - Name: Var0TimesThree
      NormType: Max
      Components: Individual
    - Name: Var1
      NormType: L1Norm
      Components: Sum
    - Name: Var1
      NormType: L1IntegralNorm
      Components: Sum
    - Name: Var1
      NormType: L2Norm
      Components: Sum
    - Name: Var1
      NormType: L2IntegralNorm
      Components: Sum
    - Name: Var1
      NormType: VolumeIntegral
      Components: Sum
    - Name: Var1
      NormType: L1Norm
      Components: Individual
    - Name: Var1
      NormType: L1IntegralNorm
      Components: Individual
    - Name: Var1
      NormType: L2Norm
      Components: Individual
    - Name: Var1
      NormType: L2IntegralNorm
      Components: Individual
    - Name: Var1
      NormType: VolumeIntegral
      Components: Individual
    - Name: Var1
      NormType: Min
      Components: Sum
        )");
  // [input_file_examples]
  auto serialized_event = serialize_and_deserialize(factory_event);
  test<void>(std::move(serialized_event), Spectral::Basis::Legendre,
             Spectral::Quadrature::GaussLobatto, std::nullopt);

  {
    INFO("Test option name");
    // Test option name
    TestHelpers::test_creation<std::unique_ptr<Event>,
                               Metavariables<3, void, ObserveMyNorms>>(
        R"(
  ObserveMyNorms:
    SubfileName: reduction0
    TensorsToObserve:
    - Name: Var0
      NormType: Max
      Components: Individual
        )");
  }

  test<void>(std::make_unique<ObserveNormsEvent<void>>(ObserveNormsEvent<void>{
                 "reduction0",
                 {{"Var0", "Max", "Individual"},
                  {"Var1", "Min", "Individual"},
                  {"Var0", "Max", "Sum"},
                  {"Var0TimesTwo", "Max", "Individual"},
                  {"Var0TimesThree", "Max", "Individual"},
                  {"Var1", "L1Norm", "Sum"},
                  {"Var1", "L1IntegralNorm", "Sum"},
                  {"Var1", "L2Norm", "Sum"},
                  {"Var1", "L2IntegralNorm", "Sum"},
                  {"Var1", "VolumeIntegral", "Sum"},
                  {"Var1", "L1Norm", "Individual"},
                  {"Var1", "L1IntegralNorm", "Individual"},
                  {"Var1", "L2Norm", "Individual"},
                  {"Var1", "L2IntegralNorm", "Individual"},
                  {"Var1", "VolumeIntegral", "Individual"},
                  {"Var1", "Min", "Sum"}}}),
             Spectral::Basis::FiniteDifference,
             Spectral::Quadrature::CellCentered, std::nullopt);

  // Test that L1Norm and L1IntegralNorm correctly take absolute values by using
  // a tensor with all-negative components. If abs() were dropped entirely, the
  // computed values would be negative rather than positive.
  {
    INFO("Negative values test");
    using metavariables = Metavariables<3, void>;
    using element_component = ElementComponent<metavariables>;
    using observer_component = MockObserverComponent<metavariables>;
    const typename element_component::array_index array_index(0);
    const Mesh<3> mesh{3, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
    const size_t num_points = mesh.number_of_grid_points();
    // det_inv_jacobian for a unit cube: volume = 1
    const Scalar<DataVector> det_inv_jacobian(num_points, cube(2.0));
    Variables<tmpl::list<Var0, Var1>> vars(num_points);
    // Fill Var0 with all -1.0; abs should give 1.0 for every point
    get(get<Var0>(vars)) = DataVector(num_points, -1.0);
    get<0>(get<Var1>(vars)) = DataVector(num_points, 0.0);
    get<1>(get<Var1>(vars)) = DataVector(num_points, 0.0);
    get<2>(get<Var1>(vars)) = DataVector(num_points, 0.0);

    ActionTesting::MockRuntimeSystem<metavariables> runner{{}};
    ActionTesting::emplace_component<element_component>(make_not_null(&runner),
                                                        array_index);
    ActionTesting::emplace_group_component<observer_component>(&runner);

    auto box = db::create<
        db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<metavariables>,
                          ::Events::Tags::ObserverMesh<3>,
                          ::Events::Tags::ObserverDetInvJacobian<
                              Frame::ElementLogical, Frame::Inertial>,
                          Tags::Variables<typename decltype(vars)::tags_list>,
                          observers::Tags::ObservationKey<void>>>(
        metavariables{}, mesh, det_inv_jacobian, vars,
        std::optional<std::string>{std::nullopt});

    const auto observe = std::make_unique<ObserveNormsEvent<void>>(
        ObserveNormsEvent<void>{"reduction0",
                                {{"Var0", "L1Norm", "Individual"},
                                 {"Var0", "L1IntegralNorm", "Individual"}}});

    auto obs_box = make_observation_box<tmpl::filter<
        typename ObserveNormsEvent<void>::compute_tags_for_observation_box,
        db::is_compute_tag<tmpl::_1>>>(make_not_null(&box));
    observe->run(make_not_null(&obs_box),
                 ActionTesting::cache<element_component>(runner, array_index),
                 array_index, std::add_pointer_t<element_component>{},
                 {"TimeName", 2.0});

    ActionTesting::invoke_queued_simple_action<observer_component>(
        make_not_null(&runner), 0);

    const auto& results = MockContributeReductionData::results;
    // L1Norm = (1/N) * sum(|u_i|) = 1.0 (sum over finalize reduction divides
    // by num_points). L1IntegralNorm = integral(|u|)/V = 1.0 (since all |-1|=1
    // and V=1). Both should be 1.0, not -1.0, proving abs() is applied.
    CHECK(results.l1_norm_values[0] == approx(1.0));
    CHECK(results.l1_integral_norm_values[0] == approx(1.0));
  }

  // varrying `Spherical` to test both spherical and axial symmetry, as well
  // as changing whether we include x=0
  test_cartoon<true, void>(
      std::make_unique<ObserveNormsEvent<void>>(
          ObserveNormsEvent<void>{"reduction0",
                                  {{"Var1", "L1IntegralNorm", "Sum"},
                                   {"Var1", "L1IntegralNorm", "Individual"},
                                   {"Var1", "L2IntegralNorm", "Sum"},
                                   {"Var1", "L2IntegralNorm", "Individual"},
                                   {"Var1", "VolumeIntegral", "Sum"},
                                   {"Var1", "VolumeIntegral", "Individual"}}}),
      std::nullopt, 0.0);
  test_cartoon<true, void>(
      std::make_unique<ObserveNormsEvent<void>>(
          ObserveNormsEvent<void>{"reduction0",
                                  {{"Var1", "L1IntegralNorm", "Sum"},
                                   {"Var1", "L1IntegralNorm", "Individual"},
                                   {"Var1", "L2IntegralNorm", "Sum"},
                                   {"Var1", "L2IntegralNorm", "Individual"},
                                   {"Var1", "VolumeIntegral", "Sum"},
                                   {"Var1", "VolumeIntegral", "Individual"}}}),
      std::nullopt, 0.5);
  test_cartoon<false, void>(
      std::make_unique<ObserveNormsEvent<void>>(
          ObserveNormsEvent<void>{"reduction0",
                                  {{"Var1", "L1IntegralNorm", "Sum"},
                                   {"Var1", "L1IntegralNorm", "Individual"},
                                   {"Var1", "L2IntegralNorm", "Sum"},
                                   {"Var1", "L2IntegralNorm", "Individual"},
                                   {"Var1", "VolumeIntegral", "Sum"},
                                   {"Var1", "VolumeIntegral", "Individual"}}}),
      std::nullopt, 0.0);
  test_cartoon<false, void>(
      std::make_unique<ObserveNormsEvent<void>>(
          ObserveNormsEvent<void>{"reduction0",
                                  {{"Var1", "L1IntegralNorm", "Sum"},
                                   {"Var1", "L1IntegralNorm", "Individual"},
                                   {"Var1", "L2IntegralNorm", "Sum"},
                                   {"Var1", "L2IntegralNorm", "Individual"},
                                   {"Var1", "VolumeIntegral", "Sum"},
                                   {"Var1", "VolumeIntegral", "Individual"}}}),
      std::nullopt, 1.5);
}
