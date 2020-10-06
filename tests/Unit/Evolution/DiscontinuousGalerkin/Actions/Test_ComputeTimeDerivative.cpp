// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivative.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/BoundarySchemes/FirstOrder/FirstOrderScheme.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "Time/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace {
enum SystemType { Conservative, Nonconservative };

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Var2 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

// Var3 is used as an extra quantity in the DataBox that the time derivative
// computation depends on. It could be loosely interpreted as a "primitive"
// variable, or a compute tag retrieved for the time derivative computation.
struct Var3 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct PackagedVar1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct PackagedVar2 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim, SystemType system_type>
struct TimeDerivative {
  struct Var3Squared : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  using temporary_tags = tmpl::list<Var3Squared>;
  using argument_tags = tmpl::list<Var1, Var2<Dim>, Var3>;

  // Conservative system
  static void apply(
      // Time derivatives returned by reference. All the tags in the
      // variables_tag in the system struct.
      const gsl::not_null<Scalar<DataVector>*> dt_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> dt_var2,

      // Fluxes returned by reference. Listed in the system struct as
      // flux_variables.
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> flux_var1,
      const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
          flux_var2,

      // Temporaries returned by reference. Listed in temporary_tags above.
      const gsl::not_null<Scalar<DataVector>*> square_var3,

      // Arguments listed in argument_tags above
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3) noexcept {
    get(*square_var3) = square(get(var3));

    // Set source terms
    get(*dt_var1) = get(*square_var3);
    for (size_t d = 0; d < Dim; ++d) {
      dt_var2->get(d) = get(var3) * d;
    }

    // Set fluxes
    for (size_t i = 0; i < Dim; ++i) {
      flux_var1->get(i) = square(get(var1)) * var2.get(i);
      for (size_t j = 0; j < Dim; ++j) {
        flux_var2->get(i, j) = var2.get(i) * var2.get(j) * get(var1);
        if (i == j) {
          flux_var2->get(i, j) += cube(get(var1));
        }
      }
    }
  }

  // Nonconservative system
  static void apply(
      // Time derivatives returned by reference. All the tags in the
      // variables_tag in the system struct.
      const gsl::not_null<Scalar<DataVector>*> dt_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> dt_var2,

      // Temporaries returned by reference. Listed in temporary_tags above.
      const gsl::not_null<Scalar<DataVector>*> square_var3,

      // Partial derivative arguments. Listed in the system struct as
      // gradient_variables.
      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_var1,
      const tnsr::iJ<DataVector, Dim, Frame::Inertial>& d_var2,

      // Arguments listed in argument_tags above
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2,
      const Scalar<DataVector>& var3) noexcept {
    get(*square_var3) = square(get(var3));

    // Set source terms and nonconservative products
    get(*dt_var1) = get(*square_var3);
    for (size_t d = 0; d < Dim; ++d) {
      get(*dt_var1) -= var2.get(d) * d_var1.get(d);
      dt_var2->get(d) = get(var3) * d;
      for (size_t i = 0; i < Dim; ++i) {
        dt_var2->get(d) -= get(var1) * var2.get(i) * d_var2.get(i, d);
      }
    }
  }
};

template <size_t Dim>
struct NonconservativeNormalDotFlux {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> var1_normal_dot_flux,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          var2_normal_dot_flux) noexcept {
    get(*var1_normal_dot_flux) = 1.1;
    for (size_t i = 0; i < Dim; ++i) {
      var2_normal_dot_flux->get(i) = 1.3 + i;
    }
  }
};

template <size_t Dim>
struct BoundaryTerms : tt::ConformsTo<dg::protocols::NumericalFlux> {
  struct MaxAbsCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  using variables_tags = tmpl::list<Var1, Var2<Dim>>;
  using variables_tag = Tags::Variables<variables_tags>;

  using package_field_tags =
      tmpl::push_back<tmpl::append<db::split_tag<db::add_tag_prefix<
                                       ::Tags::NormalDotFlux, variables_tag>>,
                                   variables_tags>,
                      MaxAbsCharSpeed>;
  using package_extra_tags = tmpl::list<>;

  using argument_tags = tmpl::push_back<tmpl::append<
      db::wrap_tags_in<::Tags::NormalDotFlux, variables_tags>, variables_tags>>;

  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT

  void package_data(
      const gsl::not_null<Scalar<DataVector>*> out_normal_dot_flux_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          out_normal_dot_flux_var2,
      const gsl::not_null<Scalar<DataVector>*> out_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> out_var2,
      const gsl::not_null<Scalar<DataVector>*> max_abs_char_speed,
      const Scalar<DataVector>& normal_dot_flux_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& normal_dot_flux_var2,
      const Scalar<DataVector>& var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& var2) const noexcept {
    *out_normal_dot_flux_var1 = normal_dot_flux_var1;
    *out_normal_dot_flux_var2 = normal_dot_flux_var2;
    *out_var1 = var1;
    *out_var2 = var2;
    get(*max_abs_char_speed) = 2.0 * max(get(var1));
  }
};

template <size_t Dim, SystemType system_type>
struct System {
  static constexpr bool has_primitive_and_conservative_vars = true;
  static constexpr size_t volume_dim = Dim;

  using variables_tag = Tags::Variables<tmpl::list<Var1, Var2<Dim>>>;
  using flux_variables =
      tmpl::conditional_t<system_type == SystemType::Conservative,
                          tmpl::list<Var1, Var2<Dim>>, tmpl::list<>>;
  using gradient_variables =
      tmpl::conditional_t<system_type == SystemType::Conservative, tmpl::list<>,
                          tmpl::list<Var1, Var2<Dim>>>;

  using compute_volume_time_derivative = TimeDerivative<Dim, system_type>;

  using normal_dot_fluxes = NonconservativeNormalDotFlux<Dim>;
};

template <typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;

  using internal_directions =
      domain::Tags::InternalDirections<Metavariables::volume_dim>;
  using boundary_directions_interior =
      domain::Tags::BoundaryDirectionsInterior<Metavariables::volume_dim>;

  using common_simple_tags = tmpl::list<
      ::Tags::TimeStepId, typename Metavariables::system::variables_tag,
      db::add_tag_prefix<::Tags::dt,
                         typename Metavariables::system::variables_tag>,
      Var3, domain::Tags::Mesh<Metavariables::volume_dim>,
      domain::Tags::Interface<
          internal_directions,
          db::add_tag_prefix<
              ::Tags::NormalDotFlux,
              typename metavariables::boundary_scheme::variables_tag>>,
      domain::Tags::Element<Metavariables::volume_dim>,
      domain::Tags::InverseJacobian<Metavariables::volume_dim, Frame::Logical,
                                    Frame::Inertial>,
      domain::Tags::MeshVelocity<Metavariables::volume_dim>,
      domain::Tags::DivMeshVelocity>;
  using simple_tags = tmpl::conditional_t<
      Metavariables::system_type == SystemType::Conservative,
      tmpl::push_back<
          common_simple_tags,
          db::add_tag_prefix<
              ::Tags::Flux, typename Metavariables::system::variables_tag,
              tmpl::size_t<Metavariables::volume_dim>, Frame::Inertial>>,
      common_simple_tags>;
  using compute_tags = tmpl::list<
      domain::Tags::InternalDirectionsCompute<Metavariables::volume_dim>,
      domain::Tags::InterfaceCompute<
          internal_directions,
          domain::Tags::Direction<Metavariables::volume_dim>>,
      domain::Tags::InterfaceCompute<
          internal_directions,
          domain::Tags::InterfaceMesh<Metavariables::volume_dim>>,
      domain::Tags::BoundaryDirectionsInteriorCompute<
          Metavariables::volume_dim>,
      domain::Tags::InterfaceCompute<
          boundary_directions_interior,
          domain::Tags::Direction<Metavariables::volume_dim>>,
      domain::Tags::InterfaceCompute<
          boundary_directions_interior,
          domain::Tags::InterfaceMesh<Metavariables::volume_dim>>,
      domain::Tags::Slice<internal_directions, Var1>,
      domain::Tags::Slice<internal_directions,
                          Var2<Metavariables::volume_dim>>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>,
              dg::Actions::InitializeMortars<
                  typename Metavariables::boundary_scheme>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              evolution::dg::Actions::ComputeTimeDerivative<Metavariables>>>>;
};

template <size_t Dim, SystemType SystemTypeIn>
struct Metavariables {
  static constexpr dg::Formulation dg_formulation =
      dg::Formulation::StrongInertial;
  static constexpr size_t volume_dim = Dim;
  static constexpr SystemType system_type = SystemTypeIn;
  using system = System<Dim, system_type>;
  using boundary_scheme = ::dg::FirstOrderScheme::FirstOrderScheme<
      Dim, typename system::variables_tag,
      db::add_tag_prefix<::Tags::dt, typename system::variables_tag>,
      Tags::NumericalFlux<BoundaryTerms<Dim>>, Tags::TimeStepId>;
  using normal_dot_numerical_flux = Tags::NumericalFlux<BoundaryTerms<Dim>>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::InitialExtents<Dim>, normal_dot_numerical_flux>;

  using component_list = tmpl::list<component<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

template <bool UseMovingMesh, size_t Dim, SystemType system_type>
void test_impl() noexcept {
  using metavars = Metavariables<Dim, system_type>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  // The reference element in 2d denoted by X below:
  // ^ eta
  // +-+-+> xi
  // |X| |
  // +-+-+
  // | | |
  // +-+-+
  //
  // The "self_id" for the element that we are considering is marked by an X in
  // the diagram. We consider a configuration with one neighbor in the +xi
  // direction (east_id), and (in 2d and 3d) one in the -eta (south_id)
  // direction.
  //
  // In 1d there aren't any projections to test, and in 3d we only have 1
  // element in the z-direction.
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  ElementId<Dim> self_id{};
  ElementId<Dim> east_id{};
  ElementId<Dim> south_id{};  // not used in 1d

  if constexpr (Dim == 1) {
    self_id = ElementId<Dim>{0, {{{1, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
  } else if constexpr (Dim == 2) {
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}}}};
    south_id = ElementId<Dim>{1, {{{1, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
    neighbors[Direction<Dim>::lower_eta()] = Neighbors<Dim>{{south_id}, {}};
  } else {
    static_assert(Dim == 3, "Only implemented tests in 1, 2, and 3d");
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}, {0, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}, {0, 0}}}};
    south_id = ElementId<Dim>{1, {{{1, 0}, {0, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
    neighbors[Direction<Dim>::lower_eta()] = Neighbors<Dim>{{south_id}, {}};
  }
  const Element<Dim> element{self_id, neighbors};
  MockRuntimeSystem runner{
      {std::vector<std::array<size_t, Dim>>{make_array<Dim>(2_st),
                                            make_array<Dim>(3_st)},
       typename metavars::normal_dot_numerical_flux::type{}}};

  const Mesh<Dim> mesh{2, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};

  ::InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial> inv_jac{
      mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < Dim; ++i) {
    inv_jac.get(i, i) = 1.0;
  }

  boost::optional<tnsr::I<DataVector, Dim, Frame::Inertial>> mesh_velocity{};
  boost::optional<Scalar<DataVector>> div_mesh_velocity{};
  if (UseMovingMesh) {
    const std::array<double, 3> velocities = {{1.2, -1.4, 0.3}};
    mesh_velocity =
        tnsr::I<DataVector, Dim, Frame::Inertial>{mesh.number_of_grid_points()};
    for (size_t i = 0; i < Dim; ++i) {
      mesh_velocity->get(i) = gsl::at(velocities, i);
    }
    div_mesh_velocity = Scalar<DataVector>{mesh.number_of_grid_points(), 1.5};
  }

  Variables<tmpl::list<Var1, Var2<Dim>>> evolved_vars{
      mesh.number_of_grid_points()};
  Scalar<DataVector> var3{mesh.number_of_grid_points()};
  // Set the variables so they are constant in y & z
  for (size_t i = 0; i < mesh.number_of_grid_points(); i += 2) {
    get(get<Var1>(evolved_vars))[i] = 3.0;
    get(get<Var1>(evolved_vars))[i + 1] = 2.0;
    for (size_t j = 0; j < Dim; ++j) {
      get<Var2<Dim>>(evolved_vars).get(j)[i] = j + 1.0;
      get<Var2<Dim>>(evolved_vars).get(j)[i + 1] = j + 3.0;
    }
    get(var3)[i] = 5.0;
    get(var3)[i + 1] = 6.0;
  }
  Variables<tmpl::list<::Tags::dt<Var1>, ::Tags::dt<Var2<Dim>>>>
      dt_evolved_vars{mesh.number_of_grid_points()};

  using flux_tags =
      tmpl::conditional_t<system_type == SystemType::Nonconservative,
                          tmpl::list<>, tmpl::list<Var1, Var2<Dim>>>;

  std::unordered_map<::Direction<Dim>,
                     Variables<tmpl::list<::Tags::NormalDotFlux<Var1>,
                                          ::Tags::NormalDotFlux<Var2<Dim>>>>>
      normal_dot_fluxes_interface{};
  const size_t interface_grid_points =
      mesh.slice_away(0).number_of_grid_points();
  for (const auto& [direction, nhbrs] : element.neighbors()) {
    (void)nhbrs;
    normal_dot_fluxes_interface[direction].initialize(interface_grid_points,
                                                      0.0);
  }

  const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};
  if constexpr (not std::is_same_v<tmpl::list<>, flux_tags>) {
    ActionTesting::emplace_component_and_initialize<component<metavars>>(
        &runner, self_id,
        {time_step_id, evolved_vars, dt_evolved_vars, var3, mesh,
         normal_dot_fluxes_interface, element, inv_jac, mesh_velocity,
         div_mesh_velocity,
         Variables<db::wrap_tags_in<::Tags::Flux, flux_tags, tmpl::size_t<Dim>,
                                    Frame::Inertial>>{2, -100.0}});
  } else {
    ActionTesting::emplace_component_and_initialize<component<metavars>>(
        &runner, self_id,
        {time_step_id, evolved_vars, dt_evolved_vars, var3, mesh,
         normal_dot_fluxes_interface, element, inv_jac, mesh_velocity,
         div_mesh_velocity});
  }
  // Initialize the mortars
  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  self_id);
  // Start testing the actual dg::ComputeTimeDerivative action
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);
  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  self_id);

  Variables<tmpl::list<::Tags::dt<Var1>, ::Tags::dt<Var2<Dim>>>>
      expected_dt_evolved_vars{mesh.number_of_grid_points()};
  if constexpr (system_type == SystemType::Nonconservative) {
    for (size_t i = 0; i < mesh.number_of_grid_points(); i += 2) {
      get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars))[i] = 25.5;
      get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars))[i + 1] = 37.5;
      for (size_t j = 0; j < Dim; ++j) {
        get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j)[i] = -3.0;
        get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j)[i + 1] =
            -6.0;
      }
    }
    for (size_t j = 0; j < Dim; ++j) {
      get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j) +=
          j * get(var3);
    }
    if (UseMovingMesh) {
      get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) +=
          -0.5 * get<0>(*mesh_velocity);
      for (size_t j = 0; j < Dim; ++j) {
        get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j) +=
            get<0>(*mesh_velocity);
      }
    }
  } else if constexpr (system_type == SystemType::Conservative) {
    // Deal with source terms:
    get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) = square(get(var3));
    for (size_t j = 0; j < Dim; ++j) {
      get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j) =
          j * get(var3);
    }
    // Deal with volume flux divergence
    get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) -= 1.5;
    get<0>(get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars)) += 2.0;
    if constexpr (Dim > 1) {
      get<1>(get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars)) -= 9.0;
    }
    if constexpr (Dim > 2) {
      get<2>(get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars)) -= 10.5;
    }
    if constexpr (UseMovingMesh) {
      get(get<::Tags::dt<Var1>>(expected_dt_evolved_vars)) -=
          1.5 * get(get<Var1>(evolved_vars)) - mesh_velocity->get(0) * -0.5;
      for (size_t j = 0; j < Dim; ++j) {
        get<::Tags::dt<Var2<Dim>>>(expected_dt_evolved_vars).get(j) -=
            1.5 * get<Var2<Dim>>(evolved_vars).get(j) -
            mesh_velocity->get(0) * 1.0;
      }
    }
  }

  CHECK(ActionTesting::get_databox_tag<
            component<metavars>,
            db::add_tag_prefix<::Tags::dt,
                               typename metavars::system::variables_tag>>(
            runner, self_id) == expected_dt_evolved_vars);

  const auto get_tag = [&runner, &self_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<component<metavars>, tag>(runner,
                                                                    self_id);
  };

  const auto check_mortar =
      [&get_tag, &time_step_id](
          const std::pair<Direction<Dim>, ElementId<Dim>>& mortar_id,
          const size_t num_points) noexcept {
        CAPTURE(mortar_id);
        CAPTURE(Dim);
        const auto& all_mortar_data = get_tag(
            ::Tags::Mortars<typename metavars::boundary_scheme::mortar_data_tag,
                            Dim>{});
        const dg::SimpleBoundaryData<
            typename BoundaryTerms<Dim>::package_field_tags>& boundary_data =
            all_mortar_data.at(mortar_id).local_data(time_step_id);
        CHECK(boundary_data.field_data.number_of_grid_points() == num_points);
        // Actually checking all the fields is a total nightmare because the
        // whole boundary scheme code is a complete disaster. The checks will be
        // added once the boundary scheme code is cleaned up.
        if constexpr (system_type == SystemType::Conservative) {
          const Scalar<DataVector> expected_var1_normal_dot_flux(num_points,
                                                                 0.0);
          CHECK(get<::Tags::NormalDotFlux<Var1>>(boundary_data.field_data) ==
                expected_var1_normal_dot_flux);
        }
      };

  const auto mortar_id_east =
      std::make_pair(Direction<Dim>::upper_xi(), east_id);
  check_mortar(mortar_id_east, Dim == 1 ? 1 : Dim == 2 ? 2 : 4);
  if constexpr (Dim > 1) {
    const auto mortar_id_south =
        std::make_pair(Direction<Dim>::lower_eta(), south_id);
    // The number of points on the mortar should be 3 in 2d and 9 and 3d because
    // we do projection to the southern neighbor, which has 3 points per
    // dimension.
    check_mortar(mortar_id_south, Dim == 2 ? 3 : 9);
  }
}

template <SystemType system_type>
void test() noexcept {
  test_impl<false, 1, system_type>();
  test_impl<true, 1, system_type>();
  test_impl<false, 2, system_type>();
  test_impl<true, 2, system_type>();
  test_impl<false, 3, system_type>();
  test_impl<true, 3, system_type>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.ComputeTimeDerivative",
                  "[Unit][Evolution][Actions]") {
  // The test is designed to test the `ComputeTimeDerivative` action for DG.
  // This action does a lot:
  //
  // - compute partial derivatives as needed
  // - compute the time derivative from
  //   `System::compute_volume_time_derivative`. This includes fluxes, sources,
  //   and nonconservative products.
  // - adds moving mesh terms as needed.
  // - compute flux divergence and add to the time derivative.
  // - compute mortar data for internal boundaries.
  //
  // The action supports conservative systems, and nonconservative systems
  // (mixed conservative-nonconservative systems will be added in the future).
  //
  // To test the action thoroughly we need to test a lot of different
  // combinations:
  //
  // - system type (conservative/nonconservative), using the enum SystemType
  // - 1d, 2d, 3d
  // - whether the mesh is moving or not

  test<SystemType::Nonconservative>();
  test<SystemType::Conservative>();
}
