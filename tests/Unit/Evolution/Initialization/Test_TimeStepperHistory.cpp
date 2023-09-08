// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Time/History.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

using VariablesType =
    Variables<tmpl::list<TestHelpers::Tags::Scalar<DataVector>>>;

using DtVariablesType =
    Variables<tmpl::list<::Tags::dt<TestHelpers::Tags::Scalar<DataVector>>>>;

template <size_t Dim>
struct TestSystem {
  using variables_tag =
      Tags::Variables<tmpl::list<TestHelpers::Tags::Scalar<DataVector>>>;
};

template <size_t Dim>
struct TestMetavariables {
  static constexpr size_t volume_dim = Dim;
  using system = TestSystem<Dim>;
};

template <typename T>
T f(const T& x, const std::array<double, 3>& c) {
  return c[0] + c[1] * x + c[2] * square(x);
}

template <size_t Dim>
VariablesType make_vars(
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& x, const double t) {
  const auto t_coeffs = std::array{0.5, 1.5, 2.5};
  const auto number_of_points = get<0>(x).size();
  VariablesType result{number_of_points, f(t, t_coeffs)};
  const auto x_coeffs = std::array{0.75, -1.75, 2.75};
  DataVector& s = get(get<TestHelpers::Tags::Scalar<DataVector>>(result));
  s *= f(x[0], x_coeffs);
  if constexpr (Dim > 1) {
    const auto y_coeffs = std::array{-0.25, 1.25, -2.25};
    s *= f(x[1], y_coeffs);
  }
  if constexpr (Dim > 2) {
    const auto z_coeffs = std::array{0.125, -1.625, -2.875};
    s *= f(x[2], z_coeffs);
  }
  return result;
}

template <size_t Dim>
DtVariablesType make_dt_vars(
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& x, const double t) {
  const auto dt_coeffs = std::array{1.5, 5.0, 0.0};
  const auto number_of_points = get<0>(x).size();
  DtVariablesType result{number_of_points, f(t, dt_coeffs)};
  const auto x_coeffs = std::array{0.75, -1.75, 2.75};
  DataVector& s =
      get(get<::Tags::dt<TestHelpers::Tags::Scalar<DataVector>>>(result));
  s *= f(x[0], x_coeffs);
  if constexpr (Dim > 1) {
    const auto y_coeffs = std::array{-0.25, 1.25, -2.25};
    s *= f(x[1], y_coeffs);
  }
  if constexpr (Dim > 2) {
    const auto z_coeffs = std::array{0.125, -1.625, -2.875};
    s *= f(x[2], z_coeffs);
  }
  return result;
}

template <size_t Dim>
void test_initialization() {
  const TimeSteppers::AdamsBashforth ab2{2};
  const Mesh<Dim> mesh{3, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  DtVariablesType dt_vars{};
  DtVariablesType expected_dt_vars{mesh.number_of_grid_points()};
  TimeSteppers::History<VariablesType> history{};
  TimeSteppers::History<VariablesType> expected_history{};
  const size_t starting_order = ab2.order() - ab2.number_of_past_steps();
  expected_history.integration_order(starting_order);
  Initialization::TimeStepperHistory<TestMetavariables<Dim>>::apply(
      make_not_null(&dt_vars), make_not_null(&history), ab2, mesh);
  CHECK(dt_vars.size() == expected_dt_vars.size());
  CHECK(history == expected_history);
}

void check_history(
    const TimeSteppers::History<VariablesType>& history,
    const TimeSteppers::History<VariablesType>& expected_history) {
  for (size_t i = 0; i < history.size(); ++i) {
    CHECK(history[i].time_step_id == expected_history[i].time_step_id);
    CHECK(history[i].value.has_value() ==
          expected_history[i].value.has_value());
    if (history[i].value.has_value()) {
      CHECK_VARIABLES_APPROX(*history[i].value, *expected_history[i].value);
    }
    CHECK_VARIABLES_APPROX(history[i].derivative,
                           expected_history[i].derivative);
  }
  const auto& substeps = history.substeps();
  const auto& expected_substeps = expected_history.substeps();
  for (size_t i = 0; i < substeps.size(); ++i) {
    CHECK(substeps[i].time_step_id == expected_substeps[i].time_step_id);
    CHECK(substeps[i].value.has_value() ==
          expected_substeps[i].value.has_value());
    if (substeps[i].value.has_value()) {
      CHECK_VARIABLES_APPROX(*substeps[i].value, *expected_substeps[i].value);
    }
    CHECK_VARIABLES_APPROX(substeps[i].derivative,
                           expected_substeps[i].derivative);
  }
}

template <size_t Dim>
void check(const TimeSteppers::History<VariablesType>& original_history,
           const TimeSteppers::History<VariablesType>& expected_history,
           const Mesh<Dim>& new_mesh, const ElementId<Dim>& element_id,
           const Mesh<Dim>& old_mesh, const Element<Dim>& element) {
  DtVariablesType dt_vars{};
  TimeSteppers::History<VariablesType> history = original_history;
  Initialization::ProjectTimeStepperHistory<TestMetavariables<Dim>>::apply(
      make_not_null(&dt_vars), make_not_null(&history), new_mesh, element_id,
      std::make_pair(old_mesh, element));
  CHECK(dt_vars.size() == new_mesh.number_of_grid_points());
  check_history(history, expected_history);
  Initialization::ProjectTimeStepperHistory<TestMetavariables<Dim>>::apply(
      make_not_null(&dt_vars), make_not_null(&history), old_mesh, element_id,
      std::make_pair(new_mesh, element));
  CHECK(dt_vars.size() == old_mesh.number_of_grid_points());
  check_history(history, original_history);
}

template <size_t Dim>
void test_p_refine() {
  const ElementId<Dim> element_id{0};
  const Element<Dim> element{element_id, DirectionMap<Dim, Neighbors<Dim>>{}};
  const Mesh<Dim> old_mesh{4, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
  std::array<size_t, Dim> new_extents{};
  std::iota(new_extents.begin(), new_extents.end(), 3_st);
  const Mesh<Dim> new_mesh{new_extents, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
  const auto x_old = logical_coordinates(old_mesh);
  const auto x_new = logical_coordinates(new_mesh);
  TimeSteppers::History<VariablesType> history{};
  TimeSteppers::History<VariablesType> expected_history{};
  check(history, expected_history, new_mesh, element_id, old_mesh, element);
  const Slab slab(0.0, 1.0);
  history.integration_order(4);
  expected_history.integration_order(4);
  TimeStepId time_step_id{true, 0, slab.start()};
  double t = time_step_id.substep_time();
  history.insert_initial(time_step_id, make_vars(x_old, t),
                         make_dt_vars(x_old, t));
  expected_history.insert_initial(time_step_id, make_vars(x_new, t),
                                  make_dt_vars(x_new, t));
  check(history, expected_history, new_mesh, element_id, old_mesh, element);
  time_step_id =
      TimeStepId{true, -1, slab.start() - Slab(-1.0, 0.0).duration() / 4};
  t = time_step_id.substep_time();
  history.insert_initial(time_step_id, make_vars(x_old, t),
                         make_dt_vars(x_old, t));
  expected_history.insert_initial(time_step_id, make_vars(x_new, t),
                                  make_dt_vars(x_new, t));
  check(history, expected_history, new_mesh, element_id, old_mesh, element);
  time_step_id =
      TimeStepId{true, -1, slab.start() - Slab(-1.0, 0.0).duration() / 2};
  t = time_step_id.substep_time();
  history.insert_initial(time_step_id, make_vars(x_old, t),
                         make_dt_vars(x_old, t));
  expected_history.insert_initial(time_step_id, make_vars(x_new, t),
                                  make_dt_vars(x_new, t));
  check(history, expected_history, new_mesh, element_id, old_mesh, element);
  time_step_id = TimeStepId{true, 0, slab.start() + slab.duration() / 4};
  t = time_step_id.substep_time();
  history.insert(time_step_id, make_vars(x_old, t), make_dt_vars(x_old, t));
  expected_history.insert(time_step_id, make_vars(x_new, t),
                          make_dt_vars(x_new, t));
  check(history, expected_history, new_mesh, element_id, old_mesh, element);
  const auto step_time = history.back().time_step_id.step_time();
  const auto step_size = slab.duration() / 4;
  time_step_id =
      TimeStepId{true, 0,         step_time,
                 1,    step_size, (step_time + slab.duration() / 4).value()};
  t = time_step_id.substep_time();
  history.insert(time_step_id, make_vars(x_old, t), make_dt_vars(x_old, t));
  expected_history.insert(time_step_id, make_vars(x_new, t),
                          make_dt_vars(x_new, t));
  check(history, expected_history, new_mesh, element_id, old_mesh, element);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Initialization.TimeStepperHistory",
                  "[Evolution][Unit]") {
  test_initialization<1>();
  test_initialization<2>();
  test_initialization<3>();
  test_p_refine<1>();
  test_p_refine<2>();
  test_p_refine<3>();
}
