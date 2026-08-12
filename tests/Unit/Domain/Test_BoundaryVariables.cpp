// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/MathWrapper.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/BoundaryVariables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Vector>
struct ScalarTag : db::SimpleTag {
  using type = Scalar<Vector>;
};

template <typename Vector>
struct VectorTag : db::SimpleTag {
  using type = tnsr::I<Vector, 2, Frame::Inertial>;
};

template <size_t Dim, typename Tags>
void check_empty(const gsl::not_null<BoundaryVariables<Dim, Tags>*> empty) {
  CHECK(empty->points_per_direction().empty());
  CHECK(empty->empty());
  CHECK(empty->size() == 0);
  CHECK(empty->buffer().size() == 0);
  CHECK(&empty->buffer() == &std::as_const(*empty).buffer());
  CHECK(empty->variables().empty());
  CHECK(std::as_const(*empty).variables().empty());
  CHECK(*empty == BoundaryVariables<Dim, Tags>{});
  CHECK_FALSE(*empty != BoundaryVariables<Dim, Tags>{});
  CHECK(not contains_allocations(*empty));
}

// The math wrapper test helper assumes you can do math on the type
// itself, which is the usual use.  It doesn't seem worth defining all
// the arithmetic overloads here, though, so we do some checks without
// the helper.
template <size_t Dim, typename Vector>
void test_math_wrapper(const typename Vector::value_type& value) {
  using BoundaryVars =
      BoundaryVariables<Dim, tmpl::list<ScalarTag<Vector>, VectorTag<Vector>>>;

  const DirectionMap<Dim, size_t> points_per_direction{
      {Direction<Dim>::upper_xi(), 3}, {Direction<Dim>::lower_xi(), 2}};

  const BoundaryVars vars1(points_per_direction, value);
  BoundaryVars vars2(points_per_direction);

  const auto const_wrapper = make_math_wrapper(vars1);
  const auto mutable_wrapper = make_math_wrapper(make_not_null(&vars2));
  static_assert(
      std::is_same_v<decltype(const_wrapper), const MathWrapper<const Vector>>);
  static_assert(
      std::is_same_v<decltype(mutable_wrapper), const MathWrapper<Vector>>);

  *mutable_wrapper = 2.0 * *const_wrapper;

  CHECK(vars2 == BoundaryVars(points_per_direction, 2.0 * value));

  {
    BoundaryVars into_math_wrapper_test{points_per_direction, value};
    const auto* const data_ptr = into_math_wrapper_test.buffer().data();
    const auto vector =
        into_math_wrapper_type(std::move(into_math_wrapper_test));
    static_assert(std::is_same_v<decltype(vector), const Vector>);
    CHECK(vector == Vector(3 * 5, value));
    CHECK(vector.data() == data_ptr);
    // Check moved-out state is valid
    check_empty(make_not_null(&into_math_wrapper_test));
  }
}

template <size_t Dim, typename Vector>
void test_make_with_value(const typename Vector::value_type& value) {
  using BoundaryVarsResult =
      BoundaryVariables<Dim, tmpl::list<ScalarTag<Vector>, VectorTag<Vector>>>;
  using BoundaryVarsPattern =
      BoundaryVariables<Dim, tmpl::list<VectorTag<Vector>>>;

  const DirectionMap<Dim, size_t> points_per_direction1{
      {Direction<Dim>::upper_xi(), 3}, {Direction<Dim>::lower_xi(), 2}};
  const DirectionMap<Dim, size_t> points_per_direction2{
      {Direction<Dim>::upper_xi(), 1}, {Direction<Dim>::lower_xi(), 4}};

  auto made = make_with_value<BoundaryVarsResult>(
      BoundaryVarsPattern{points_per_direction1}, value);
  CHECK(made == BoundaryVarsResult{points_per_direction1, value});

  // Setting to the current value does nothing.
  set_number_of_grid_points(make_not_null(&made),
                            BoundaryVarsPattern{points_per_direction1});
  CHECK(made == BoundaryVarsResult{points_per_direction1, value});

  set_number_of_grid_points(make_not_null(&made),
                            BoundaryVarsPattern{points_per_direction2});
  CHECK(made.points_per_direction() == points_per_direction2);
  CHECK(get(get<ScalarTag<Vector>>(
                made.variables().at(Direction<Dim>::upper_xi())))
            .size() == 1);
}

template <size_t Dim, typename Vector>
void test(const typename Vector::value_type& value) {
  using BoundaryVars =
      BoundaryVariables<Dim, tmpl::list<ScalarTag<Vector>, VectorTag<Vector>>>;

  {
    BoundaryVars empty{};
    check_empty(make_not_null(&empty));
    BoundaryVars empty2{DirectionMap<Dim, size_t>{}};
    check_empty(make_not_null(&empty2));
  }

  const size_t upper_points = 5;
  const size_t lower_points = 3;
  const DirectionMap<Dim, size_t> points_per_direction{
      {Direction<Dim>::upper_xi(), upper_points},
      {Direction<Dim>::lower_xi(), lower_points}};

  BoundaryVars boundary_vars(points_per_direction, value);
  CHECK(boundary_vars.points_per_direction() == points_per_direction);
  CHECK(not boundary_vars.empty());
  CHECK(boundary_vars.size() == 3 * (upper_points + lower_points));
  CHECK(&boundary_vars.buffer() == &std::as_const(boundary_vars).buffer());
  CHECK(contains_allocations(boundary_vars));

  {
    const DirectionMap<Dim, size_t> points_per_direction2{
        {Direction<Dim>::upper_xi(), upper_points + 2}};
    BoundaryVars boundary_vars2(points_per_direction2);
    CHECK(boundary_vars2.points_per_direction() == points_per_direction2);
    CHECK(not boundary_vars2.empty());
    CHECK(boundary_vars2.size() == 3 * (upper_points + 2));
    CHECK(&boundary_vars2.buffer() == &std::as_const(boundary_vars2).buffer());
    CHECK(boundary_vars2.variables().size() == 1);
    CHECK(get(get<ScalarTag<Vector>>(
                  boundary_vars2.variables().at(Direction<Dim>::upper_xi())))
              .size() == upper_points + 2);
    CHECK(contains_allocations(boundary_vars2));

    boundary_vars2.initialize(points_per_direction, value);
    CHECK(boundary_vars2.points_per_direction() == points_per_direction);
    CHECK(boundary_vars2 == boundary_vars);
    CHECK(boundary_vars2.variables().size() == 2);
    CHECK(get(get<ScalarTag<Vector>>(
                  boundary_vars2.variables().at(Direction<Dim>::upper_xi())))
              .size() == upper_points);

    boundary_vars2.initialize(points_per_direction2);
    CHECK(boundary_vars2.points_per_direction() == points_per_direction2);
    CHECK(boundary_vars2.size() == 3 * (upper_points + 2));
    CHECK(boundary_vars2.variables().size() == 1);
    CHECK(get(get<ScalarTag<Vector>>(
                  boundary_vars2.variables().at(Direction<Dim>::upper_xi())))
              .size() == upper_points + 2);
  }

  const BoundaryVars single_side_boundary_vars(
      {{Direction<Dim>::upper_xi(), upper_points + lower_points}}, value);
  CHECK(boundary_vars != single_side_boundary_vars);
  CHECK_FALSE(boundary_vars == single_side_boundary_vars);
  CHECK(boundary_vars.buffer() == single_side_boundary_vars.buffer());

  const auto check_values =
      [&lower_points, &upper_points](
          const auto& vars_map,
          const std::array<typename Vector::value_type, 4>& expected_values) {
        CHECK(vars_map.size() == 2);
        const auto& upper_vars = vars_map.at(Direction<Dim>::upper_xi());
        const auto& lower_vars = vars_map.at(Direction<Dim>::lower_xi());
        CHECK(upper_vars.number_of_grid_points() == upper_points);
        CHECK(lower_vars.number_of_grid_points() == lower_points);
        CHECK(get(get<ScalarTag<Vector>>(upper_vars)) ==
              Vector(upper_points, expected_values[0]));
        CHECK(get<1>(get<VectorTag<Vector>>(upper_vars)) ==
              Vector(upper_points, expected_values[1]));
        CHECK(get(get<ScalarTag<Vector>>(lower_vars)) ==
              Vector(lower_points, expected_values[2]));
        CHECK(get<1>(get<VectorTag<Vector>>(lower_vars)) ==
              Vector(lower_points, expected_values[3]));
        // The order has to be consistent, although we don't document it.
        // Turns out xi_upper is after xi_lower.
        const auto ptr_difference =
            get(get<ScalarTag<Vector>>(upper_vars)).data() -
            get(get<ScalarTag<Vector>>(lower_vars)).data();
        CHECK(ptr_difference == 3 * lower_points);
      };

  check_values(boundary_vars.variables(), {value, value, value, value});
  check_values(std::as_const(boundary_vars).variables(),
               {value, value, value, value});

  get(get<ScalarTag<Vector>>(
      boundary_vars.variables().at(Direction<Dim>::upper_xi()))) *= 2.0;
  get<1>(get<VectorTag<Vector>>(
      boundary_vars.variables().at(Direction<Dim>::lower_xi()))) *= 3.0;

  check_values(boundary_vars.variables(),
               {2.0 * value, value, value, 3.0 * value});
  check_values(std::as_const(boundary_vars).variables(),
               {2.0 * value, value, value, 3.0 * value});

  const auto check_against_original = [&](BoundaryVars& other) {
    CHECK(boundary_vars == other);
    CHECK_FALSE(boundary_vars != other);
    CHECK(boundary_vars.buffer().data() != other.buffer().data());
    check_values(other.variables(), {2.0 * value, value, value, 3.0 * value});
    check_values(std::as_const(other).variables(),
                 {2.0 * value, value, value, 3.0 * value});
  };
  BoundaryVars copy1(boundary_vars);
  check_against_original(copy1);
  BoundaryVars copy2{};
  copy2 = boundary_vars;
  check_against_original(copy2);
  const auto* const copy_data = copy2.buffer().data();
  BoundaryVars moved1(std::move(copy2));
  // Check moved-out state is valid
  check_empty(make_not_null(&copy2));
  check_against_original(moved1);
  CHECK(moved1.buffer().data() == copy_data);
  BoundaryVars moved2{};
  moved2 = std::move(moved1);
  // Check moved-out state is valid
  check_empty(make_not_null(&moved1));
  check_against_original(moved2);
  CHECK(moved2.buffer().data() == copy_data);

  boundary_vars.buffer() *= 4.0;

  check_values(boundary_vars.variables(),
               {8.0 * value, 4.0 * value, 4.0 * value, 12.0 * value});
  check_values(std::as_const(boundary_vars).variables(),
               {8.0 * value, 4.0 * value, 4.0 * value, 12.0 * value});

  CHECK(boundary_vars != copy1);
  CHECK_FALSE(boundary_vars == copy1);

  auto serialized = serialize_and_deserialize(boundary_vars);
  CHECK(serialized == boundary_vars);
  check_values(serialized.variables(),
               {8.0 * value, 4.0 * value, 4.0 * value, 12.0 * value});
  check_values(std::as_const(serialized).variables(),
               {8.0 * value, 4.0 * value, 4.0 * value, 12.0 * value});

  boundary_vars.clear();
  check_empty(make_not_null(&boundary_vars));

  test_math_wrapper<Dim, Vector>(value);
  test_make_with_value<Dim, Vector>(value);
}

SPECTRE_TEST_CASE("Unit.Domain.BoundaryVariables", "[Unit][Domain]") {
  test<1, DataVector>(2.0);
  test<1, ComplexDataVector>({2.0, 4.0});
  test<2, DataVector>(2.0);
  test<2, ComplexDataVector>({2.0, 4.0});
  test<3, DataVector>(2.0);
  test<3, ComplexDataVector>({2.0, 4.0});
}
}  // namespace
