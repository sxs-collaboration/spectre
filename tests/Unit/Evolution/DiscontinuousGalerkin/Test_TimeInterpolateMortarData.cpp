// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeInterpolateMortarData.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/Slab.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Math.hpp"

namespace {
#ifdef SPECTRE_DEBUG
constexpr bool spectre_debug = true;
#else
constexpr bool spectre_debug = false;
#endif

class RandomPolynomial {
 public:
  RandomPolynomial(const size_t degree,
                   const gsl::not_null<std::mt19937*> gen) {
    std::uniform_real_distribution dist(-1.0, 1.0);
    coefs_.reserve(degree + 1);
    for (size_t i = 0; i <= degree; ++i) {
      coefs_.push_back(
          make_with_random_values<DataVector>(gen, make_not_null(&dist), 5_st));
    }
  }

  DataVector operator()(const double t) const {
    return evaluate_polynomial(coefs_, DataVector(5_st, t));
  }

 private:
  std::vector<DataVector> coefs_{};
};

template <size_t Dim>
evolution::dg::MortarData<Dim> make_data(
    const std::array<bool, 7>& fields,
    const std::array<RandomPolynomial, 4>& values, const double time) {
  const Mesh<Dim - 1> mortar_mesh(3, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::Gauss);
  const Mesh<Dim - 1> face_mesh(5, Spectral::Basis::Legendre,
                                Spectral::Quadrature::GaussLobatto);
  const Mesh<Dim> volume_mesh(6, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto);
  const auto opt = [](const bool include, auto value) {
    return include ? std::optional{std::move(value)} : std::nullopt;
  };
  return {
      .mortar_data = opt(fields[0], values[0](time)),
      .face_normal_magnitude =
          opt(fields[1], Scalar<DataVector>{values[1](time)}),
      .face_det_jacobian = opt(fields[2], Scalar<DataVector>{values[2](time)}),
      .volume_det_inv_jacobian =
          opt(fields[3], Scalar<DataVector>{values[3](time)}),
      .mortar_mesh = opt(fields[4], mortar_mesh),
      .face_mesh = opt(fields[5], face_mesh),
      .volume_mesh = opt(fields[6], volume_mesh),
  };
}

// Create data with wrong values to make sure they are overwritten.
template <size_t Dim>
evolution::dg::MortarData<Dim> make_data_to_overwrite() {
  const Mesh<Dim - 1> mortar_mesh(7, Spectral::Basis::Legendre,
                                  Spectral::Quadrature::GaussLobatto);
  const Mesh<Dim - 1> face_mesh(8, Spectral::Basis::Legendre,
                                Spectral::Quadrature::Gauss);
  const Mesh<Dim> volume_mesh(9, Spectral::Basis::Legendre,
                              Spectral::Quadrature::Gauss);
  return {
      .mortar_data = DataVector(12),
      .face_normal_magnitude = Scalar<DataVector>{DataVector(12)},
      .face_det_jacobian = Scalar<DataVector>{DataVector(12)},
      .volume_det_inv_jacobian = Scalar<DataVector>{DataVector(12)},
      .mortar_mesh = mortar_mesh,
      .face_mesh = face_mesh,
      .volume_mesh = volume_mesh,
  };
}

template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> gen) {
  std::bernoulli_distribution true_false(0.5);

  const auto do_interpolate = [](const TimeSteppers::ConstBoundaryHistorySide<
                                     evolution::dg::MortarData<Dim>>& hist,
                                 const double time,
                                 const bool include_geometric_data) {
    auto interpolated = make_data_to_overwrite<Dim>();
    time_interpolate_mortar_data(make_not_null(&interpolated), hist, time,
                                 include_geometric_data);
    const auto returned =
        time_interpolate_mortar_data(hist, time, include_geometric_data);
    CHECK(*returned == interpolated);
    CHECK(returned.is_owning() != alg::found_if(hist, [&](const auto& entry) {
            return entry.substep_time() == time;
          }));
    return interpolated;
  };

  const auto check_approx_opt = [](const auto& a, const auto& b) {
    CHECK(a.has_value() == b.has_value());
    if (a.has_value() and b.has_value()) {
      CHECK_ITERABLE_APPROX(*a, *b);
    }
  };

  const Slab slab(-1.0, 2.0);
  const auto make_id = [&slab](const Rational& frac) {
    return TimeStepId(true, 0, slab.start() + slab.duration() * frac);
  };

  for (size_t stepper_order = 1; stepper_order < 5; ++stepper_order) {
    const auto fields_to_include = make_with_random_values<std::array<bool, 7>>(
        gen, make_not_null(&true_false));
    const std::array<RandomPolynomial, 4> polys{{{stepper_order - 1, gen},
                                                 {stepper_order - 1, gen},
                                                 {stepper_order - 1, gen},
                                                 {stepper_order - 1, gen}}};

    TimeSteppers::BoundaryHistory<evolution::dg::MortarData<Dim>,
                                  evolution::dg::MortarData<Dim>, DataVector>
        boundary_history{};
    const auto side_hist = boundary_history.remote();
    if constexpr (spectre_debug) {
      CHECK_THROWS_WITH(do_interpolate(side_hist, 0.0, false),
                        Catch::Matchers::ContainsSubstring(
                            "Cannot interpolate from an empty history"));
      CHECK_THROWS_WITH(do_interpolate(side_hist, 0.0, true),
                        Catch::Matchers::ContainsSubstring(
                            "Cannot interpolate from an empty history"));
    }
    {
      const TimeStepId id = make_id({7, 8});
      const auto data =
          make_data<Dim>(fields_to_include, polys, id.substep_time());
      side_hist.insert_initial(id, stepper_order, data);
    }

    const double test_time = (slab.start() + slab.duration() * 15 / 16).value();

    for (size_t i = 0; i < stepper_order - 1; ++i) {
      for (const auto& id : side_hist) {
        // Requesting an entry in the history should always work.
        CHECK(do_interpolate(side_hist, id.substep_time(), false) ==
              side_hist.data(id));
        CHECK(do_interpolate(side_hist, id.substep_time(), true) ==
              side_hist.data(id));
      }
      if constexpr (spectre_debug) {
        CHECK_THROWS_WITH(do_interpolate(side_hist, test_time, false),
                          Catch::Matchers::ContainsSubstring(
                              "Insufficient entries to interpolate"));
        CHECK_THROWS_WITH(do_interpolate(side_hist, test_time, true),
                          Catch::Matchers::ContainsSubstring(
                              "Insufficient entries to interpolate"));
      }

      const TimeStepId id = make_id({3 - static_cast<int32_t>(i), 4});
      side_hist.insert_initial(
          id, stepper_order,
          make_data<Dim>(fields_to_include, polys, id.substep_time()));
    }

    const auto test_interpolated_geom =
        do_interpolate(side_hist, test_time, true);
    const auto test_interpolated_nogeom =
        do_interpolate(side_hist, test_time, false);
    const auto expected = make_data<Dim>(fields_to_include, polys, test_time);

    check_approx_opt(test_interpolated_geom.mortar_data, expected.mortar_data);
    CHECK(test_interpolated_geom.mortar_mesh == expected.mortar_mesh);
    check_approx_opt(test_interpolated_geom.face_normal_magnitude,
                     expected.face_normal_magnitude);
    check_approx_opt(test_interpolated_geom.face_det_jacobian,
                     expected.face_det_jacobian);
    check_approx_opt(test_interpolated_geom.volume_det_inv_jacobian,
                     expected.volume_det_inv_jacobian);
    CHECK(test_interpolated_geom.face_mesh == expected.face_mesh);
    CHECK(test_interpolated_geom.volume_mesh == expected.volume_mesh);

    check_approx_opt(test_interpolated_nogeom.mortar_data,
                     expected.mortar_data);
    CHECK(test_interpolated_nogeom.mortar_mesh == expected.mortar_mesh);
    CHECK(not test_interpolated_nogeom.face_normal_magnitude.has_value());
    CHECK(not test_interpolated_nogeom.face_det_jacobian.has_value());
    CHECK(not test_interpolated_nogeom.volume_det_inv_jacobian.has_value());
    CHECK(not test_interpolated_nogeom.face_mesh.has_value());
    CHECK(not test_interpolated_nogeom.volume_mesh.has_value());

    // Any earlier values should be ignored
    {
      const TimeStepId id = make_id(0);
      side_hist.insert_initial(id, stepper_order,
                               make_data_to_overwrite<Dim>());
      CHECK(do_interpolate(side_hist, test_time, true) ==
            test_interpolated_geom);
      CHECK(do_interpolate(side_hist, test_time, false) ==
            test_interpolated_nogeom);
    }

    // First-order only uses one value, so it can't be inconsistent.
    if (stepper_order > 1) {
      const auto toggle_optional = [](auto& opt) {
        if (opt.has_value()) {
          opt.reset();
        } else {
          opt.emplace();
        }
      };

      const auto check_toggled =
          [&](const auto field, const bool geometric,
              [[maybe_unused]] const char* const message) {
            auto hist_copy = boundary_history;
            const auto side_copy = hist_copy.remote();
            toggle_optional(side_copy.data(side_copy.back()).*field);
            if constexpr (spectre_debug) {
              CHECK_THROWS_WITH(do_interpolate(side_copy, test_time, true),
                                Catch::Matchers::ContainsSubstring(message));
            }
            if (geometric) {
              CHECK(do_interpolate(side_copy, test_time, false) ==
                    test_interpolated_nogeom);
            } else if constexpr (spectre_debug) {
              CHECK_THROWS_WITH(do_interpolate(side_copy, test_time, false),
                                Catch::Matchers::ContainsSubstring(message));
            }
          };

      check_toggled(&evolution::dg::MortarData<Dim>::mortar_data, false,
                    "Some entries have mortar_data but not others");
      check_toggled(&evolution::dg::MortarData<Dim>::face_normal_magnitude,
                    true,
                    "Some entries have face_normal_magnitude but not others");
      check_toggled(&evolution::dg::MortarData<Dim>::face_det_jacobian, true,
                    "Some entries have face_det_jacobian but not others");
      check_toggled(&evolution::dg::MortarData<Dim>::volume_det_inv_jacobian,
                    true,
                    "Some entries have volume_det_inv_jacobian but not others");
      check_toggled(&evolution::dg::MortarData<Dim>::mortar_mesh, false,
                    "mortar_mesh does not match");
      check_toggled(&evolution::dg::MortarData<Dim>::face_mesh, true,
                    "face_mesh does not match");
      check_toggled(&evolution::dg::MortarData<Dim>::volume_mesh, true,
                    "volume_mesh does not match");

      const auto check_mismatch =
          [&](const auto field, const bool geometric,
              [[maybe_unused]] const char* const message) {
            // 0D meshes are all equal
            if ((side_hist.data(side_hist.back()).*field).has_value() and
                (side_hist.data(side_hist.back()).*field)->dim > 0) {
              auto hist_copy = boundary_history;
              const auto side_copy = hist_copy.remote();
              side_copy.data(side_copy.back()).*field =
                  make_data_to_overwrite<Dim>().*field;
              if constexpr (spectre_debug) {
                CHECK_THROWS_WITH(do_interpolate(side_copy, test_time, true),
                                  Catch::Matchers::ContainsSubstring(message));
              }
              if (geometric) {
                CHECK(do_interpolate(side_copy, test_time, false) ==
                      test_interpolated_nogeom);
              } else if constexpr (spectre_debug) {
                CHECK_THROWS_WITH(do_interpolate(side_copy, test_time, false),
                                  Catch::Matchers::ContainsSubstring(message));
              }
            }
          };

      check_mismatch(&evolution::dg::MortarData<Dim>::mortar_mesh, false,
                     "mortar_mesh does not match");
      check_mismatch(&evolution::dg::MortarData<Dim>::face_mesh, true,
                     "face_mesh does not match");
      check_mismatch(&evolution::dg::MortarData<Dim>::volume_mesh, true,
                     "volume_mesh does not match");
    }
  }

  // Test positioning the stencil
  //
  // First-order integration does nearest-neighbor
  {
    const auto fields_to_include = make_with_random_values<std::array<bool, 7>>(
        gen, make_not_null(&true_false));
    const std::array<RandomPolynomial, 4> polys{
        {{3, gen}, {3, gen}, {3, gen}, {3, gen}}};

    TimeSteppers::BoundaryHistory<evolution::dg::MortarData<Dim>,
                                  evolution::dg::MortarData<Dim>, DataVector>
        boundary_history{};
    const auto side_hist = boundary_history.remote();

    for (const auto& frac :
         {Rational{0}, Rational{1, 4}, Rational{3, 4}, Rational{1}}) {
      const TimeStepId id = make_id(frac);
      side_hist.insert(
          id, 1, make_data<Dim>(fields_to_include, polys, id.substep_time()));
    }

    const auto check_nearest = [&](const Rational& goal,
                                   const Rational& expected_nearest) {
      CAPTURE(goal);
      CHECK(do_interpolate(side_hist,
                           (slab.start() + slab.duration() * goal).value(),
                           true) == side_hist.data(make_id(expected_nearest)));
    };
    check_nearest(0, 0);
    check_nearest({1, 16}, 0);
    check_nearest({3, 16}, {1, 4});
    check_nearest({7, 16}, {1, 4});
    check_nearest({9, 16}, {3, 4});
    check_nearest(1, 1);
  }
  // Stencil positioning, higher-order case: interpolating to a time
  // near the end with extra data at the start was tested above, so
  // just test one case.
  {
    const size_t stepper_order = 4;
    const auto fields_to_include = make_with_random_values<std::array<bool, 7>>(
        gen, make_not_null(&true_false));
    const std::array<RandomPolynomial, 4> polys{{{stepper_order - 1, gen},
                                                 {stepper_order - 1, gen},
                                                 {stepper_order - 1, gen},
                                                 {stepper_order - 1, gen}}};

    TimeSteppers::BoundaryHistory<evolution::dg::MortarData<Dim>,
                                  evolution::dg::MortarData<Dim>, DataVector>
        boundary_history{};
    const auto side_hist = boundary_history.remote();

    for (size_t i = 0; i < stepper_order; ++i) {
      const TimeStepId id =
          make_id({static_cast<int32_t>(i + 8 - stepper_order / 2), 16});
      side_hist.insert(
          id, stepper_order,
          make_data<Dim>(fields_to_include, polys, id.substep_time()));
    }
    {
      const TimeStepId id = make_id(0);
      side_hist.insert_initial(id, stepper_order,
                               make_data_to_overwrite<Dim>());
    }
    {
      const TimeStepId id = make_id(1);
      side_hist.insert(id, stepper_order, make_data_to_overwrite<Dim>());
    }

    const double test_time = (slab.start() + slab.duration() * 17 / 32).value();
    const auto test_interpolated = do_interpolate(side_hist, test_time, true);
    const auto expected = make_data<Dim>(fields_to_include, polys, test_time);

    check_approx_opt(test_interpolated.mortar_data, expected.mortar_data);
    check_approx_opt(test_interpolated.face_normal_magnitude,
                     expected.face_normal_magnitude);
    check_approx_opt(test_interpolated.face_det_jacobian,
                     expected.face_det_jacobian);
    check_approx_opt(test_interpolated.volume_det_inv_jacobian,
                     expected.volume_det_inv_jacobian);
    CHECK(test_interpolated.mortar_mesh == expected.mortar_mesh);
    CHECK(test_interpolated.face_mesh == expected.face_mesh);
    CHECK(test_interpolated.volume_mesh == expected.volume_mesh);
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.TimeInterpolateMortarData",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  test<1>(&gen);
  test<2>(&gen);
  test<3>(&gen);
}
}  // namespace
