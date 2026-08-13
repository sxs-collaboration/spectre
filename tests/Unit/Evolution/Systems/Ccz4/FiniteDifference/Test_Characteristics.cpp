// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Characteristics.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/Derivatives.hpp"
#include "Evolution/Systems/Ccz4/FiniteDifference/System.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/Systems/Ccz4/CharacteristicsTestHelpers.hpp"
#include "Helpers/Evolution/Systems/Ccz4/PrimReconstructor.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"

namespace {
constexpr size_t Dim = Ccz4::fd::System::volume_dim;

void test_characteristics(
    const size_t points_per_dimension,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_one_form) {
  ASSERT(pow<3>(points_per_dimension) == get<0>(normal_one_form).size(),
         "The size of the unit normal one form must match the number of grid "
         "points per dimension.");

  const size_t ghost_zone_size = 3;
  const size_t fd_deriv_order = 4;
  const Mesh<Dim> subcell_mesh{points_per_dimension,
                               Spectral::Basis::FiniteDifference,
                               Spectral::Quadrature::CellCentered};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

  const std::array<double, Dim> lower_bound{5.8, 5.0, 2.3};
  const std::array<double, Dim> upper_bound{6.2, 5.2, 2.4};
  const std::array<double, Dim> coords_range = upper_bound - lower_bound;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1., 1., lower_bound[0], upper_bound[0]},
              Affine{-1., 1., lower_bound[1], upper_bound[1]},
              Affine{-1., 1., lower_bound[2], upper_bound[2]},
          });
  // set up displaced logical coords
  const auto logical_coords =
      TestHelpers::Ccz4::fd::detail::set_logical_coordinates(subcell_mesh);
  const auto x = coord_map(logical_coords);
  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      cell_centered_logical_to_inertial_inv_jacobian{
          subcell_mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < Dim; ++i) {
    cell_centered_logical_to_inertial_inv_jacobian.get(i, i) =
        2.0 / gsl::at(coords_range, i);
  }

  const Element<Dim> element = TestHelpers::Ccz4::fd::detail::set_element();

  // Setup solution
  const double mass = 2.0;
  const std::array<double, Dim> spin{{0.2, 0.4, 0.8}};
  const std::array<double, Dim> center{{0.2, 0.5, 0.1}};
  const gr::Solutions::KerrSchild solution(mass, spin, center);

  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

  const double f = Ccz4::fd::System::f;
  const bool evolve_shift = true;
  const DirectionalIdMap<Dim, evolution::dg::subcell::GhostData>
      all_ghost_data =
          TestHelpers::Ccz4::fd::detail::compute_ghost_data<Frame::Inertial>(
              subcell_mesh, x, element.neighbors(), ghost_zone_size,
              TestHelpers::Ccz4::fd::detail::KerrSchild::
                  compute_prim_solution_for_KerrSchild,
              coords_range, t, f, evolve_shift, solution);

  auto volume_evolved_vars = TestHelpers::Ccz4::fd::detail::KerrSchild::
      compute_prim_solution_for_KerrSchild(x, t, f, evolve_shift, solution);

  Variables<
      db::wrap_tags_in<Tags::deriv, typename Ccz4::fd::System::gradients_tags,
                       tmpl::size_t<Dim>, Frame::Inertial>>
      deriv_of_Ccz4_vars{subcell_mesh.number_of_grid_points()};

  ::Ccz4::fd::spacetime_derivatives(
      make_not_null(&deriv_of_Ccz4_vars), volume_evolved_vars, all_ghost_data,
      fd_deriv_order, subcell_mesh,
      cell_centered_logical_to_inertial_inv_jacobian);

  // Compute normalized one-form
  const auto& conformal_spatial_metric =
      get<::Ccz4::Tags::ConformalMetric<DataVector, Dim>>(volume_evolved_vars);
  const auto& conformal_factor =
      get<::Ccz4::Tags::ConformalFactor<DataVector>>(volume_evolved_vars);
  tnsr::ii<DataVector, Dim, Frame::Inertial> spatial_metric{};
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&spatial_metric),
      conformal_spatial_metric(ti::i, ti::j) /
          (conformal_factor() * conformal_factor()));
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  tnsr::i<DataVector, Dim, Frame::Inertial> unit_normal_one_form{};
  DataVector magnitude =
    sqrt(get(::tenex::evaluate(inverse_spatial_metric(ti::I, ti::J) *
                          normal_one_form(ti::i) * normal_one_form(ti::j))));
  for (size_t i = 0; i < Dim; ++i) {
    unit_normal_one_form.get(i) = normal_one_form.get(i) / magnitude;
  }

  // Test characteristic speeds by coding twice
  const auto char_speeds = Ccz4::fd::characteristic_speeds<Frame::Inertial>(
      get<gr::Tags::Lapse<DataVector>>(volume_evolved_vars),
      get<gr::Tags::Shift<DataVector, Dim>>(volume_evolved_vars),
      get<::Ccz4::Tags::ConformalFactor<DataVector>>(volume_evolved_vars), f,
      unit_normal_one_form);
  const auto expected_char_speeds =
      TestHelpers::Ccz4::fd::detail::compute_expected_characteristic_speeds(
          get<gr::Tags::Lapse<DataVector>>(volume_evolved_vars),
          get<gr::Tags::Shift<DataVector, Dim>>(volume_evolved_vars),
          get<::Ccz4::Tags::ConformalFactor<DataVector>>(volume_evolved_vars),
          f, unit_normal_one_form);
  for (size_t i = 0; i < char_speeds.size(); ++i) {
    CHECK_ITERABLE_APPROX(char_speeds.at(i), expected_char_speeds.at(i));
  }

  // Test characteristic transformation by coding twice
  ::Ccz4::fd::Tags::CharacteristicFields<DataVector, Dim, Frame::Inertial>::type
      char_fields{};
  ::Ccz4::fd::CharacteristicFieldsCompute<Frame::Inertial>::function(
      make_not_null(&char_fields), unit_normal_one_form,
      get<::Ccz4::Tags::ConformalMetric<DataVector, Dim>>(volume_evolved_vars),
      get<::Ccz4::Tags::ConformalFactor<DataVector>>(volume_evolved_vars),
      get<gr::Tags::Lapse<DataVector>>(volume_evolved_vars),
      get<gr::Tags::Shift<DataVector, Dim>>(volume_evolved_vars),
      get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(volume_evolved_vars),
      get<::Ccz4::Tags::ATilde<DataVector, Dim>>(volume_evolved_vars),
      get<::Ccz4::Tags::Theta<DataVector>>(volume_evolved_vars),
      get<::Ccz4::Tags::GammaHat<DataVector, Dim>>(volume_evolved_vars),
      get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim>>(volume_evolved_vars),
      get<Tags::deriv<::Ccz4::Tags::ConformalMetric<DataVector, Dim>,
                      tmpl::size_t<Dim>, Frame::Inertial>>(deriv_of_Ccz4_vars),
      get<Tags::deriv<::Ccz4::Tags::ConformalFactor<DataVector>,
                      tmpl::size_t<Dim>, Frame::Inertial>>(deriv_of_Ccz4_vars),
      get<Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(deriv_of_Ccz4_vars),
      get<Tags::deriv<gr::Tags::Shift<DataVector, Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(deriv_of_Ccz4_vars));
  const auto expected_char_fields =
      TestHelpers::Ccz4::fd::detail::compute_expected_characteristic_fields(
          unit_normal_one_form,
          get<::Ccz4::Tags::ConformalMetric<DataVector, Dim>>(
              volume_evolved_vars),
          get<::Ccz4::Tags::ConformalFactor<DataVector>>(volume_evolved_vars),
          get<gr::Tags::Lapse<DataVector>>(volume_evolved_vars),
          get<gr::Tags::Shift<DataVector, Dim>>(volume_evolved_vars),
          get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(
              volume_evolved_vars),
          get<::Ccz4::Tags::ATilde<DataVector, Dim>>(volume_evolved_vars),
          get<::Ccz4::Tags::Theta<DataVector>>(volume_evolved_vars),
          get<::Ccz4::Tags::GammaHat<DataVector, Dim>>(volume_evolved_vars),
          get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim>>(
              volume_evolved_vars),
          get<Tags::deriv<::Ccz4::Tags::ConformalMetric<DataVector, Dim>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(
              deriv_of_Ccz4_vars),
          get<Tags::deriv<::Ccz4::Tags::ConformalFactor<DataVector>,
                          tmpl::size_t<Dim>, Frame::Inertial>>(
              deriv_of_Ccz4_vars),
          get<Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                          Frame::Inertial>>(deriv_of_Ccz4_vars),
          get<Tags::deriv<gr::Tags::Shift<DataVector, Dim>, tmpl::size_t<Dim>,
                          Frame::Inertial>>(deriv_of_Ccz4_vars),
          f);
  tmpl::for_each<typename ::Ccz4::fd::Tags::CharacteristicFields<
      DataVector, Dim, Frame::Inertial>::type::tags_list>(
      [&char_fields,
       &expected_char_fields]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        const std::string tag_name = db::tag_name<Tag>();
        CAPTURE(tag_name);
        double max_value = 0.0;
        const auto& tensor = get<Tag>(char_fields);
        for (size_t i = 0; i < tensor.size(); ++i) {
          max_value = std::max(max_value, max(abs(tensor[i])));
        }
        const Approx custom_approx =
            Approx::custom().epsilon(1.0e-12).scale(max_value);
        CHECK_ITERABLE_CUSTOM_APPROX(get<Tag>(char_fields),
                                     get<Tag>(expected_char_fields),
                                     custom_approx);
      });

  // Test that inverse transformation recovers the original variables
  typename ::Ccz4::fd::Tags::EvolvedSpaceFromCharacteristicFields<
      DataVector, Dim, Frame::Inertial>::type recovered_vars{};
  ::Ccz4::fd::EvolvedSpaceFromCharacteristicFieldsCompute<Frame::Inertial>::
      function(
          make_not_null(&recovered_vars),
          get<::Ccz4::fd::Tags::UTensorPlus<DataVector, Dim, Frame::Inertial>>(
              char_fields),
          get<::Ccz4::fd::Tags::UTensorMinus<DataVector, Dim, Frame::Inertial>>(
              char_fields),
          get<::Ccz4::fd::Tags::UVector1Zero<DataVector, Dim, Frame::Inertial>>(
              char_fields),
          get<::Ccz4::fd::Tags::UVector2Plus<DataVector, Dim, Frame::Inertial>>(
              char_fields),
          get<::Ccz4::fd::Tags::UVector2Minus<DataVector, Dim,
                                              Frame::Inertial>>(char_fields),
          get<::Ccz4::fd::Tags::UVector3Plus<DataVector, Dim, Frame::Inertial>>(
              char_fields),
          get<::Ccz4::fd::Tags::UVector3Minus<DataVector, Dim,
                                              Frame::Inertial>>(char_fields),
          get<::Ccz4::fd::Tags::UScalar1Zero<DataVector>>(char_fields),
          get<::Ccz4::fd::Tags::UScalar2Plus<DataVector>>(char_fields),
          get<::Ccz4::fd::Tags::UScalar2Minus<DataVector>>(char_fields),
          get<::Ccz4::fd::Tags::UScalar3Plus<DataVector>>(char_fields),
          get<::Ccz4::fd::Tags::UScalar3Minus<DataVector>>(char_fields),
          get<::Ccz4::fd::Tags::UScalar4Plus<DataVector>>(char_fields),
          get<::Ccz4::fd::Tags::UScalar4Minus<DataVector>>(char_fields),
          get<::Ccz4::fd::Tags::UScalar5Plus<DataVector>>(char_fields),
          get<::Ccz4::fd::Tags::UScalar5Minus<DataVector>>(char_fields),
          unit_normal_one_form,
          get<::Ccz4::Tags::ConformalMetric<DataVector, Dim>>(
              volume_evolved_vars),
          get<::Ccz4::Tags::ConformalFactor<DataVector>>(volume_evolved_vars),
          get<gr::Tags::Lapse<DataVector>>(volume_evolved_vars),
          get<gr::Tags::Shift<DataVector, Dim>>(volume_evolved_vars));

  const Approx custom_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Ccz4::Tags::ATilde<DataVector, Dim>>(recovered_vars)),
      (get<::Ccz4::Tags::ATilde<DataVector, Dim>>(volume_evolved_vars)),
      custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(recovered_vars)),
      (get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(volume_evolved_vars)),
      custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Ccz4::Tags::Theta<DataVector>>(recovered_vars)),
      (get<::Ccz4::Tags::Theta<DataVector>>(volume_evolved_vars)),
      custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Ccz4::Tags::GammaHat<DataVector, Dim>>(recovered_vars)),
      (get<::Ccz4::Tags::GammaHat<DataVector, Dim>>(volume_evolved_vars)),
      custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim>>(recovered_vars)),
      (get<::Ccz4::Tags::AuxiliaryShiftB<DataVector, Dim>>(
          volume_evolved_vars)),
      custom_approx);

  const tnsr::I<DataVector, Dim, Frame::Inertial> unit_normal_vector =
      raise_or_lower_index(unit_normal_one_form, inverse_spatial_metric);
  magnitude = sqrt(get(dot_product(unit_normal_one_form, unit_normal_vector)));
  CHECK_ITERABLE_APPROX(magnitude, DataVector(magnitude.size(), 1.0));

  const auto& d_conformal_spatial_metric =
      get<Tags::deriv<::Ccz4::Tags::ConformalMetric<DataVector, Dim>,
                      tmpl::size_t<Dim>, Frame::Inertial>>(deriv_of_Ccz4_vars);
  tnsr::ii<DataVector, Dim, Frame::Inertial>
      original_dn_conformal_spatial_metric{};
  ::tenex::evaluate<ti::i, ti::j>(
      make_not_null(&original_dn_conformal_spatial_metric),
      unit_normal_vector(ti::K) *
          d_conformal_spatial_metric(ti::k, ti::i, ti::j));
  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Ccz4::fd::Tags::DnConformalMetric<
           DataVector, Dim, Frame::Inertial>>(recovered_vars)),
      original_dn_conformal_spatial_metric, custom_approx);

  const auto& d_conformal_factor =
      get<Tags::deriv<::Ccz4::Tags::ConformalFactor<DataVector>,
                      tmpl::size_t<Dim>, Frame::Inertial>>(deriv_of_Ccz4_vars);
  Scalar<DataVector> original_dn_conformal_factor{};
  ::tenex::evaluate(make_not_null(&original_dn_conformal_factor),
                    unit_normal_vector(ti::I) * d_conformal_factor(ti::i));
  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Ccz4::fd::Tags::DnConformalFactor<DataVector>>(recovered_vars)),
      original_dn_conformal_factor, custom_approx);

  const auto& d_lapse =
      get<Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(deriv_of_Ccz4_vars);
  Scalar<DataVector> original_dn_lapse{};
  ::tenex::evaluate(make_not_null(&original_dn_lapse),
                    unit_normal_vector(ti::I) * d_lapse(ti::i));
  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Ccz4::fd::Tags::DnLapse<DataVector>>(recovered_vars)),
      original_dn_lapse, custom_approx);

  const auto& d_shift =
      get<Tags::deriv<gr::Tags::Shift<DataVector, Dim>, tmpl::size_t<Dim>,
                      Frame::Inertial>>(deriv_of_Ccz4_vars);
  tnsr::I<DataVector, Dim, Frame::Inertial> original_dn_shift{};
  ::tenex::evaluate<ti::I>(make_not_null(&original_dn_shift),
                           unit_normal_vector(ti::J) * d_shift(ti::j, ti::I));
  CHECK_ITERABLE_CUSTOM_APPROX(
      (get<::Ccz4::fd::Tags::DnShift<DataVector, Dim, Frame::Inertial>>(
          recovered_vars)),
      original_dn_shift, custom_approx);
}
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.Evolution.Systems.Ccz4.Characteristics",
                  "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  const std::uniform_real_distribution<> distribution(1.0, 2.0);
  const size_t points_per_dimension = 5;
  auto normal_one_form =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          make_not_null(&generator), distribution,
          DataVector(pow<3>(points_per_dimension),
                     std::numeric_limits<double>::signaling_NaN()));
  test_characteristics(points_per_dimension, normal_one_form);
}
