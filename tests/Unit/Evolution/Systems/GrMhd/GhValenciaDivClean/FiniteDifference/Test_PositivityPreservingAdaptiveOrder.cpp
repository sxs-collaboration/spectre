// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <pup.h>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Factory.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/PositivityPreservingAdaptiveOrder.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/ReconstructWork.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Tag.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/System.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "Evolution/VariableFixing/FixToAtmosphere.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/PrimReconstructor.hpp"
#include "NumericalAlgorithms/FiniteDifference/FallbackReconstructorType.hpp"
#include "NumericalAlgorithms/Interpolation/LagrangePolynomial.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
template <typename Tag>
struct value_tag;

template <template <typename, size_t, typename...> typename Tag>
struct value_tag<Tag<DataVector, 3>> {
  using type = Tag<double, 3>;
};

template <typename Tag>
using value_tag_t = typename value_tag<Tag>::type;

template <typename Reconstructor>
void test_neighbor_positivity(const Reconstructor& reconstructor) {
  using positive_tags = tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                                   hydro::Tags::ElectronFraction<DataVector>,
                                   hydro::Tags::Temperature<DataVector>>;
  using non_positive_tags =
      tmpl::list<hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 hydro::Tags::DivergenceCleaningField<DataVector>>;
  using spacetime_tags =
      ::grmhd::GhValenciaDivClean::Tags::spacetime_reconstruction_tags;

  DirectionMap<3, Neighbors<3>> element_neighbors{};
  element_neighbors[Direction<3>::upper_xi()] =
      Neighbors<3>{{ElementId<3>{1, {}}}, OrientationMap<3>::create_aligned()};
  element_neighbors[Direction<3>::lower_xi()] =
      Neighbors<3>{{ElementId<3>{2, {}}}, OrientationMap<3>::create_aligned()};
  const Element<3> element{ElementId<3>{0, {}}, std::move(element_neighbors)};

  const Mesh<3> subcell_mesh{{{11, 1, 1}},
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};

  tmpl::wrap<tmpl::transform<spacetime_tags, value_tag<tmpl::_1>>,
             tuples::TaggedTuple>
      spacetime_values{};
  {
    auto& spacetime_metric =
        get<gr::Tags::SpacetimeMetric<double, 3>>(spacetime_values);
    std::fill(spacetime_metric.begin(), spacetime_metric.end(), 0.0);
    get<0, 0>(spacetime_metric) = -1.0;
    for (size_t i = 1; i < 4; ++i) {
      spacetime_metric.get(i, i) = 1.0;
    }
    auto& pi = get<gh::Tags::Pi<double, 3>>(spacetime_values);
    std::fill(pi.begin(), pi.end(), 0.0);
    auto& phi = get<gh::Tags::Phi<double, 3>>(spacetime_values);
    std::fill(phi.begin(), phi.end(), 0.0);
  }

  Variables<spacetime_tags> volume_spacetime_vars{
      subcell_mesh.number_of_grid_points()};
  tmpl::for_each<spacetime_tags>([&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
    const auto& value = get<value_tag_t<Tag>>(spacetime_values);
    std::copy(value.begin(), value.end(),
              get<Tag>(volume_spacetime_vars).begin());
  });

  const double atmosphere = 1.0e-15;
  const Variables<hydro::grmhd_tags<DataVector>> volume_prims{
      subcell_mesh.number_of_grid_points(), atmosphere};

  const auto ghost_zone_size = reconstructor.ghost_zone_size();
  std::vector<double> ghost_values{};
  ghost_values.reserve(ghost_zone_size);
  ASSERT(ghost_zone_size > 1, "Not supported by test");
  {
    // Construct ghost data so polynomial interpolation will give
    // -3*atmosphere at the boundary.  We choose the
    // 2*ghost_zone_size-1 points on a polynomial defined by
    // 2*ghost_zone_size-2 control points, as this will trick the
    // Persson TCI into thinking the solution is smooth.
    std::vector<double> points{};
    points.reserve(2 * ghost_zone_size - 2);
    std::vector<double> values{};
    values.reserve(2 * ghost_zone_size - 2);
    points.push_back(-0.5);
    values.push_back(-3.0 * atmosphere);

    for (size_t i = 1; i < ghost_zone_size; ++i) {
      points.push_back(-static_cast<double>(i));
      values.push_back(atmosphere);
    }

    for (size_t i = 0; i < ghost_zone_size - 2; ++i) {
      points.push_back(static_cast<double>(i));
      values.push_back(atmosphere);
      ghost_values.push_back(atmosphere);
    }

    for (const double ghost_x : {static_cast<double>(ghost_zone_size - 2),
                                 static_cast<double>(ghost_zone_size - 1)}) {
      double ghost_value = 0.0;
      for (size_t i = 0; i < points.size(); ++i) {
        ghost_value +=
            values[i] *
            lagrange_polynomial(i, ghost_x, points.begin(), points.end());
      }
      ghost_values.push_back(ghost_value);
    }
  }

  using GhostData = evolution::dg::subcell::GhostData;
  DirectionalIdMap<3, GhostData> ghost_data{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    const DirectionalId<3> mortar_id{direction, *neighbors.begin()};
    Variables<::grmhd::GhValenciaDivClean::Tags::
                  primitive_grmhd_and_spacetime_reconstruction_tags>
        ghost_vars{};
    auto& mortar_data = ghost_data[mortar_id];
    mortar_data = GhostData{1};
    mortar_data.neighbor_ghost_data_for_reconstruction() = DataVector{
        ghost_vars.number_of_independent_components * ghost_zone_size};
    ghost_vars.set_data_ref(
        mortar_data.neighbor_ghost_data_for_reconstruction().data(),
        mortar_data.neighbor_ghost_data_for_reconstruction().size());
    DataVector ghost_component{ghost_zone_size};
    if (direction.side() == Side::Upper) {
      std::copy(ghost_values.begin(), ghost_values.end(),
                ghost_component.begin());
    } else {
      std::copy(ghost_values.rbegin(), ghost_values.rend(),
                ghost_component.begin());
    }
    tmpl::for_each<tmpl::append<positive_tags, non_positive_tags>>(
        [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          auto& tensor = get<Tag>(ghost_vars);
          std::fill(tensor.begin(), tensor.end(), ghost_component);
        });
    tmpl::for_each<spacetime_tags>(
        [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          const auto& value = get<value_tag_t<Tag>>(spacetime_values);
          std::copy(value.begin(), value.end(), get<Tag>(ghost_vars).begin());
        });
  }

  const EquationsOfState::IdealFluid<true> eos{1.4};
  const VariableFixing::FixToAtmosphere<3> fix_to_atmosphere{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    CAPTURE(direction);
    Variables<
        ::grmhd::GhValenciaDivClean::fd::tags_list_for_reconstruct_fd_neighbor>
        vars_on_mortar{1};
    reconstructor.reconstruct_fd_neighbor(
        make_not_null(&vars_on_mortar), volume_prims, volume_spacetime_vars,
        eos, element, ghost_data, subcell_mesh, fix_to_atmosphere, direction);

    tmpl::for_each<positive_tags>([&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
      CAPTURE(pretty_type::name<Tag>());
      CHECK(get(get<Tag>(vars_on_mortar))[0] > 0.0);
    });

    // Check that the test is actually testing something.
    tmpl::for_each<non_positive_tags>(
        [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          CAPTURE(pretty_type::name<Tag>());
          for (const auto& component : get<Tag>(vars_on_mortar)) {
            CHECK(component[0] < 0.0);
          }
        });
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.GrMhd.GhValenciaDivClean.Fd.Ppao",
                  "[Unit][Evolution]") {
  using NeutrinoTransportSystem = RadiationTransport::NoNeutrinos::System;
  using System = grmhd::GhValenciaDivClean::System<NeutrinoTransportSystem>;

  namespace helpers = TestHelpers::grmhd::GhValenciaDivClean::fd;
  PUPable_reg(SINGLE_ARG(
      grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
          System>));
  const auto ppao_from_options_base = TestHelpers::test_factory_creation<
      grmhd::GhValenciaDivClean::fd::Reconstructor<System>,
      grmhd::GhValenciaDivClean::fd::OptionTags::Reconstructor<System>>(
      "PositivityPreservingAdaptiveOrderPrim:\n"
      "  Alpha5: 3.7\n"
      "  Alpha7: None\n"
      "  Alpha9: None\n"
      "  LowOrderReconstructor: MonotonisedCentral\n"
      "  AtmosphereTreatment: Never\n"
      "  ReconstructRhoTimesTemperature: true\n");
  const auto ppao_deserialized =
      serialize_and_deserialize(ppao_from_options_base);
  auto* const ppao_from_options = dynamic_cast<
      const grmhd::GhValenciaDivClean::fd::
          PositivityPreservingAdaptiveOrderPrim<System>*>(
      ppao_deserialized.get());
  REQUIRE(ppao_from_options != nullptr);
  CHECK(grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
            System>{
            3.7, std::nullopt, std::nullopt,
            fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
            ::VariableFixing::FixReconstructedStateToAtmosphere::Always,
            false} !=
        grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
            System>{
            3.7, std::nullopt, std::nullopt,
            fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
            ::VariableFixing::FixReconstructedStateToAtmosphere::Never, false});
  CHECK(
      grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
          System>{
          3.7, std::nullopt, std::nullopt,
          fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
          ::VariableFixing::FixReconstructedStateToAtmosphere::Always, false} !=
      grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
          System>{
          3.8, std::nullopt, std::nullopt,
          fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
          ::VariableFixing::FixReconstructedStateToAtmosphere::Always, false});
  // Can't use high-order reconstruction yet. We'll enable these tests later.
  //
  // CHECK(
  //     grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
  //         System>{
  //         3.7, std::nullopt, std::nullopt,
  //         fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
  //         ::VariableFixing::FixReconstructedStateToAtmosphere::Always, false}
  //         !=
  //     grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
  //         System>{
  //         3.7, 3.5, std::nullopt,
  //         fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
  //         ::VariableFixing::FixReconstructedStateToAtmosphere::Always,
  //         false});
  // CHECK(
  //     grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
  //         System>{
  //         3.7, std::nullopt, std::nullopt,
  //         fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
  //         ::VariableFixing::FixReconstructedStateToAtmosphere::Always, false}
  //         !=
  //     grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
  //         System>{
  //         3.7, std::nullopt, 3.6,
  //         fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
  //         ::VariableFixing::FixReconstructedStateToAtmosphere::Always,
  //         false});
  CHECK(grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
            System>{
            3.7, std::nullopt, std::nullopt,
            fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
            ::VariableFixing::FixReconstructedStateToAtmosphere::Always,
            false} !=
        grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
            System>{
            3.7, std::nullopt, std::nullopt,
            fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
            ::VariableFixing::FixReconstructedStateToAtmosphere::Always, true});
  CHECK(*ppao_from_options ==
        grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
            System>{
            3.7, std::nullopt, std::nullopt,
            fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
            ::VariableFixing::FixReconstructedStateToAtmosphere::Never, true});
  test_move_semantics(
      grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
          System>{
          3.7, std::nullopt, std::nullopt,
          fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
          ::VariableFixing::FixReconstructedStateToAtmosphere::Never, true},
      grmhd::GhValenciaDivClean::fd::PositivityPreservingAdaptiveOrderPrim<
          System>{
          3.7, std::nullopt, std::nullopt,
          fd::reconstruction::FallbackReconstructorType::MonotonisedCentral,
          ::VariableFixing::FixReconstructedStateToAtmosphere::Never, true});
  helpers::test_prim_reconstructor(10, *ppao_from_options);

  test_neighbor_positivity(*ppao_from_options);
}
}  // namespace
