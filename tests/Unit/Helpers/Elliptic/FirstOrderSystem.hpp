// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Helper functions to test elliptic first-order systems

#pragma once

#include <cstddef>
#include <limits>
#include <random>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Elliptic/Systems/GetSourcesComputer.hpp"
#include "Framework/TestHelpers.hpp"
#include "Framework/TestingFramework.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "ParallelAlgorithms/NonlinearSolver/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace TestHelpers {
/// \ingroup TestingFrameworkGroup
/// Helper functions to test elliptic first-order systems
namespace elliptic {
namespace detail {

template <typename System, typename... PrimalFields, typename... PrimalFluxes,
          typename... FluxesArgsTags>
void test_first_order_fluxes_computer_impl(
    const DataVector& used_for_size, tmpl::list<PrimalFields...> /*meta*/,
    tmpl::list<PrimalFluxes...> /*meta*/,
    tmpl::list<FluxesArgsTags...> /*meta*/) {
  static constexpr size_t Dim = System::volume_dim;
  using FluxesComputer = typename System::fluxes_computer;
  using inv_metric_tag = typename System::inv_metric_tag;
  using vars_tag = ::Tags::Variables<tmpl::list<PrimalFields...>>;
  using VarsType = typename vars_tag::type;
  using deriv_vars_tag = db::add_tag_prefix<::Tags::deriv, vars_tag,
                                            tmpl::size_t<Dim>, Frame::Inertial>;
  using DerivVarsType = typename deriv_vars_tag::type;
  using fluxes_tag = ::Tags::Variables<tmpl::list<PrimalFluxes...>>;
  using FluxesType = typename fluxes_tag::type;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(0.5, 2.);

  // Generate random variables
  const auto vars = make_with_random_values<VarsType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto deriv_vars = make_with_random_values<DerivVarsType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);

  // Generate fluxes from the variables with random arguments
  tuples::TaggedTuple<FluxesArgsTags...> fluxes_args{
      make_with_random_values<typename FluxesArgsTags::type>(
          make_not_null(&generator), make_not_null(&dist), used_for_size)...};
  // Silence unused variable warning when fluxes args is empty
  (void)fluxes_args;
  FluxesType expected_fluxes{used_for_size.size()};
  FluxesComputer::apply(
      make_not_null(&get<PrimalFluxes>(expected_fluxes))...,
      get<FluxesArgsTags>(fluxes_args)..., get<PrimalFields>(vars)...,
      get<::Tags::deriv<PrimalFields, tmpl::size_t<Dim>, Frame::Inertial>>(
          deriv_vars)...);

  // Create a DataBox
  auto box = db::create<db::AddSimpleTags<vars_tag, deriv_vars_tag, fluxes_tag,
                                          FluxesArgsTags...>>(
      vars, deriv_vars,
      make_with_value<typename fluxes_tag::type>(
          used_for_size, std::numeric_limits<double>::signaling_NaN()),
      get<FluxesArgsTags>(fluxes_args)...);

  // Apply the fluxes computer to the DataBox
  db::mutate_apply<tmpl::list<PrimalFluxes...>,
                   tmpl::list<FluxesArgsTags..., PrimalFields...,
                              ::Tags::deriv<PrimalFields, tmpl::size_t<Dim>,
                                            Frame::Inertial>...>>(
      FluxesComputer{}, make_not_null(&box));
  CHECK(expected_fluxes == get<fluxes_tag>(box));

  {
    INFO("Fluxes computer on face");
    // Test that the two overloads are consistent: the one applied on the face
    // just has the face normal "baked in".
    const auto face_normal = make_with_random_values<tnsr::i<DataVector, Dim>>(
        make_not_null(&generator), make_not_null(&dist), used_for_size);
    tnsr::I<DataVector, Dim> face_normal_vector{used_for_size.size()};
    if constexpr (std::is_same_v<inv_metric_tag, void>) {
      for (size_t i = 0; i < Dim; ++i) {
        face_normal_vector.get(i) = face_normal.get(i);
      }
    } else {
      const auto& inv_metric = get<inv_metric_tag>(fluxes_args);
      raise_or_lower_index(make_not_null(&face_normal_vector), face_normal,
                           inv_metric);
    }
    DerivVarsType n_times_vars{used_for_size.size()};
    normal_times_flux(make_not_null(&n_times_vars), face_normal, vars);
    VarsType zero_vars{used_for_size.size(), 0.};
    FluxesComputer::apply(
        make_not_null(&get<PrimalFluxes>(expected_fluxes))...,
        get<FluxesArgsTags>(fluxes_args)..., get<PrimalFields>(zero_vars)...,
        get<::Tags::deriv<PrimalFields, tmpl::size_t<Dim>, Frame::Inertial>>(
            n_times_vars)...);
    FluxesType fluxes_on_face{used_for_size.size()};
    FluxesComputer::apply(make_not_null(&get<PrimalFluxes>(fluxes_on_face))...,
                          get<FluxesArgsTags>(fluxes_args)..., face_normal,
                          face_normal_vector, get<PrimalFields>(vars)...);
  CHECK_VARIABLES_APPROX(fluxes_on_face, expected_fluxes);
  }
}

}  // namespace detail

/*!
 * \brief Test the `System::fluxes_computer` is functional
 *
 * This function tests the following properties of the
 * `System::fluxes_computer`:
 *
 * - It works with the fields and fluxes specified in the `System`.
 * - It can be applied to a DataBox, i.e. its argument tags are consistent with
 *   its apply function.
 */
template <typename System>
void test_first_order_fluxes_computer(const DataVector& used_for_size) {
  detail::test_first_order_fluxes_computer_impl<System>(
      used_for_size, typename System::primal_fields{},
      typename System::primal_fluxes{},
      typename System::fluxes_computer::argument_tags{});
}

namespace detail {

template <typename SourcesComputer, typename... PrimalFields,
          typename... PrimalFluxes, typename... SourcesArgsTags>
void test_first_order_sources_computer_impl(
    const DataVector& used_for_size, tmpl::list<PrimalFields...> /*meta*/,
    tmpl::list<PrimalFluxes...> /*meta*/,
    tmpl::list<SourcesArgsTags...> /*meta*/) {
  using vars_tag = ::Tags::Variables<tmpl::list<PrimalFields...>>;
  using VarsType = typename vars_tag::type;
  using fluxes_tag = ::Tags::Variables<tmpl::list<PrimalFluxes...>>;
  using FluxesType = typename fluxes_tag::type;
  using sources_tag = db::add_tag_prefix<::Tags::Source, vars_tag>;
  using SourcesType = typename sources_tag::type;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(0.5, 2.);

  // Generate random variables and fluxes. Note that the sources make use of the
  // pre-computed fluxes as an optimization (see elliptic::first_order_sources),
  // but for the purpose of this random-value test the fluxes don't have to be
  // computed from the variables.
  const auto vars = make_with_random_values<VarsType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);
  const auto fluxes = make_with_random_values<FluxesType>(
      make_not_null(&generator), make_not_null(&dist), used_for_size);

  // Generate sources from the variables with random arguments
  tuples::TaggedTuple<SourcesArgsTags...> sources_args{
      make_with_random_values<typename SourcesArgsTags::type>(
          make_not_null(&generator), make_not_null(&dist), used_for_size)...};
  // Silence unused variable warning when sources args is empty
  (void)sources_args;
  SourcesType expected_sources{used_for_size.size(), 0.};
  SourcesComputer::apply(
      make_not_null(&get<::Tags::Source<PrimalFields>>(expected_sources))...,
      get<SourcesArgsTags>(sources_args)..., get<PrimalFields>(vars)...,
      get<PrimalFluxes>(fluxes)...);

  // Create a DataBox
  auto box = db::create<
      db::AddSimpleTags<vars_tag, fluxes_tag, sources_tag, SourcesArgsTags...>>(
      vars, fluxes,
      make_with_value<typename sources_tag::type>(used_for_size, 0.),
      get<SourcesArgsTags>(sources_args)...);

  // Apply the sources computer to the DataBox
  db::mutate_apply<
      tmpl::list<::Tags::Source<PrimalFields>...>,
      tmpl::list<SourcesArgsTags..., PrimalFields..., PrimalFluxes...>>(
      SourcesComputer{}, make_not_null(&box));
  CHECK(expected_sources == get<sources_tag>(box));
}

}  // namespace detail

/*!
 * \brief Test the `System::sources_computer` is functional
 *
 * This function tests the following properties of the
 * `System::sources_computer`:
 *
 * - It works with the fields and fluxes specified in the `System`.
 * - It can be applied to a DataBox, i.e. its argument tags are consistent with
 *   its apply function.
 */
template <typename System, bool Linearized = false>
void test_first_order_sources_computer(const DataVector& used_for_size) {
  using sources_computer = ::elliptic::get_sources_computer<System, Linearized>;
  if constexpr (not std::is_same_v<sources_computer, void>) {
    using primal_fields =
        tmpl::conditional_t<Linearized,
                            db::wrap_tags_in<NonlinearSolver::Tags::Correction,
                                             typename System::primal_fields>,
                            typename System::primal_fields>;
    using primal_fluxes =
        tmpl::conditional_t<Linearized,
                            db::wrap_tags_in<NonlinearSolver::Tags::Correction,
                                             typename System::primal_fluxes>,
                            typename System::primal_fluxes>;
    detail::test_first_order_sources_computer_impl<sources_computer>(
        used_for_size, primal_fields{}, primal_fluxes{},
        typename sources_computer::argument_tags{});
  }
}

}  // namespace elliptic
}  // namespace TestHelpers
