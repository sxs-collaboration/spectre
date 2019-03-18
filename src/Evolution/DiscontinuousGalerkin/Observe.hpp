// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <initializer_list>
#include <pup.h>
#include <string>
#include <type_traits>  // IWYU pragma: keep
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"  // IWYU pragma: keep
#include "Domain/Tags.hpp"
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"  // IWYU pragma: keep
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/ReductionActions.hpp"  // IWYU pragma: keep
#include "IO/Observer/VolumeActions.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Time.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
template <size_t Dim>
class Mesh;
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace dg {
namespace Events {
template <size_t VolumeDim, typename Tensors, typename AnalyticSolutionTensors,
          typename EventRegistrars,
          typename NonSolutionTensors =
              tmpl::list_difference<Tensors, AnalyticSolutionTensors>>
class Observe;

namespace Registrars {
template <size_t VolumeDim, typename Tensors,
          typename AnalyticSolutionTensors = tmpl::list<>>
struct Observe {
  template <typename RegistrarList>
  using f = Events::Observe<VolumeDim, Tensors, AnalyticSolutionTensors,
                            RegistrarList>;
};
}  // namespace Registrars

template <size_t VolumeDim, typename Tensors,
          typename AnalyticSolutionTensors = tmpl::list<>,
          typename EventRegistrars = tmpl::list<
              Registrars::Observe<VolumeDim, Tensors, AnalyticSolutionTensors>>,
          typename NonSolutionTensors>
class Observe;  // IWYU pragma: keep

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief %Observe the fields in a system
 *
 * Writes volume quantities:
 * - `InertialCoordinates`
 * - Tensors listed in `Tensors` template parameter
 * - `Error*` = errors in `AnalyticSolutionTensors` =
 *   \f$\text{value} - \text{analytic solution}\f$
 *
 * Writes reduction quantities:
 * - `Time`
 * - `NumberOfPoints` = total number of points in the domain
 * - `Error*` = errors in `AnalyticSolutionTensors` =
 *   \f$\operatorname{RMS}\left(\sqrt{\sum_{\text{independent components}}\left[
 *   \text{value} - \text{analytic solution}\right]^2}\right)\f$
 *   over all points
 *
 * \warning Currently, only one observation event can be triggered at
 * a given time.  Causing multiple events to run at once will produce
 * unpredictable results.
 */
template <size_t VolumeDim, typename... Tensors,
          typename... AnalyticSolutionTensors, typename EventRegistrars,
          typename... NonSolutionTensors>
class Observe<VolumeDim, tmpl::list<Tensors...>,
              tmpl::list<AnalyticSolutionTensors...>, EventRegistrars,
              tmpl::list<NonSolutionTensors...>>
    : public Event<EventRegistrars> {
 private:
  static_assert(
      cpp17::is_same_v<
          tmpl::list_difference<tmpl::list<AnalyticSolutionTensors...>,
                                tmpl::list<Tensors...>>,
          tmpl::list<>>,
      "All AnalyticSolutionTensors must be listed in Tensors.");
  using coordinates_tag = ::Tags::Coordinates<VolumeDim, Frame::Inertial>;

  template <typename Tag>
  struct LocalSquareError {
    using type = double;
  };

  using L2ErrorDatum = Parallel::ReductionDatum<double, funcl::Plus<>,
                                                funcl::Sqrt<funcl::Divides<>>,
                                                std::index_sequence<1>>;
  using ReductionData = tmpl::wrap<
      tmpl::append<
          tmpl::list<Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
                     Parallel::ReductionDatum<size_t, funcl::Plus<>>>,
          tmpl::filled_list<L2ErrorDatum, sizeof...(AnalyticSolutionTensors)>>,
      Parallel::ReductionData>;

  template <typename T>
  static std::string component_suffix(const T& tensor,
                                      size_t component_index) noexcept {
    return tensor.rank() == 0
               ? ""
               : "_" + tensor.component_name(
                           tensor.get_tensor_index(component_index));
  }

 public:
  /// \cond
  explicit Observe(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Observe);  // NOLINT
  /// \endcond

  struct VariablesToObserve {
    static constexpr OptionString help =
        "Subset of system variables to observe in the volume";
    using type = std::vector<std::string>;
    static type default_value() noexcept { return {Tensors::name()...}; }
    static size_t lower_bound_on_size() noexcept { return 1; }
  };

  using options = tmpl::list<VariablesToObserve>;
  static constexpr OptionString help =
      "Observe the fields in a system.\n"
      "\n"
      "Writes volume quantities:\n"
      " * InertialCoordinates\n"
      " * Tensors listed in Tensors template parameter\n"
      " * Error* = errors in AnalyticSolutionTensors\n"
      "          = value - analytic solution\n"
      "\n"
      "Writes reduction quantities:\n"
      " * Time\n"
      " * NumberOfPoints = total number of points in the domain\n"
      " * Error* = errors in AnalyticSolutionTensors\n"
      "            (see online help for exact definition)\n"
      "\n"
      "Warning: Currently, only one observation event can be triggered at\n"
      "a given time.  Causing multiple events to run at once will produce\n"
      "unpredictable results.";

  explicit Observe(const std::vector<std::string>& variables_to_observe =
                       VariablesToObserve::default_value(),
                   const OptionContext& context = {})
      : variables_to_observe_(variables_to_observe.begin(),
                              variables_to_observe.end()) {
    const std::unordered_set<std::string> valid_tensors{Tensors::name()...};
    for (const auto& name : variables_to_observe_) {
      if (valid_tensors.count(name) != 1) {
        PARSE_ERROR(
            context,
            name << " is not a variable in the system.  Available variables:\n"
            << (std::vector<std::string>{Tensors::name()...}));
      }
      if (alg::count(variables_to_observe, name) != 1) {
        PARSE_ERROR(context, name << " specified multiple times");
      }
    }
    variables_to_observe_.insert(coordinates_tag::name());
  }

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using argument_tags =
      tmpl::list<::Tags::Time, ::Tags::Mesh<VolumeDim>, coordinates_tag,
                 AnalyticSolutionTensors..., NonSolutionTensors...>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(
      const Time& time, const Mesh<VolumeDim>& mesh,
      const db::item_type<coordinates_tag>& inertial_coordinates,
      const db::item_type<
          AnalyticSolutionTensors>&... analytic_solution_tensors,
      const db::item_type<NonSolutionTensors>&... non_solution_tensors,
      Parallel::ConstGlobalCache<Metavariables>& cache,
      const ElementIndex<VolumeDim>& array_index,
      const ParallelComponent* const /*meta*/) const noexcept {
    const std::string element_name =
        MakeString{} << ElementId<VolumeDim>(array_index) << '/';

    // Remove tensor types, only storing individual components.
    std::vector<TensorComponent> components;
    // This is larger than we need if we are only observing some
    // tensors, but that's not a big deal and calculating the correct
    // size is nontrivial.
    components.reserve(alg::accumulate(
        std::initializer_list<size_t>{
            inertial_coordinates.size(),
            db::item_type<AnalyticSolutionTensors>::size()...,
            db::item_type<NonSolutionTensors>::size()...},
        0_st));

    const auto record_tensor_components = [this, &components, &element_name](
        const auto tensor_tag_v, const auto& tensor) noexcept {
      using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
      if (variables_to_observe_.count(tensor_tag::name()) == 1) {
        for (size_t i = 0; i < tensor.size(); ++i) {
          components.emplace_back(
              element_name + tensor_tag::name() + component_suffix(tensor, i),
              tensor[i]);
        }
      }
    };
    record_tensor_components(tmpl::type_<coordinates_tag>{},
                             inertial_coordinates);
    EXPAND_PACK_LEFT_TO_RIGHT(record_tensor_components(
        tmpl::type_<AnalyticSolutionTensors>{}, analytic_solution_tensors));
    EXPAND_PACK_LEFT_TO_RIGHT(record_tensor_components(
        tmpl::type_<NonSolutionTensors>{}, non_solution_tensors));

    using solution_tag = OptionTags::AnalyticSolutionBase;
    const auto exact_solution = Parallel::get<solution_tag>(cache).variables(
        inertial_coordinates, time.value(),
        tmpl::list<AnalyticSolutionTensors...>{});

    tuples::TaggedTuple<LocalSquareError<AnalyticSolutionTensors>...>
        local_square_errors;
    const auto record_errors = [
      this, &components, &element_name, &exact_solution, &local_square_errors
    ](const auto tensor_tag_v, const auto& tensor) noexcept {
      using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
      double local_square_error = 0.0;
      for (size_t i = 0; i < tensor.size(); ++i) {
        DataVector error = tensor[i] - get<tensor_tag>(exact_solution)[i];
        local_square_error += alg::accumulate(square(error), 0.0);
        // The reduction has to observe all variables because the
        // reduction type is determined at compile time, so we can
        // only restrict the volume measurement based on
        // variables_to_observe_.
        if (variables_to_observe_.count(tensor_tag::name()) == 1) {
          components.emplace_back(element_name + "Error" + tensor_tag::name() +
                                      component_suffix(tensor, i),
                                  std::move(error));
        }
      }
      get<LocalSquareError<tensor_tag>>(local_square_errors) =
          local_square_error;
    };
    EXPAND_PACK_LEFT_TO_RIGHT(record_errors(
        tmpl::type_<AnalyticSolutionTensors>{}, analytic_solution_tensors));

    // Send data to volume observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        local_observer,
        observers::ObservationId(
            time.value(), typename Metavariables::element_observation_type{}),
        std::string{"/element_data"},
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementIndex<VolumeDim>>(array_index)),
        std::move(components), mesh.extents());

    // Send data to reduction observer
    Parallel::simple_action<observers::Actions::ContributeReductionData>(
        local_observer,
        observers::ObservationId(
            time.value(), typename Metavariables::element_observation_type{}),
        std::string{"/element_data"},
        std::vector<std::string>{
            "Time", "NumberOfPoints",
            ("Error" + AnalyticSolutionTensors::name())...},
        ReductionData{time.value(), mesh.number_of_grid_points(),
                      std::move(get<LocalSquareError<AnalyticSolutionTensors>>(
                          local_square_errors))...});
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override {
    Event<EventRegistrars>::pup(p);
    p | variables_to_observe_;
  }

 private:
  std::unordered_set<std::string> variables_to_observe_{};
};

/// \cond
template <size_t VolumeDim, typename... Tensors,
          typename... AnalyticSolutionTensors, typename EventRegistrars,
          typename... NonSolutionTensors>
PUP::able::PUP_ID
    Observe<VolumeDim, tmpl::list<Tensors...>,
            tmpl::list<AnalyticSolutionTensors...>, EventRegistrars,
            tmpl::list<NonSolutionTensors...>>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
}  // namespace dg
