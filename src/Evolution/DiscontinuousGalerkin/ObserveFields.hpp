// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <initializer_list>
#include <pup.h>
#include <string>
#include <type_traits>
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
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/VolumeActions.hpp"  // IWYU pragma: keep
#include "Options/Options.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Time.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"

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
class ObserveFields;

namespace Registrars {
template <size_t VolumeDim, typename Tensors,
          typename AnalyticSolutionTensors = tmpl::list<>>
struct ObserveFields {
  template <typename RegistrarList>
  using f = Events::ObserveFields<VolumeDim, Tensors, AnalyticSolutionTensors,
                                  RegistrarList>;
};
}  // namespace Registrars

template <size_t VolumeDim, typename Tensors,
          typename AnalyticSolutionTensors = tmpl::list<>,
          typename EventRegistrars = tmpl::list<Registrars::ObserveFields<
              VolumeDim, Tensors, AnalyticSolutionTensors>>,
          typename NonSolutionTensors>
class ObserveFields;  // IWYU pragma: keep

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief %Observe volume tensor fields.
 *
 * Writes volume quantities:
 * - `InertialCoordinates`
 * - Tensors listed in `Tensors` template parameter
 * - `Error(*)` = errors in `AnalyticSolutionTensors` =
 *   \f$\text{value} - \text{analytic solution}\f$
 *
 * \warning Currently, only one volume observation event can be
 * triggered at a given time.  Causing multiple events to run at once
 * will produce unpredictable results.
 */
template <size_t VolumeDim, typename... Tensors,
          typename... AnalyticSolutionTensors, typename EventRegistrars,
          typename... NonSolutionTensors>
class ObserveFields<VolumeDim, tmpl::list<Tensors...>,
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
  explicit ObserveFields(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveFields);  // NOLINT
  /// \endcond

  struct VariablesToObserve {
    static constexpr OptionString help = "Subset of variables to observe";
    using type = std::vector<std::string>;
    static type default_value() noexcept { return {Tensors::name()...}; }
    static size_t lower_bound_on_size() noexcept { return 1; }
  };

  using options = tmpl::list<VariablesToObserve>;
  static constexpr OptionString help =
      "Observe volume tensor fields.\n"
      "\n"
      "Writes volume quantities:\n"
      " * InertialCoordinates\n"
      " * Tensors listed in Tensors template parameter\n"
      " * Error(*) = errors in AnalyticSolutionTensors\n"
      "            = value - analytic solution\n"
      "\n"
      "Warning: Currently, only one volume observation event can be\n"
      "triggered at a given time.  Causing multiple events to run at once\n"
      "will produce unpredictable results.";

  explicit ObserveFields(const std::vector<std::string>& variables_to_observe =
                             VariablesToObserve::default_value(),
                         const OptionContext& context = {})
      : variables_to_observe_(variables_to_observe.begin(),
                              variables_to_observe.end()) {
    const std::unordered_set<std::string> valid_tensors{Tensors::name()...};
    for (const auto& name : variables_to_observe_) {
      if (valid_tensors.count(name) != 1) {
        PARSE_ERROR(
            context,
            name << " is not an available variable.  Available variables:\n"
            << (std::vector<std::string>{Tensors::name()...}));
      }
      if (alg::count(variables_to_observe, name) != 1) {
        PARSE_ERROR(context, name << " specified multiple times");
      }
    }
    variables_to_observe_.insert(coordinates_tag::name());
  }

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

    const auto record_errors =
        [ this, &inertial_coordinates, &time, &components, &
          element_name ](const auto tensor_tag_v, const auto& tensor,
                         const auto& local_cache) noexcept {
      const auto analytic_solution =
          Parallel::get<OptionTags::AnalyticSolutionBase>(local_cache)
              .variables(inertial_coordinates, time.value(),
                         tmpl::list<AnalyticSolutionTensors...>{});
      using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
      if (variables_to_observe_.count(tensor_tag::name()) == 1) {
        for (size_t i = 0; i < tensor.size(); ++i) {
          DataVector error = tensor[i] - get<tensor_tag>(analytic_solution)[i];
          components.emplace_back(element_name + "Error(" + tensor_tag::name() +
                                      ")" + component_suffix(tensor, i),
                                  std::move(error));
        }
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(
        record_errors(tmpl::type_<AnalyticSolutionTensors>{},
                      analytic_solution_tensors, cache));

    (void)(record_errors);  // Silence GCC warning about unused variable

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
    ObserveFields<VolumeDim, tmpl::list<Tensors...>,
                  tmpl::list<AnalyticSolutionTensors...>, EventRegistrars,
                  tmpl::list<NonSolutionTensors...>>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
}  // namespace dg
