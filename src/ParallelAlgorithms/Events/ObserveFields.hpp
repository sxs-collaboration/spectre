// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <initializer_list>
#include <optional>
#include <pup.h>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/VolumeActions.hpp"      // IWYU pragma: keep
#include "NumericalAlgorithms/Interpolation/RegularGridInterpolant.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/StdHelpers.hpp"
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
template <size_t VolumeDim, typename ObservationValueTag, typename Tensors,
          typename AnalyticSolutionTensors, typename EventRegistrars,
          typename NonSolutionTensors =
              tmpl::list_difference<Tensors, AnalyticSolutionTensors>>
class ObserveFields;

namespace Registrars {
template <size_t VolumeDim, typename ObservationValueTag, typename Tensors,
          typename AnalyticSolutionTensors = tmpl::list<>>
struct ObserveFields {
  template <typename RegistrarList>
  using f = Events::ObserveFields<VolumeDim, ObservationValueTag, Tensors,
                                  AnalyticSolutionTensors, RegistrarList>;
};
}  // namespace Registrars

template <
    size_t VolumeDim, typename ObservationValueTag, typename Tensors,
    typename AnalyticSolutionTensors = tmpl::list<>,
    typename EventRegistrars = tmpl::list<Registrars::ObserveFields<
        VolumeDim, ObservationValueTag, Tensors, AnalyticSolutionTensors>>,
    typename NonSolutionTensors>
class ObserveFields;  // IWYU pragma: keep

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief %Observe volume tensor fields.
 *
 * A class that writes volume quantities to an h5 file during the simulation.
 * The observed quantitites are:
 * - `InertialCoordinates`
 * - Tensors listed in `Tensors` template parameter
 * - `Error(*)` = errors in `AnalyticSolutionTensors` =
 *   \f$\text{value} - \text{analytic solution}\f$
 *
 * The user may specify an `interpolation_mesh` to which the
 * data is interpolated.
 */
template <size_t VolumeDim, typename ObservationValueTag, typename... Tensors,
          typename... AnalyticSolutionTensors, typename EventRegistrars,
          typename... NonSolutionTensors>
class ObserveFields<VolumeDim, ObservationValueTag, tmpl::list<Tensors...>,
                    tmpl::list<AnalyticSolutionTensors...>, EventRegistrars,
                    tmpl::list<NonSolutionTensors...>>
    : public Event<EventRegistrars> {
 private:
  static_assert(
      std::is_same_v<
          tmpl::list_difference<tmpl::list<AnalyticSolutionTensors...>,
                                tmpl::list<Tensors...>>,
          tmpl::list<>>,
      "All AnalyticSolutionTensors must be listed in Tensors.");
  using coordinates_tag = domain::Tags::Coordinates<VolumeDim, Frame::Inertial>;

 public:
  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'."};
  };

  /// \cond
  explicit ObserveFields(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveFields);  // NOLINT
  /// \endcond

  struct VariablesToObserve {
    static constexpr Options::String help = "Subset of variables to observe";
    using type = std::vector<std::string>;
    static size_t lower_bound_on_size() noexcept { return 1; }
  };

  struct InterpolateToMesh {
    using type = Options::Auto<Mesh<VolumeDim>, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "An optional mesh to which the variables are interpolated. This mesh "
        "specifies any number of collocation points, basis, and quadrature on "
        "which the observed quantities are evaluated. If no mesh is given, the "
        "results will be evaluated on the mesh the simulation runs on. The "
        "user may add several ObserveField Events e.g. with and without an "
        "interpolating mesh to output the data both on the original mesh and "
        "on a new mesh.";
  };

  using options =
      tmpl::list<SubfileName, VariablesToObserve, InterpolateToMesh>;
  static constexpr Options::String help =
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

  ObserveFields() = default;

  explicit ObserveFields(const std::string& subfile_name,
                         const std::vector<std::string>& variables_to_observe,
                         std::optional<Mesh<VolumeDim>> interpolation_mesh = {},
                         const Options::Context& context = {})
      : subfile_path_("/" + subfile_name),
        variables_to_observe_(variables_to_observe.begin(),
                              variables_to_observe.end()),
        interpolation_mesh_(interpolation_mesh) {
    using ::operator<<;
    const std::unordered_set<std::string> valid_tensors{
        db::tag_name<Tensors>()...};
    for (const auto& name : variables_to_observe_) {
      if (valid_tensors.count(name) != 1) {
        PARSE_ERROR(
            context,
            name << " is not an available variable.  Available variables:\n"
                 << (std::vector<std::string>{db::tag_name<Tensors>()...}));
      }
      if (alg::count(variables_to_observe, name) != 1) {
        PARSE_ERROR(context, name << " specified multiple times");
      }
    }
  }

  using argument_tags = tmpl::list<
      ObservationValueTag, domain::Tags::Mesh<VolumeDim>, coordinates_tag,
      domain::Tags::ElementMap<VolumeDim>, AnalyticSolutionTensors...,
      ::Tags::Analytic<AnalyticSolutionTensors>..., NonSolutionTensors...>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(
      const typename ObservationValueTag::type& observation_value,
      const Mesh<VolumeDim>& mesh,
      const tnsr::I<DataVector, VolumeDim, Frame::Inertial>&
          inertial_coordinates,
      const ElementMap<VolumeDim, Frame::Inertial>& element_map,
      const typename AnalyticSolutionTensors::
          type&... analytic_solution_tensors,
      const typename ::Tags::Analytic<
          AnalyticSolutionTensors>::type&... analytic_solutions,
      const typename NonSolutionTensors::type&... non_solution_tensors,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<VolumeDim>& array_index,
      const ParallelComponent* const /*meta*/) const noexcept {
    const std::string element_name =
        MakeString{} << ElementId<VolumeDim>(array_index) << '/';

    // if no interpolation_mesh is provided, the interpolation is essentially
    // ignored by the RegularGridInterpolant except for a single copy.
    const intrp::RegularGrid interpolant(mesh,
                                         interpolation_mesh_.value_or(mesh));

    // Remove tensor types, only storing individual components.
    std::vector<TensorComponent> components;
    // This is larger than we need if we are only observing some
    // tensors, but that's not a big deal and calculating the correct
    // size is nontrivial.
    components.reserve(alg::accumulate(
        std::initializer_list<size_t>{
            inertial_coordinates.size(),
            2 * AnalyticSolutionTensors::type::size()...,
            NonSolutionTensors::type::size()...},
        0_st));

    // interpolated coordinates can be calculated exactly with element_map
    auto mapped_coordinates =
        interpolation_mesh_.has_value()
            ? element_map(logical_coordinates(*interpolation_mesh_))
            : inertial_coordinates;
    for (size_t i = 0; i < VolumeDim; ++i) {
      components.emplace_back(element_name + coordinates_tag::name() +
                                  inertial_coordinates.component_suffix(i),
                              std::move(mapped_coordinates[i]));
    }

    const auto record_tensor_components = [this, &components, &element_name,
                                           &interpolant](
                                              const auto tensor_tag_v,
                                              const auto& tensor) noexcept {
      using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
      if (variables_to_observe_.count(db::tag_name<tensor_tag>()) == 1) {
        for (size_t i = 0; i < tensor.size(); ++i) {
          components.emplace_back(element_name + db::tag_name<tensor_tag>() +
                                      tensor.component_suffix(i),
                                  interpolant.interpolate(tensor[i]));
        }
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(record_tensor_components(
        tmpl::type_<AnalyticSolutionTensors>{}, analytic_solution_tensors));
    EXPAND_PACK_LEFT_TO_RIGHT(record_tensor_components(
        tmpl::type_<NonSolutionTensors>{}, non_solution_tensors));

    const auto record_errors = [this, &components, &element_name, &interpolant](
                                   const auto tensor_tag_v, const auto& tensor,
                                   const auto& analytic_tensor) noexcept {
      using tensor_tag = tmpl::type_from<decltype(tensor_tag_v)>;
      if (variables_to_observe_.count(db::tag_name<tensor_tag>()) == 1) {
        for (size_t i = 0; i < tensor.size(); ++i) {
          DataVector error = tensor[i] - analytic_tensor[i];
          components.emplace_back(element_name + "Error(" +
                                      db::tag_name<tensor_tag>() + ")" +
                                      tensor.component_suffix(i),
                                  interpolant.interpolate(error));
        }
      }
    };
    EXPAND_PACK_LEFT_TO_RIGHT(
        record_errors(tmpl::type_<AnalyticSolutionTensors>{},
                      analytic_solution_tensors, analytic_solutions));

    (void)(record_errors);  // Silence GCC warning about unused variable

    // Send data to volume observer
    auto& local_observer =
        *Parallel::get_parallel_component<observers::Observer<Metavariables>>(
             cache)
             .ckLocalBranch();
    Parallel::simple_action<observers::Actions::ContributeVolumeData>(
        local_observer,
        observers::ObservationId(observation_value, subfile_path_ + ".vol"),
        subfile_path_,
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementId<VolumeDim>>(array_index)),
        std::move(components), interpolation_mesh_.value_or(mesh).extents(),
        interpolation_mesh_.value_or(mesh).basis(),
        interpolation_mesh_.value_or(mesh).quadrature());
  }

  using observation_registration_tags = tmpl::list<>;
  std::pair<observers::TypeOfObservation, observers::ObservationKey>
  get_observation_type_and_key_for_registration() const noexcept {
    return {observers::TypeOfObservation::Volume,
            observers::ObservationKey(subfile_path_ + ".vol")};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override {
    Event<EventRegistrars>::pup(p);
    p | subfile_path_;
    p | variables_to_observe_;
    p | interpolation_mesh_;
  }

 private:
  std::string subfile_path_;
  std::unordered_set<std::string> variables_to_observe_{};
  std::optional<Mesh<VolumeDim>> interpolation_mesh_{};
};

/// \cond
template <size_t VolumeDim, typename ObservationValueTag, typename... Tensors,
          typename... AnalyticSolutionTensors, typename EventRegistrars,
          typename... NonSolutionTensors>
PUP::able::PUP_ID
    ObserveFields<VolumeDim, ObservationValueTag, tmpl::list<Tensors...>,
                  tmpl::list<AnalyticSolutionTensors...>, EventRegistrars,
                  tmpl::list<NonSolutionTensors...>>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
}  // namespace dg
