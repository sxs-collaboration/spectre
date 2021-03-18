// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>
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
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/TypeTraits.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/VolumeActions.hpp"      // IWYU pragma: keep
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "ParallelAlgorithms/Events/ObserveFields.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
template <size_t Dim>
class Mesh;
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

namespace evolution::dg::subcell::Events {
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
                         const Options::Context& context = {});

  using coordinates_tag =
      ::domain::Tags::Coordinates<VolumeDim, Frame::Inertial>;

  using argument_tags =
      tmpl::list<ObservationValueTag, ::domain::Tags::Mesh<VolumeDim>,
                 subcell::Tags::Mesh<VolumeDim>, subcell::Tags::ActiveGrid,
                 ::domain::Tags::Coordinates<VolumeDim, Frame::Grid>,
                 subcell::Tags::Coordinates<VolumeDim, Frame::Grid>,
                 ::domain::CoordinateMaps::Tags::CoordinateMap<
                     VolumeDim, Frame::Grid, Frame::Inertial>,
                 ::domain::Tags::FunctionsOfTime, AnalyticSolutionTensors...,
                 NonSolutionTensors...>;

  template <typename Metavariables, typename ParallelComponent>
  void operator()(
      const double observation_value, const Mesh<VolumeDim>& dg_mesh,
      const Mesh<VolumeDim>& subcell_mesh,
      const subcell::ActiveGrid active_grid,
      const tnsr::I<DataVector, VolumeDim, Frame::Grid>& dg_grid_coordinates,
      const tnsr::I<DataVector, VolumeDim, Frame::Grid>&
          subcell_grid_coordinates,
      const ::domain::CoordinateMapBase<Frame::Grid, Frame::Inertial,
                                        VolumeDim>& grid_to_inertial_map,
      const std::unordered_map<
          std::string,
          std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
          functions_of_time,
      const typename AnalyticSolutionTensors::
          type&... analytic_solution_tensors,
      const typename NonSolutionTensors::type&... non_solution_tensors,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<VolumeDim>& array_index,
      const ParallelComponent* const component) const noexcept {
    std::optional<
        Variables<tmpl::list<::Tags::Analytic<AnalyticSolutionTensors>...>>>
        analytic_solution_variables{};
    const auto set_analytic_soln =
        [&analytic_solution_variables, &cache, &observation_value](
            const Mesh<VolumeDim>& mesh,
            const tnsr::I<DataVector, VolumeDim, Frame::Inertial>&
                inertial_coords) noexcept {
          if constexpr (evolution::is_analytic_solution_v<
                            typename Metavariables::initial_data>) {
            Variables<tmpl::list<AnalyticSolutionTensors...>> soln_vars{
                mesh.number_of_grid_points()};
            soln_vars.assign_subset(
                Parallel::get<::Tags::AnalyticSolutionBase>(cache).variables(
                    inertial_coords, observation_value,
                    tmpl::list<AnalyticSolutionTensors...>{}));
            analytic_solution_variables = std::move(soln_vars);
          } else {
            (void)analytic_solution_variables;
            (void)cache;
            (void)observation_value;
          }
        };
    if (active_grid == subcell::ActiveGrid::Dg) {
      const auto dg_inertial_coords = grid_to_inertial_map(
          dg_grid_coordinates, observation_value, functions_of_time);
      set_analytic_soln(dg_mesh, dg_inertial_coords);
      ::dg::Events::ObserveFields<
          VolumeDim, ObservationValueTag, tmpl::list<Tensors...>,
          tmpl::list<AnalyticSolutionTensors...>, EventRegistrars,
          tmpl::list<NonSolutionTensors...>>::
          call_operator_impl(
              subfile_path_, variables_to_observe_, interpolation_mesh_,
              observation_value, dg_mesh, dg_inertial_coords,
              analytic_solution_tensors..., non_solution_tensors...,
              analytic_solution_variables, cache, array_index, component);
    } else {
      ASSERT(active_grid == subcell::ActiveGrid::Subcell,
             "Active grid must be either Dg or Subcell");
      const auto subcell_inertial_coords = grid_to_inertial_map(
          subcell_grid_coordinates, observation_value, functions_of_time);
      set_analytic_soln(subcell_mesh, subcell_inertial_coords);
      ::dg::Events::ObserveFields<
          VolumeDim, ObservationValueTag, tmpl::list<Tensors...>,
          tmpl::list<AnalyticSolutionTensors...>, EventRegistrars,
          tmpl::list<NonSolutionTensors...>>::
          call_operator_impl(
              subfile_path_, variables_to_observe_, interpolation_mesh_,
              observation_value, subcell_mesh, subcell_inertial_coords,
              analytic_solution_tensors..., non_solution_tensors...,
              analytic_solution_variables, cache, array_index, component);
    }
  }

  // This overload is called when the list of analytic-solution tensors is
  // empty, i.e. it is clear at compile-time that no analytic solutions are
  // available
  template <typename Metavariables, typename ParallelComponent>
  void operator()(
      const double observation_value, const Mesh<VolumeDim>& mesh,
      const tnsr::I<DataVector, VolumeDim, Frame::Inertial>&
          inertial_coordinates,
      const typename NonSolutionTensors::type&... non_solution_tensors,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<VolumeDim>& array_index,
      const ParallelComponent* const component) const noexcept {
    this->operator()(observation_value, mesh, inertial_coordinates,
                     non_solution_tensors..., std::nullopt, cache, array_index,
                     component);
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

  bool needs_evolved_variables() const noexcept override { return true; }

 private:
  std::string subfile_path_;
  std::unordered_set<std::string> variables_to_observe_{};
  std::optional<Mesh<VolumeDim>> interpolation_mesh_{};
};

template <size_t VolumeDim, typename ObservationValueTag, typename... Tensors,
          typename... AnalyticSolutionTensors, typename EventRegistrars,
          typename... NonSolutionTensors>
ObserveFields<VolumeDim, ObservationValueTag, tmpl::list<Tensors...>,
              tmpl::list<AnalyticSolutionTensors...>, EventRegistrars,
              tmpl::list<NonSolutionTensors...>>::
    ObserveFields(const std::string& subfile_name,
                  const std::vector<std::string>& variables_to_observe,
                  std::optional<Mesh<VolumeDim>> interpolation_mesh,
                  const Options::Context& context)
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
  variables_to_observe_.insert(coordinates_tag::name());
  if (interpolation_mesh.has_value()) {
    PARSE_ERROR(
        context,
        "We don't yet support interpolating to a mesh with DG-subcell because "
        "we don't have an interpolator on the subcell grid. Trilinear "
        "interpolation could be added to support interpolation.");
  }
}

/// \cond
template <size_t VolumeDim, typename ObservationValueTag, typename... Tensors,
          typename... AnalyticSolutionTensors, typename EventRegistrars,
          typename... NonSolutionTensors>
PUP::able::PUP_ID
    ObserveFields<VolumeDim, ObservationValueTag, tmpl::list<Tensors...>,
                  tmpl::list<AnalyticSolutionTensors...>, EventRegistrars,
                  tmpl::list<NonSolutionTensors...>>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace evolution::dg::subcell::Events
