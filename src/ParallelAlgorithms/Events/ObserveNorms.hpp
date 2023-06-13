// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/GetSectionObservationKey.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/ReductionActions.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/Events/Tags.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/OptionalHelpers.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Events {
/// @{
/*!
 * \brief Compute norms of tensors in the DataBox and write them to disk.
 *
 * The L2 norm is computed as the RMS, so
 *
 * \f{align*}{
 * L_2(u)=\sqrt{\frac{1}{N}\sum_{i=0}^{N} u_i^2}
 * \f}
 *
 * where \f$N\f$ is the number of grid points.
 *
 * The norm can be taken for each individual component, or summed over
 * components. For the max/min it is then the max/min over all components, while
 * for the L2 norm we have (for a 3d vector, 2d and 1d are similar)
 *
 * \f{align*}{
 * L_2(v^k)=\sqrt{\frac{1}{N}\sum_{i=0}^{N} \left[(v^x_i)^2 + (v^y_i)^2
 *          + (v^z_i)^2\right]}
 * \f}
 *
 * The L2 integral norm is:
 *
 * \begin{equation}
 * L_{2,\mathrm{int}}(v^k) = \sqrt{\frac{1}{V}\int_\Omega \left[
 *   (v^x_i)^2 + (v^y_i)^2 + (v^z_i)^2\right] \mathrm{d}V}
 * \end{equation}
 *
 * where $V=\int_\Omega$ is the volume of the entire domain in inertial
 * coordinates.
 *
 * Here is an example of an input file:
 *
 * \snippet Test_ObserveNorms.cpp input_file_examples
 *
 * \note The `NonTensorComputeTags` are intended to be used for `Variables`
 * compute tags like `Tags::DerivCompute`
 *
 * \par Array sections
 * This event supports sections (see `Parallel::Section`). Set the
 * `ArraySectionIdTag` template parameter to split up observations into subsets
 * of elements. The `observers::Tags::ObservationKey<ArraySectionIdTag>` must be
 * available in the DataBox. It identifies the section and is used as a suffix
 * for the path in the output file.
 */
template <typename ObservationValueTag, typename ObservableTensorTagsList,
          typename NonTensorComputeTagsList = tmpl::list<>,
          typename ArraySectionIdTag = void>
class ObserveNorms;

template <typename ObservationValueTag, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
class ObserveNorms<ObservationValueTag, tmpl::list<ObservableTensorTags...>,
                   tmpl::list<NonTensorComputeTags...>, ArraySectionIdTag>
    : public Event {
 private:
  struct ObserveTensor {
    static constexpr Options::String help = {
        "The tensor to reduce, and how to reduce it."};

    struct Name {
      using type = std::string;
      static constexpr Options::String help = {
          "The name of the tensor to observe."};
    };
    struct NormType {
      using type = std::string;
      static constexpr Options::String help = {
          "The type of norm to use. Must be one of Max, Min, L2Norm, or "
          "L2IntegralNorm."};
    };
    struct Components {
      using type = std::string;
      static constexpr Options::String help = {
          "How to handle tensor components. Must be Individual or Sum."};
    };

    using options = tmpl::list<Name, NormType, Components>;

    ObserveTensor() = default;

    ObserveTensor(std::string in_tensor, std::string in_norm_type,
                  std::string in_components,
                  const Options::Context& context = {});

    std::string tensor{};
    std::string norm_type{};
    std::string components{};
  };

  using ReductionData = Parallel::ReductionData<
      // Observation value
      Parallel::ReductionDatum<double, funcl::AssertEqual<>>,
      // Number of grid points
      Parallel::ReductionDatum<size_t, funcl::Plus<>>,
      // Total volume
      Parallel::ReductionDatum<double, funcl::Plus<>>,
      // Max
      Parallel::ReductionDatum<std::vector<double>,
                               funcl::ElementWise<funcl::Max<>>>,
      // Min
      Parallel::ReductionDatum<std::vector<double>,
                               funcl::ElementWise<funcl::Min<>>>,
      // L2Norm
      Parallel::ReductionDatum<
          std::vector<double>, funcl::ElementWise<funcl::Plus<>>,
          funcl::ElementWise<funcl::Sqrt<funcl::Divides<>>>,
          std::index_sequence<1>>,
      // L2IntegralNorm
      Parallel::ReductionDatum<
          std::vector<double>, funcl::ElementWise<funcl::Plus<>>,
          funcl::ElementWise<funcl::Sqrt<funcl::Divides<>>>,
          std::index_sequence<2>>>;

 public:
  /// The name of the subfile inside the HDF5 file
  struct SubfileName {
    using type = std::string;
    static constexpr Options::String help = {
        "The name of the subfile inside the HDF5 file without an extension and "
        "without a preceding '/'."};
  };
  /// The tensor to observe and how to do the reduction
  struct TensorsToObserve {
    using type = std::vector<ObserveTensor>;
    static constexpr Options::String help = {
        "List specifying each tensor to observe and how it is reduced."};
  };

  explicit ObserveNorms(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ObserveNorms);  // NOLINT

  using options = tmpl::list<SubfileName, TensorsToObserve>;

  static constexpr Options::String help =
      "Observe norms of tensors in the DataBox.\n"
      "\n"
      "You can choose the norm type for each observation. Note that the\n"
      "'L2Norm' (root mean square) emphasizes regions of the domain with many\n"
      "grid points, whereas the 'L2IntegralNorm' emphasizes regions of the\n"
      "domain with large volume. Choose wisely! When in doubt, try the\n"
      "'L2Norm' first.\n"
      "The 'L2IntegralNorm' does not currently work with finite difference\n"
      "(subcell) meshes. [We need to figure out how to provide the proper \n"
      "determinant of the Jacobian]\n"
      "\n"
      "Writes reduction quantities:\n"
      " * ObservationValueTag (e.g. Time or IterationId)\n"
      " * NumberOfPoints = total number of points in the domain\n"
      " * Volume = total volume of the domain in inertial coordinates\n"
      " * Max values\n"
      " * Min values\n"
      " * L2-norm values\n"
      " * L2 integral norm values\n";

  ObserveNorms() = default;

  ObserveNorms(const std::string& subfile_name,
               const std::vector<ObserveTensor>& observe_tensors);

  using observed_reduction_data_tags =
      observers::make_reduction_data_tags<tmpl::list<ReductionData>>;

  using compute_tags_for_observation_box =
      tmpl::list<ObservableTensorTags..., NonTensorComputeTags...>;

  using argument_tags = tmpl::list<ObservationValueTag, ::Tags::ObservationBox>;

  template <typename ComputeTagsList, typename DataBoxType,
            typename Metavariables, size_t VolumeDim,
            typename ParallelComponent>
  void operator()(const typename ObservationValueTag::type& observation_value,
                  const ObservationBox<ComputeTagsList, DataBoxType>& box,
                  Parallel::GlobalCache<Metavariables>& cache,
                  const ElementId<VolumeDim>& array_index,
                  const ParallelComponent* const /*meta*/) const;

  using observation_registration_tags = tmpl::list<::Tags::DataBox>;

  template <typename DbTagsList>
  std::optional<
      std::pair<observers::TypeOfObservation, observers::ObservationKey>>
  get_observation_type_and_key_for_registration(
      const db::DataBox<DbTagsList>& box) const {
    const std::optional<std::string> section_observation_key =
        observers::get_section_observation_key<ArraySectionIdTag>(box);
    if (not section_observation_key.has_value()) {
      return std::nullopt;
    }
    return {{observers::TypeOfObservation::Reduction,
             observers::ObservationKey(
                 subfile_path_ + section_observation_key.value() + ".dat")}};
  }

  using is_ready_argument_tags = tmpl::list<>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const {
    return true;
  }

  bool needs_evolved_variables() const override { return true; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  std::string subfile_path_;
  std::vector<std::string> tensor_names_{};
  std::vector<std::string> tensor_norm_types_{};
  std::vector<std::string> tensor_components_{};
};
/// @}

/// \cond
template <typename ObservationValueTag, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
ObserveNorms<ObservationValueTag, tmpl::list<ObservableTensorTags...>,
             tmpl::list<NonTensorComputeTags...>,
             ArraySectionIdTag>::ObserveNorms(CkMigrateMessage* msg)
    : Event(msg) {}

template <typename ObservationValueTag, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
ObserveNorms<ObservationValueTag, tmpl::list<ObservableTensorTags...>,
             tmpl::list<NonTensorComputeTags...>,
             ArraySectionIdTag>::ObserveNorms(const std::string& subfile_name,
                                              const std::vector<ObserveTensor>&
                                                  observe_tensors)
    : subfile_path_("/" + subfile_name) {
  tensor_names_.reserve(observe_tensors.size());
  tensor_norm_types_.reserve(observe_tensors.size());
  tensor_components_.reserve(observe_tensors.size());
  for (const auto& observe_tensor : observe_tensors) {
    tensor_names_.push_back(observe_tensor.tensor);
    tensor_norm_types_.push_back(observe_tensor.norm_type);
    tensor_components_.push_back(observe_tensor.components);
  }
}

template <typename ObservationValueTag, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
ObserveNorms<ObservationValueTag, tmpl::list<ObservableTensorTags...>,
             tmpl::list<NonTensorComputeTags...>, ArraySectionIdTag>::
    ObserveTensor::ObserveTensor(std::string in_tensor,
                                 std::string in_norm_type,
                                 std::string in_components,
                                 const Options::Context& context)
    : tensor(std::move(in_tensor)),
      norm_type(std::move(in_norm_type)),
      components(std::move(in_components)) {
  if (((tensor != db::tag_name<ObservableTensorTags>()) and ...)) {
    PARSE_ERROR(
        context, "Tensor '"
                     << tensor << "' is not known. Known tensors are: "
                     << ((db::tag_name<ObservableTensorTags>() + ",") + ...));
  }
  if (norm_type != "Max" and norm_type != "Min" and norm_type != "L2Norm" and
      norm_type != "L2IntegralNorm") {
    PARSE_ERROR(
        context,
        "NormType must be one of Max, Min, L2Norm, or L2IntegralNorm, not "
            << norm_type);
  }
  if (components != "Individual" and components != "Sum") {
    PARSE_ERROR(context,
                "Components must be Individual or Sum, not " << components);
  }
}

template <typename ObservationValueTag, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
template <typename ComputeTagsList, typename DataBoxType,
          typename Metavariables, size_t VolumeDim,
          typename ParallelComponent>
void ObserveNorms<ObservationValueTag, tmpl::list<ObservableTensorTags...>,
                  tmpl::list<NonTensorComputeTags...>, ArraySectionIdTag>::
operator()(const typename ObservationValueTag::type& observation_value,
           const ObservationBox<ComputeTagsList, DataBoxType>& box,
           Parallel::GlobalCache<Metavariables>& cache,
           const ElementId<VolumeDim>& array_index,
           const ParallelComponent* const /*meta*/) const {
  // Skip observation on elements that are not part of a section
  const std::optional<std::string> section_observation_key =
      observers::get_section_observation_key<ArraySectionIdTag>(box);
  if (not section_observation_key.has_value()) {
    return;
  }

  using tensor_tags = tmpl::list<ObservableTensorTags...>;

  std::unordered_map<std::string,
                     std::pair<std::vector<double>, std::vector<std::string>>>
      norm_values_and_names{};
  const auto& mesh = get<::Events::Tags::ObserverMesh<VolumeDim>>(box);
  const DataVector det_jacobian =
    1. / get(get<domain::Tags::DetInvJacobian<Frame::ElementLogical,
                                              Frame::Inertial>>(box));
  const size_t number_of_points = mesh.number_of_grid_points();
  const double local_volume = [&mesh, &det_jacobian]() {
    if (mesh.basis(0) == Spectral::Basis::FiniteDifference) {
      return std::numeric_limits<double>::quiet_NaN();
    } else {
      return definite_integral(det_jacobian, mesh);
    }
  }();

  // Loop over ObservableTensorTags and see if it was requested to be observed.
  // This approach allows us to delay evaluating any compute tags until they're
  // actually needed for observing.
  tmpl::for_each<tensor_tags>([this, &box, &norm_values_and_names,
                               &number_of_points, &mesh,
                               &det_jacobian](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    const std::string tensor_name = db::tag_name<tag>();
    for (size_t i = 0; i < tensor_names_.size(); ++i) {
      if (tensor_name == tensor_names_[i]) {
        if (UNLIKELY(not has_value(get<tag>(box)))) {
          ERROR("Cannot observe a norm of '"
                << tensor_name
                << "' because it is a std::optional and wasn't able to be "
                   "computed. This can happen when you try to observe errors "
                   "without an analytic solution.");
        }
        const auto& tensor = value(get<tag>(box));

        auto& [values, names] = norm_values_and_names[tensor_norm_types_[i]];
        const auto names_and_components = tensor.get_vector_of_data();
        const auto& component_names = names_and_components.first;
        const auto& components = names_and_components.second;
        if (components[0].size() != number_of_points) {
          ERROR("The number of grid points of the mesh is "
                << number_of_points << " but the tensor '" << tensor_name
                << "' has " << components[0].size()
                << " points. This means you're computing norms of tensors over "
                   "different grids, which will give the wrong answer for "
                   "norms that use the grid points.");
        }

        if (tensor_components_[i] == "Individual") {
          for (size_t storage_index = 0; storage_index < component_names.size();
               ++storage_index) {
            if (tensor_norm_types_[i] == "Max") {
              values.push_back(max(components[storage_index]));
            } else if (tensor_norm_types_[i] == "Min") {
              values.push_back(min(components[storage_index]));
            } else if (tensor_norm_types_[i] == "L2Norm") {
              values.push_back(
                  alg::accumulate(square(components[storage_index]), 0.0));
            } else if (tensor_norm_types_[i] == "L2IntegralNorm") {
             if (mesh.basis(0) == Spectral::Basis::FiniteDifference) {
                ERROR(
                    "The 'L2IntegralNorm' is currently not supported on finite "
                    "difference (subcell) meshes.");
              }
              values.push_back(definite_integral(
                  square(components[storage_index]) * det_jacobian, mesh));
            }
            names.push_back(
                tensor_norm_types_[i] + "(" +
                (component_names.size() == 1
                     ? tensor_name
                     : (tensor_name + "_" + component_names[storage_index])) +
                ")");
          }
        } else if (tensor_components_[i] == "Sum") {
          double value = 0.0;
          if (tensor_norm_types_[i] == "Max") {
            value = std::numeric_limits<double>::min();
          } else if (tensor_norm_types_[i] == "Min") {
            value = std::numeric_limits<double>::max();
          }
          for (size_t storage_index = 0; storage_index < component_names.size();
               ++storage_index) {
            if (tensor_norm_types_[i] == "Max") {
              value = std::max(value, max(components[storage_index]));
            } else if (tensor_norm_types_[i] == "Min") {
              value = std::min(value, min(components[storage_index]));
            } else if (tensor_norm_types_[i] == "L2Norm") {
              value += alg::accumulate(square(components[storage_index]), 0.0);
            } else if (tensor_norm_types_[i] == "L2IntegralNorm") {
              value += definite_integral(
                  square(components[storage_index]) * det_jacobian, mesh);
            }
          }

          names.push_back(tensor_norm_types_[i] + "(" + tensor_name + ")");
          values.push_back(value);
        }
      }
    }
  });

  // Concatenate the legend info together.
  std::vector<std::string> legend{db::tag_name<ObservationValueTag>(),
                                  "NumberOfPoints", "Volume"};
  legend.insert(legend.end(), norm_values_and_names["Max"].second.begin(),
                norm_values_and_names["Max"].second.end());
  legend.insert(legend.end(), norm_values_and_names["Min"].second.begin(),
                norm_values_and_names["Min"].second.end());
  legend.insert(legend.end(), norm_values_and_names["L2Norm"].second.begin(),
                norm_values_and_names["L2Norm"].second.end());
  legend.insert(legend.end(),
                norm_values_and_names["L2IntegralNorm"].second.begin(),
                norm_values_and_names["L2IntegralNorm"].second.end());

  // Send data to reduction observer
  auto& local_observer = *Parallel::local_branch(
      Parallel::get_parallel_component<observers::Observer<Metavariables>>(
          cache));
  const std::string subfile_path_with_suffix =
      subfile_path_ + section_observation_key.value();
  Parallel::simple_action<observers::Actions::ContributeReductionData>(
      local_observer,
      observers::ObservationId(observation_value,
                               subfile_path_with_suffix + ".dat"),
      observers::ArrayComponentId{
          std::add_pointer_t<ParallelComponent>{nullptr},
          Parallel::ArrayIndex<ElementId<VolumeDim>>(array_index)},
      subfile_path_with_suffix, std::move(legend),
      ReductionData{static_cast<double>(observation_value), number_of_points,
                    local_volume, std::move(norm_values_and_names["Max"].first),
                    std::move(norm_values_and_names["Min"].first),
                    std::move(norm_values_and_names["L2Norm"].first),
                    std::move(norm_values_and_names["L2IntegralNorm"].first)});
}

template <typename ObservationValueTag, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
void ObserveNorms<ObservationValueTag, tmpl::list<ObservableTensorTags...>,
                  tmpl::list<NonTensorComputeTags...>,
                  ArraySectionIdTag>::pup(PUP::er& p) {
  Event::pup(p);
  p | subfile_path_;
  p | tensor_names_;
  p | tensor_norm_types_;
  p | tensor_components_;
}

template <typename ObservationValueTag, typename... ObservableTensorTags,
          typename... NonTensorComputeTags, typename ArraySectionIdTag>
PUP::able::PUP_ID
    ObserveNorms<ObservationValueTag, tmpl::list<ObservableTensorTags...>,
                 tmpl::list<NonTensorComputeTags...>,
                 ArraySectionIdTag>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Events
