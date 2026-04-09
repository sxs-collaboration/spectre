// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <pup.h>
#include <string>
#include <variant>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave {

/*!
 * \brief Numeric initial data loaded from volume data files
 */
class NumericInitialData : public evolution::initial_data::InitialData {
 public:
  template <typename Tag>
  struct VarName {
    using tag = Tag;
    static std::string name() { return db::tag_name<Tag>(); }
    using type = std::string;
    static constexpr Options::String help =
        "Name of the variable in the volume data file";
  };

  // These are the scalar variables that we support loading from volume
  // data files
  using all_vars =
      tmpl::list<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
                 CurvedScalarWave::Tags::Phi<3>>;
  using optional_primitive_vars = tmpl::list<>;

  struct ScalarVars : tuples::tagged_tuple_from_typelist<
                          db::wrap_tags_in<VarName, all_vars>> {
    static constexpr Options::String help =
        "Scalar variables: 'Psi', 'Pi' and 'Phi'.";
    using options = tags_list;
    using TaggedTuple::TaggedTuple;
  };

  // Input-file options
  struct Variables {
    using type = ScalarVars;
    static constexpr Options::String help =
        "Set of initial data variables for the CurvedScalarWave system.";
  };

  using options =
      tmpl::push_back<importers::ImporterOptions::tags_list, Variables>;

  static constexpr Options::String help =
      "Numeric initial data loaded from volume data files";

  NumericInitialData() = default;
  NumericInitialData(const NumericInitialData& rhs) = default;
  NumericInitialData& operator=(const NumericInitialData& rhs) = default;
  NumericInitialData(NumericInitialData&& /*rhs*/) = default;
  NumericInitialData& operator=(NumericInitialData&& /*rhs*/) = default;
  ~NumericInitialData() override = default;

  /// \cond
  explicit NumericInitialData(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(NumericInitialData);
  /// \endcond

  std::unique_ptr<evolution::initial_data::InitialData> get_clone()
      const override {
    return std::make_unique<NumericInitialData>(*this);
  }

  NumericInitialData(
      std::string file_glob, std::string subfile_name,
      std::variant<double, importers::ObservationSelector> observation_value,
      std::optional<double> observation_value_epsilon,
      bool enable_interpolation, ScalarVars selected_variables);

  const importers::ImporterOptions& importer_options() const;

  const ScalarVars& selected_variables() const { return selected_variables_; }

  size_t volume_data_id() const;

  template <typename... AllTags>
  void select_for_import(
      const gsl::not_null<tuples::TaggedTuple<AllTags...>*> fields) const {
    // Select the subset of the available variables that we want to read from
    // the volume data file
    using selected_vars = std::decay_t<decltype(selected_variables_)>;
    tmpl::for_each<typename selected_vars::tags_list>(
        [&fields, this](const auto tag_v) {
          using tag = typename std::decay_t<decltype(tag_v)>::type::tag;
          get<importers::Tags::Selected<tag>>(*fields) =
              get<VarName<tag>>(selected_variables_);
        });
  }

  template <typename... AllTags>
  void set_initial_data(const gsl::not_null<Scalar<DataVector>*> psi_scalar,
                        const gsl::not_null<Scalar<DataVector>*> pi_scalar,
                        const gsl::not_null<tnsr::i<DataVector, 3>*> phi_scalar,
                        const gsl::not_null<tuples::TaggedTuple<AllTags...>*>
                            numeric_data) const {
    *psi_scalar = std::move(get<CurvedScalarWave::Tags::Psi>(*numeric_data));
    *pi_scalar = std::move(get<CurvedScalarWave::Tags::Pi>(*numeric_data));
    *phi_scalar = std::move(get<CurvedScalarWave::Tags::Phi<3>>(*numeric_data));
  }

  void pup(PUP::er& p) override;

  friend bool operator==(const NumericInitialData& lhs,
                         const NumericInitialData& rhs);

 private:
  importers::ImporterOptions importer_options_{};
  ScalarVars selected_variables_{};
};

namespace Actions {

/*!
 * \brief Dispatch loading numeric initial data from files.
 *
 * Place this action before
 * CurvedScalarWave::Actions::SetNumericInitialData in the action list.
 * See importers::Actions::ReadAllVolumeDataAndDistribute for details, which is
 * invoked by this action.
 */
struct SetInitialData {
  using const_global_cache_tags =
      tmpl::list<evolution::initial_data::Tags::InitialData>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const parallel_component) {
    // Dispatch to the correct `apply` overload based on type of initial data
    using initial_data_classes =
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 evolution::initial_data::InitialData>;
    return call_with_dynamic_type<Parallel::iterable_action_return_t,
                                  initial_data_classes>(
        &db::get<evolution::initial_data::Tags::InitialData>(box),
        [&box, &cache, &array_index,
         &parallel_component](const auto* const initial_data) {
          return apply(make_not_null(&box), *initial_data, cache, array_index,
                       parallel_component);
        });
  }

 private:
  // Numeric initial data
  template <typename DbTagsList, typename Metavariables, typename ArrayIndex,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      const gsl::not_null<db::DataBox<DbTagsList>*> /*box*/,
      const NumericInitialData& initial_data,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const ParallelComponent* const /*meta*/) {
    // Select the subset of the available variables that we want to read from
    // the volume data file
    tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
        importers::Tags::Selected, NumericInitialData::all_vars>>
        selected_fields{};
    initial_data.select_for_import(make_not_null(&selected_fields));
    auto& reader_component = Parallel::get_parallel_component<
        importers::ElementDataReader<Metavariables>>(cache);
    Parallel::simple_action<importers::Actions::ReadAllVolumeDataAndDistribute<
        3, NumericInitialData::all_vars, ParallelComponent>>(
        reader_component, initial_data.importer_options(),
        initial_data.volume_data_id(), std::move(selected_fields));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

  // "AnalyticData"-type initial data
  template <typename DbTagsList, typename InitialData, typename Metavariables,
            typename ArrayIndex, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      const gsl::not_null<db::DataBox<DbTagsList>*> box,
      const InitialData& initial_data,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/,
      const ParallelComponent* const /*meta*/) {
    static constexpr size_t Dim = Metavariables::volume_dim;
    using flat_variables_tag = typename ScalarWave::System<Dim>::variables_tag;
    using curved_variables_tag =
        typename CurvedScalarWave::System<Dim>::variables_tag;
    const auto inertial_coords =
        db::get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(*box);
    const double initial_time = db::get<::Tags::Time>(*box);
    if constexpr (is_analytic_data_v<InitialData> or
                  is_analytic_solution_v<InitialData>) {
      if constexpr (tmpl::list_contains_v<typename InitialData::tags,
                                          CurvedScalarWave::Tags::Psi>) {
        const auto curved_initial_data =
            evolution::Initialization::initial_data(
                initial_data, inertial_coords, initial_time,
                typename curved_variables_tag::tags_list{});

        db::mutate<typename CurvedScalarWave::System<Dim>::variables_tag>(
            [&curved_initial_data](
                const gsl::not_null<typename curved_variables_tag::type*>
                    evolved_vars) {
              evolved_vars->assign_subset(curved_initial_data);
            },
            box);
      } else {
        static_assert(tmpl::list_contains_v<typename InitialData::tags,
                                            ScalarWave::Tags::Psi>,
                      "The initial data class must either calculate ScalarWave "
                      "or CurvedScalarWave variables.");
        const auto flat_initial_data = evolution::Initialization::initial_data(
            initial_data, inertial_coords, initial_time,
            typename flat_variables_tag::tags_list{});
        const auto& shift = db::get<gr::Tags::Shift<DataVector, Dim>>(*box);
        const auto& lapse = db::get<gr::Tags::Lapse<DataVector>>(*box);
        const auto shift_dot_dpsi = dot_product(
            shift, get<ScalarWave::Tags::Phi<Dim>>(flat_initial_data));
        db::mutate<typename CurvedScalarWave::System<Dim>::variables_tag>(
            [&flat_initial_data, &shift_dot_dpsi,
             &lapse](const gsl::not_null<typename curved_variables_tag::type*>
                         evolved_vars) {
              get<CurvedScalarWave::Tags::Psi>(*evolved_vars) =
                  get<ScalarWave::Tags::Psi>(flat_initial_data);
              get<CurvedScalarWave::Tags::Phi<Dim>>(*evolved_vars) =
                  get<ScalarWave::Tags::Phi<Dim>>(flat_initial_data);
              get(get<CurvedScalarWave::Tags::Pi>(*evolved_vars)) =
                  (get(shift_dot_dpsi) +
                   get(get<ScalarWave::Tags::Pi>(flat_initial_data))) /
                  get(lapse);
            },
            box);
      }
    } else {
      ERROR(
          "Trying to use "
          "'evolution::Initialization::Actions::SetVariables' with a "
          "class that's not marked as analytic solution or analytic "
          "data. To support numeric initial data, add a "
          "system-specific initialization routine to your executable.");
    }

    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

/*!
 * \brief Receive numeric initial data loaded by
 * CurvedScalarWave::Actions::ReadNumericInitialData.
 */
struct ReceiveNumericInitialData {
  static constexpr size_t Dim = 3;
  using inbox_tags =
      tmpl::list<importers::Tags::VolumeData<NumericInitialData::all_vars>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    if constexpr (Metavariables::volume_dim != Dim) {
      ERROR(
          "CurvedScalarWave numeric initial data currently requires a 3D "
          "domain.");
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    } else {
      auto& inbox = tuples::get<
          importers::Tags::VolumeData<NumericInitialData::all_vars>>(inboxes);
      const auto& initial_data = dynamic_cast<const NumericInitialData&>(
          db::get<evolution::initial_data::Tags::InitialData>(box));
      const size_t volume_data_id = initial_data.volume_data_id();
      if (inbox.find(volume_data_id) == inbox.end()) {
        return {Parallel::AlgorithmExecution::Retry, std::nullopt};
      }
      auto numeric_data = std::move(inbox.extract(volume_data_id).mapped());

      db::mutate<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
                 CurvedScalarWave::Tags::Phi<Dim>>(
          [&initial_data, &numeric_data](
              const gsl::not_null<Scalar<DataVector>*> psi_scalar,
              const gsl::not_null<Scalar<DataVector>*> pi_scalar,
              const gsl::not_null<tnsr::i<DataVector, Dim>*> phi_scalar) {
            initial_data.set_initial_data(psi_scalar, pi_scalar, phi_scalar,
                                          make_not_null(&numeric_data));
          },
          make_not_null(&box));

      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
  }
};

}  // namespace Actions

}  // namespace CurvedScalarWave
