// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <string>
#include <variant>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Initialization/InitialData.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
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
#include "Utilities/TaggedTuple.hpp"

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
      tmpl::list<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi>;
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
        "Set of initial data variables from which the Valencia evolution "
        "variables are computed.";
  };

  using options =
      tmpl::list<importers::OptionTags::FileGlob,
                 importers::OptionTags::Subgroup,
                 importers::OptionTags::ObservationValue,
                 importers::OptionTags::EnableInterpolation, Variables>;

  static constexpr Options::String help =
      "Numeric initial data loaded from volume data files";

  NumericInitialData() = default;
  NumericInitialData(const NumericInitialData& rhs) = default;
  NumericInitialData& operator=(const NumericInitialData& rhs) = default;
  NumericInitialData(NumericInitialData&& /*rhs*/) = default;
  NumericInitialData& operator=(NumericInitialData&& /*rhs*/) = default;
  ~NumericInitialData() = default;

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
      bool enable_interpolation, ScalarVars selected_variables);

  const importers::ImporterOptions& importer_options() const {
    return importer_options_;
  }

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
  void set_initial_data(
      const gsl::not_null<Scalar<DataVector>*> psi_scalar,
      const gsl::not_null<Scalar<DataVector>*> pi_scalar,
      const gsl::not_null<tnsr::i<DataVector, 3>*> phi_scalar,
      const gsl::not_null<tuples::TaggedTuple<AllTags...>*> numeric_data,
      const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian) const {
    *psi_scalar = std::move(get<CurvedScalarWave::Tags::Psi>(*numeric_data));
    *pi_scalar = std::move(get<CurvedScalarWave::Tags::Pi>(*numeric_data));
    // Set Phi to the numerical spatial derivative of the scalar
    partial_derivative(phi_scalar, *psi_scalar, mesh, inv_jacobian);
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
struct ReadNumericInitialData {
  using const_global_cache_tags =
      tmpl::list<evolution::initial_data::Tags::InitialData>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    // Select the subset of the available variables that we want to read from
    // the volume data file
    const auto& initial_data = dynamic_cast<const NumericInitialData&>(
        db::get<evolution::initial_data::Tags::InitialData>(box));
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
};

/*!
 * \brief Receive numeric initial data loaded by
 * CurvedScalarWave::Actions::ReadNumericInitialData.
 */
struct SetNumericInitialData {
  static constexpr size_t Dim = 3;
  using inbox_tags =
      tmpl::list<importers::Tags::VolumeData<NumericInitialData::all_vars>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*element_id*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& inbox =
        tuples::get<importers::Tags::VolumeData<NumericInitialData::all_vars>>(
            inboxes);
    const auto& initial_data = dynamic_cast<const NumericInitialData&>(
        db::get<evolution::initial_data::Tags::InitialData>(box));
    const size_t volume_data_id = initial_data.volume_data_id();
    if (inbox.find(volume_data_id) == inbox.end()) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }
    auto numeric_data = std::move(inbox.extract(volume_data_id).mapped());

    const auto& mesh = db::get<domain::Tags::Mesh<Dim>>(box);
    const auto& inv_jacobian =
        db::get<domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                              Frame::Inertial>>(box);

    db::mutate<CurvedScalarWave::Tags::Psi, CurvedScalarWave::Tags::Pi,
               CurvedScalarWave::Tags::Phi<3>>(
        [&initial_data, &numeric_data, &mesh, &inv_jacobian](
            const gsl::not_null<Scalar<DataVector>*> psi_scalar,
            const gsl::not_null<Scalar<DataVector>*> pi_scalar,
            const gsl::not_null<tnsr::i<DataVector, 3>*> phi_scalar) {
          initial_data.set_initial_data(psi_scalar, pi_scalar, phi_scalar,
                                        make_not_null(&numeric_data), mesh,
                                        inv_jacobian);
        },
        make_not_null(&box));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace Actions

}  // namespace CurvedScalarWave
