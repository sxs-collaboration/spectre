// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/FloatingPointType.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
namespace observers::Actions {
struct ContributeVolumeData;
}  // namespace observers::Actions
/// \endcond

namespace TestHelpers::dg::Events::ObserveFields {
struct TestSectionIdTag {};

struct MockContributeVolumeData {
  struct Results {
    observers::ObservationId observation_id{};
    std::string subfile_name{};
    Parallel::ArrayComponentId array_component_id{};
    ElementVolumeData received_volume_data{};
  };
  static Results results;

  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationId& observation_id,
                    const std::string& subfile_name,
                    const Parallel::ArrayComponentId& array_component_id,
                    ElementVolumeData&& received_volume_data) {
    results.observation_id = observation_id;
    results.subfile_name = subfile_name;
    results.array_component_id = array_component_id;
    results.received_volume_data = std::move(received_volume_data);
  }
};

inline MockContributeVolumeData::Results MockContributeVolumeData::results{};

template <typename Metavariables>
struct ElementComponent {
  using component_being_mocked = void;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::system::volume_dim>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

template <typename Metavariables>
struct MockObserverComponent {
  using component_being_mocked = observers::Observer<Metavariables>;
  using replace_these_simple_actions =
      tmpl::list<observers::Actions::ContributeVolumeData>;
  using with_these_simple_actions = tmpl::list<MockContributeVolumeData>;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockGroupChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;
};

template <typename System, bool HasAnalyticSolution>
struct Metavariables {
  using system = System;
  using component_list = tmpl::list<ElementComponent<Metavariables>,
                                    MockObserverComponent<Metavariables>>;
  using const_global_cache_tags =
      tmpl::list<Tags::AnalyticSolution<typename System::solution_for_test>>;
  using initial_data =
      tmpl::conditional_t<HasAnalyticSolution,
                          typename System::solution_for_test, NoSuchType>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Event, tmpl::list<typename system::ObserveEvent>>>;
  };
};

// Helper tags
namespace Tags {
template <typename DataType>
struct ScalarVarTimesTwo : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename ScalarVar>
struct ScalarVarTimesTwoCompute
    : db::ComputeTag,
      ScalarVarTimesTwo<typename ScalarVar::type::type> {
  using DataType = typename ScalarVar::type::type;
  using base = ScalarVarTimesTwo<DataType>;
  using return_type = Scalar<DataType>;
  using argument_tags = tmpl::list<ScalarVar>;
  static void function(const gsl::not_null<Scalar<DataType>*> result,
                       const Scalar<DataType>& scalar_var) {
    get(*result) = 2.0 * get(scalar_var);
  }
};

template <typename DataType>
struct ScalarVarTimesThree : db::SimpleTag {
  using type = Scalar<DataType>;
};

template <typename ScalarVar>
struct ScalarVarTimesThreeCompute
    : db::ComputeTag,
      ::Tags::Variables<
          tmpl::list<ScalarVarTimesThree<typename ScalarVar::type::type>>> {
  using DataType = typename ScalarVar::type::type;
  using base = ScalarVarTimesThree<DataType>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<ScalarVar>;
  static void function(
      const gsl::not_null<
          ::Variables<tmpl::list<ScalarVarTimesThree<DataType>>>*>
          result,
      const Scalar<DataType>& scalar_var) {
    result->initialize(get(scalar_var).size());
    get(get<ScalarVarTimesThree<DataType>>(*result)) = 3.0 * get(scalar_var);
  }
};
}  // namespace Tags

// Test systems

template <template <size_t, class...> class ObservationEvent,
          typename ArraySectionId = void, typename DataType = DataVector>
struct ScalarSystem {
  using array_section_id = ArraySectionId;
  using data_type = DataType;

  struct ScalarVar : db::SimpleTag {
    static std::string name() { return "Scalar"; }
    using type = Scalar<DataType>;
  };

  static constexpr size_t volume_dim = 1;
  using variables_tag = ::Tags::Variables<tmpl::list<ScalarVar>>;

  template <typename CheckComponent>
  static void check_data(const CheckComponent& check_component) {
    check_component("Scalar", ScalarVar{});
    check_component("ScalarVarTimesTwo", Tags::ScalarVarTimesTwo<DataType>{});
    check_component("ScalarVarTimesThree",
                    Tags::ScalarVarTimesThree<DataType>{});
  }

  using all_vars_for_test = tmpl::list<ScalarVar>;
  struct solution_for_test : public MarkAsAnalyticSolution {
    using vars_for_test = typename variables_tag::tags_list;

    template <typename CheckComponent>
    static void check_data(const CheckComponent& check_component) {
      check_component("Error(Scalar)", ScalarVar{});
    }
    static tuples::tagged_tuple_from_typelist<vars_for_test> variables(

        const tnsr::I<DataVector, 1>& x, const double t,
        const vars_for_test /*meta*/) {
      return {Scalar<DataType>{1.0 - t * get<0>(x)}};
    }

    void pup(PUP::er& /*p*/) {}  // NOLINT
  };

  using ObserveEvent = ObservationEvent<
      volume_dim,
      tmpl::push_back<all_vars_for_test,
                      Tags::ScalarVarTimesTwoCompute<ScalarVar>,
                      Tags::ScalarVarTimesThree<DataType>,
                      ::domain::Tags::Coordinates<volume_dim, Frame::Inertial>,
                      ::Tags::Error<ScalarVar>>,
      tmpl::list<Tags::ScalarVarTimesThreeCompute<ScalarVar>,
                 ::Tags::ErrorsCompute<tmpl::list<ScalarVar>>>,
      ArraySectionId>;
  static constexpr auto creation_string_for_test =
      "ObserveFields:\n"
      "  SubfileName: element_data\n"
      "  CoordinatesFloatingPointType: Double\n"
      "  VariablesToObserve: [Scalar, ScalarVarTimesTwo, ScalarVarTimesThree, "
      "Error(Scalar)]\n"
      "  FloatingPointTypes: [Double]\n"
      "  BlocksToObserve: All\n";
  static ObserveEvent make_test_object(
      const std::optional<Mesh<volume_dim>>& interpolating_mesh,
      std::optional<std::vector<std::string>> active_block_or_block_groups =
          std::nullopt) {
    return ObserveEvent{
        "element_data",
        FloatingPointType::Double,
        {FloatingPointType::Double},
        {"Scalar", "ScalarVarTimesTwo", "ScalarVarTimesThree", "Error(Scalar)"},
        std::move(active_block_or_block_groups),
        interpolating_mesh};
  }
};

template <template <size_t, class...> class ObservationEvent,
          typename ArraySectionId = void, typename DataType = DataVector>
struct ComplicatedSystem {
  using array_section_id = ArraySectionId;
  using data_type = DataType;

  struct ScalarVar : db::SimpleTag {
    static std::string name() { return "Scalar"; }
    using type = Scalar<DataType>;
  };

  struct VectorVar : db::SimpleTag {
    static std::string name() { return "Vector"; }
    using type = tnsr::I<DataType, 2>;
  };

  struct TensorVar : db::SimpleTag {
    static std::string name() { return "Tensor"; }
    using type = tnsr::ii<DataType, 2>;
  };

  struct TensorVar2 : db::SimpleTag {
    static std::string name() { return "Tensor2"; }
    using type = tnsr::ii<DataType, 2>;
  };

  struct UnobservedVar : db::SimpleTag {
    static std::string name() { return "Unobserved"; }
    using type = Scalar<DataType>;
  };

  struct UnobservedVar2 : db::SimpleTag {
    static std::string name() { return "Unobserved2"; }
    using type = Scalar<DataType>;
  };

  static constexpr size_t volume_dim = 2;
  using variables_tag =
      ::Tags::Variables<tmpl::list<TensorVar, ScalarVar, UnobservedVar>>;
  using primitive_variables_tag =
      ::Tags::Variables<tmpl::list<VectorVar, TensorVar2, UnobservedVar2>>;

  template <typename CheckComponent>
  static void check_data(const CheckComponent& check_component) {
    check_component("Scalar", ScalarVar{});
    check_component("ScalarVarTimesTwo", Tags::ScalarVarTimesTwo<DataType>{});
    check_component("ScalarVarTimesThree",
                    Tags::ScalarVarTimesThree<DataType>{});
    check_component("Tensor_xx", TensorVar{}, 0, 0);
    check_component("Tensor_yx", TensorVar{}, 0, 1);
    check_component("Tensor_yy", TensorVar{}, 1, 1);
    check_component("Vector_x", VectorVar{}, 0);
    check_component("Vector_y", VectorVar{}, 1);
    check_component("Tensor2_xx", TensorVar2{}, 0, 0);
    check_component("Tensor2_yx", TensorVar2{}, 0, 1);
    check_component("Tensor2_yy", TensorVar2{}, 1, 1);
  }

  using all_vars_for_test = tmpl::list<TensorVar, ScalarVar, UnobservedVar,
                                       VectorVar, TensorVar2, UnobservedVar2>;
  struct solution_for_test : public MarkAsAnalyticSolution {
    using vars_for_test = typename primitive_variables_tag::tags_list;

    template <typename CheckComponent>
    static void check_data(const CheckComponent& check_component) {
      check_component("Error(Vector)_x", VectorVar{}, 0);
      check_component("Error(Vector)_y", VectorVar{}, 1);
      check_component("Error(Tensor2)_xx", TensorVar2{}, 0, 0);
      check_component("Error(Tensor2)_yx", TensorVar2{}, 0, 1);
      check_component("Error(Tensor2)_yy", TensorVar2{}, 1, 1);
    }

    static tuples::tagged_tuple_from_typelist<vars_for_test> variables(
        const tnsr::I<DataVector, 2>& x, const double t,
        const vars_for_test /*meta*/) {
      auto vector = make_with_value<tnsr::I<DataVector, 2>>(x, 0.0);
      auto tensor = make_with_value<tnsr::ii<DataVector, 2>>(x, 0.0);
      auto unobserved = make_with_value<Scalar<DataVector>>(x, 0.0);
      // Arbitrary functions
      get<0>(vector) = 1.0 - t * get<0>(x);
      get<1>(vector) = 1.0 - t * get<1>(x);
      get<0, 0>(tensor) = get<0>(x) + get<1>(x);
      get<0, 1>(tensor) = get<0>(x) - get<1>(x);
      get<1, 1>(tensor) = get<0>(x) * get<1>(x);
      get(unobserved) = 2.0 * get<0>(x);
      return {std::move(vector), std::move(tensor), std::move(unobserved)};
    }

    void pup(PUP::er& /*p*/) {}  // NOLINT
  };

  using ObserveEvent = ObservationEvent<
      volume_dim,
      tmpl::push_back<all_vars_for_test,
                      Tags::ScalarVarTimesTwoCompute<ScalarVar>,
                      Tags::ScalarVarTimesThree<DataType>,
                      ::domain::Tags::Coordinates<volume_dim, Frame::Inertial>,
                      ::Tags::Error<VectorVar>, ::Tags::Error<TensorVar2>>,
      tmpl::list<
          Tags::ScalarVarTimesThreeCompute<ScalarVar>,
          ::Tags::ErrorsCompute<typename primitive_variables_tag::tags_list>>,
      ArraySectionId>;
  static constexpr auto creation_string_for_test =
      "ObserveFields:\n"
      "  SubfileName: element_data\n"
      "  CoordinatesFloatingPointType: Double\n"
      "  VariablesToObserve: [Scalar, ScalarVarTimesTwo, ScalarVarTimesThree,"
      "                       Vector, Tensor, Tensor2,"
      "                       Error(Vector), Error(Tensor2)]\n"
      "  FloatingPointTypes: [Double, Double, Double, Double, Float, Float,"
      "                       Double, Float]\n"
      "  BlocksToObserve: All\n";

  static ObserveEvent make_test_object(
      const std::optional<Mesh<volume_dim>>& interpolating_mesh,
      std::optional<std::vector<std::string>> active_block_or_block_groups =
          std::nullopt) {
    return ObserveEvent(
        "element_data", FloatingPointType::Double,
        {FloatingPointType::Double, FloatingPointType::Double,
         FloatingPointType::Double, FloatingPointType::Double,
         FloatingPointType::Float, FloatingPointType::Float,
         FloatingPointType::Double, FloatingPointType::Float},
        {"Scalar", "ScalarVarTimesTwo", "ScalarVarTimesThree", "Vector",
         "Tensor", "Tensor2", "Error(Vector)", "Error(Tensor2)"},
        std::move(active_block_or_block_groups), interpolating_mesh);
  }
};
}  // namespace TestHelpers::dg::Events::ObserveFields
