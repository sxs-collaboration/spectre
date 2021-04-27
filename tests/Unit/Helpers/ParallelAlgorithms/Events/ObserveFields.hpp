// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/NoSuchType.hpp"
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
struct ObservationTimeTag : db::SimpleTag {
  using type = double;
};

struct MockContributeVolumeData {
  struct Results {
    observers::ObservationId observation_id{};
    std::string subfile_name{};
    observers::ArrayComponentId array_component_id{};
    std::vector<TensorComponent> in_received_tensor_data{};
    std::vector<size_t> received_extents{};
    std::vector<Spectral::Basis> received_basis{};
    std::vector<Spectral::Quadrature> received_quadrature{};
  };
  static Results results;

  template <typename ParallelComponent, typename... DbTags,
            typename Metavariables, typename ArrayIndex, size_t Dim>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ObservationId& observation_id,
                    const std::string& subfile_name,
                    const observers::ArrayComponentId& array_component_id,
                    std::vector<TensorComponent>&& in_received_tensor_data,
                    const Index<Dim>& received_extents,
                    const std::array<Spectral::Basis, Dim>& received_basis,
                    const std::array<Spectral::Quadrature, Dim>&
                        received_quadrature) noexcept {
    results.observation_id = observation_id;
    results.subfile_name = subfile_name;
    results.array_component_id = array_component_id;
    results.in_received_tensor_data = in_received_tensor_data;
    results.received_extents.assign(received_extents.indices().begin(),
                                    received_extents.indices().end());
    results.received_basis.assign(received_basis.begin(), received_basis.end());
    results.received_quadrature.assign(received_quadrature.begin(),
                                       received_quadrature.end());
  }
};

inline MockContributeVolumeData::Results MockContributeVolumeData::results{};

template <typename Metavariables>
struct ElementComponent {
  using component_being_mocked = void;

  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::system::volume_dim>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
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
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
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
  enum class Phase { Initialization, Testing, Exit };
};

// Test systems

template <template <size_t, class...> class ObservationEvent>
struct ScalarSystem {
  struct ScalarVar : db::SimpleTag {
    static std::string name() noexcept { return "Scalar"; }
    using type = Scalar<DataVector>;
  };

  static constexpr size_t volume_dim = 1;
  using variables_tag = Tags::Variables<tmpl::list<ScalarVar>>;

  template <typename CheckComponent>
  static void check_data(const CheckComponent& check_component) noexcept {
    check_component("Scalar", ScalarVar{});
  }

  using all_vars_for_test = tmpl::list<ScalarVar>;
  struct solution_for_test : public MarkAsAnalyticSolution {
    using vars_for_test = typename variables_tag::tags_list;

    template <typename CheckComponent>
    static void check_data(const CheckComponent& check_component) noexcept {
      check_component("Error(Scalar)", ScalarVar{});
    }
    static tuples::tagged_tuple_from_typelist<vars_for_test> variables(

        const tnsr::I<DataVector, 1>& x, const double t,
        const vars_for_test /*meta*/) noexcept {
      return {Scalar<DataVector>{1.0 - t * get<0>(x)}};
    }

    void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
  };

  using ObserveEvent =
      ObservationEvent<volume_dim, ObservationTimeTag, all_vars_for_test,
                       typename solution_for_test::vars_for_test>;
  static constexpr auto creation_string_for_test =
      "ObserveFields:\n"
      "  SubfileName: element_data\n"
      "  VariablesToObserve: [Scalar]\n";
  static ObserveEvent make_test_object(
      const std::optional<Mesh<volume_dim>>& interpolating_mesh) noexcept {
    return ObserveEvent{"element_data", {"Scalar"}, interpolating_mesh};
  }
};

template <template <size_t, class...> class ObservationEvent>
struct ComplicatedSystem {
  struct ScalarVar : db::SimpleTag {
    static std::string name() noexcept { return "Scalar"; }
    using type = Scalar<DataVector>;
  };

  struct VectorVar : db::SimpleTag {
    static std::string name() noexcept { return "Vector"; }
    using type = tnsr::I<DataVector, 2>;
  };

  struct TensorVar : db::SimpleTag {
    static std::string name() noexcept { return "Tensor"; }
    using type = tnsr::ii<DataVector, 2>;
  };

  struct TensorVar2 : db::SimpleTag {
    static std::string name() noexcept { return "Tensor2"; }
    using type = tnsr::ii<DataVector, 2>;
  };

  struct UnobservedVar : db::SimpleTag {
    static std::string name() noexcept { return "Unobserved"; }
    using type = Scalar<DataVector>;
  };

  struct UnobservedVar2 : db::SimpleTag {
    static std::string name() noexcept { return "Unobserved2"; }
    using type = Scalar<DataVector>;
  };

  static constexpr size_t volume_dim = 2;
  using variables_tag =
      Tags::Variables<tmpl::list<TensorVar, ScalarVar, UnobservedVar>>;
  using primitive_variables_tag =
      Tags::Variables<tmpl::list<VectorVar, TensorVar2, UnobservedVar2>>;

  template <typename CheckComponent>
  static void check_data(const CheckComponent& check_component) noexcept {
    check_component("Scalar", ScalarVar{});
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
    static void check_data(const CheckComponent& check_component) noexcept {
      check_component("Error(Vector)_x", VectorVar{}, 0);
      check_component("Error(Vector)_y", VectorVar{}, 1);
      check_component("Error(Tensor2)_xx", TensorVar2{}, 0, 0);
      check_component("Error(Tensor2)_yx", TensorVar2{}, 0, 1);
      check_component("Error(Tensor2)_yy", TensorVar2{}, 1, 1);
    }

    static tuples::tagged_tuple_from_typelist<vars_for_test> variables(
        const tnsr::I<DataVector, 2>& x, const double t,
        const vars_for_test /*meta*/) noexcept {
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

    void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
  };

  using ObserveEvent =
      ObservationEvent<volume_dim, ObservationTimeTag, all_vars_for_test,
                       typename solution_for_test::vars_for_test>;
  static constexpr auto creation_string_for_test =
      "ObserveFields:\n"
      "  SubfileName: element_data\n"
      "  VariablesToObserve: [Scalar, Vector, Tensor, Tensor2]\n";

  static ObserveEvent make_test_object(
      const std::optional<Mesh<volume_dim>>& interpolating_mesh) noexcept {
    return ObserveEvent("element_data",
                        {"Scalar", "Vector", "Tensor", "Tensor2"},
                        interpolating_mesh);
  }
};
}  // namespace TestHelpers::dg::Events::ObserveFields
