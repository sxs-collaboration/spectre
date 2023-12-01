// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "PointwiseFunctions/Elasticity/Strain.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"

namespace Xcts {

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Xcts.LongitudinalOperator",
                  "[Unit][PointwiseFunctions]") {
  {
    INFO("Regression test with Python");
    pypp::SetupLocalPythonEnvironment local_python_env{
        "PointwiseFunctions/Xcts"};
    const DataVector used_for_size{5};
    pypp::check_with_random_values<1>(
        static_cast<void (*)(gsl::not_null<tnsr::II<DataVector, 3>*>,
                             const tnsr::ii<DataVector, 3>&,
                             const tnsr::II<DataVector, 3>&)>(
            &Xcts::longitudinal_operator<DataVector>),
        "LongitudinalOperator", {"longitudinal_operator"}, {{{-1., 1.}}},
        used_for_size);
    pypp::check_with_random_values<1>(
        static_cast<void (*)(gsl::not_null<tnsr::II<DataVector, 3>*>,
                             const tnsr::ii<DataVector, 3>&)>(
            &Xcts::longitudinal_operator_flat_cartesian<DataVector>),
        "LongitudinalOperator", {"longitudinal_operator_flat_cartesian"},
        {{{-1., 1.}}}, used_for_size);
  }
  {
    INFO("Test equivalence of overloads");
    MAKE_GENERATOR(generator);
    const DataVector used_for_size(5);
    std::uniform_real_distribution<> dist{0., 1.};
    const auto spatial_metric =
        TestHelpers::gr::random_spatial_metric<3, DataVector, Frame::Inertial>(
            make_not_null(&generator), used_for_size);
    const auto inv_spatial_metric =
        determinant_and_inverse(spatial_metric).second;
    const auto shift = make_with_random_values<tnsr::I<DataVector, 3>>(
        make_not_null(&generator), make_not_null(&dist), used_for_size);

    const auto deriv_spatial_metric =
        make_with_random_values<tnsr::ijj<DataVector, 3>>(
            make_not_null(&generator), make_not_null(&dist),
            used_for_size);
    const auto deriv_shift = make_with_random_values<tnsr::iJ<DataVector, 3>>(
        make_not_null(&generator), make_not_null(&dist), used_for_size);

    const auto christoffel_first_kind =
        ::gr::christoffel_first_kind(deriv_spatial_metric);
    const auto christoffel_second_kind =
        ::gr::christoffel_second_kind(deriv_spatial_metric, inv_spatial_metric);

    tnsr::ii<DataVector, 3> strain{used_for_size.size()};
    Elasticity::strain(make_not_null(&strain), deriv_shift, spatial_metric,
                       deriv_spatial_metric, christoffel_first_kind, shift);
    tnsr::II<DataVector, 3> result1{used_for_size.size()};
    longitudinal_operator(make_not_null(&result1), strain, inv_spatial_metric);
    tnsr::II<DataVector, 3> result2{used_for_size.size()};
    longitudinal_operator(make_not_null(&result2), shift, deriv_shift,
                          inv_spatial_metric, christoffel_second_kind);
    CHECK_ITERABLE_APPROX(result1, result2);
  }
}

}  // namespace Xcts
