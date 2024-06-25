// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/StaticMatrix.h>
#include <blaze/math/StaticVector.h>
#include <utility>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/LinearSolver/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearSolver/BuildMatrix.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = TestHelpers::LinearSolver;

namespace {
struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace

namespace LinearSolver::Serial {

SPECTRE_TEST_CASE("Unit.LinearSolver.Serial.BuildMatrix",
                  "[Unit][NumericalAlgorithms][LinearSolver]") {
  {
    INFO("Build a simple dense matrix");
    const blaze::DynamicMatrix<double> matrix{{4., 1.}, {3., 1.}};
    const helpers::ApplyMatrix<double> linear_operator{matrix};
    blaze::DynamicMatrix<double> matrix_representation(2, 2);
    blaze::DynamicVector<double> operand_buffer(2, 0.);
    blaze::DynamicVector<double> result_buffer(2, 0.);
    build_matrix(make_not_null(&matrix_representation),
                 make_not_null(&operand_buffer), make_not_null(&result_buffer),
                 linear_operator);
    CHECK_ITERABLE_APPROX(matrix_representation, matrix);
    CHECK(linear_operator.invocations == 2);
  }
  {
    INFO("Build a simple sparse matrix");
    const blaze::StaticMatrix<double, 2, 2> matrix{{4., 0.}, {3., 0.}};
    size_t operator_invocations = 0;
    const auto linear_operator =
        [&matrix, &operator_invocations](
            const gsl::not_null<blaze::StaticVector<double, 2>*> local_result,
            const blaze::StaticVector<double, 2>& local_operand) {
          *local_result = matrix * local_operand;
          ++operator_invocations;
        };
    blaze::CompressedMatrix<double> matrix_representation(2, 2);
    blaze::StaticVector<double, 2> operand_buffer(0.);
    blaze::StaticVector<double, 2> result_buffer(0.);
    build_matrix(make_not_null(&matrix_representation),
                 make_not_null(&operand_buffer), make_not_null(&result_buffer),
                 linear_operator);
    CHECK(matrix_representation == matrix);
    CHECK(operator_invocations == 2);
    CHECK(matrix_representation.nonZeros() == 2);
  }
  {
    INFO("Build matrix from a heterogeneous data structure");
    using SubdomainData = ::LinearSolver::Schwarz::ElementCenteredSubdomainData<
        1, tmpl::list<ScalarFieldTag>>;

    const Matrix matrix_element{{4., 1., 1.}, {1., 1., 3.}, {0., 2., 0.}};
    const Matrix matrix_overlap{{4., 1.}, {3., 1.}};
    Matrix expected_matrix(5, 5, 0.);
    blaze::submatrix(expected_matrix, 0, 0, 3, 3) = matrix_element;
    blaze::submatrix(expected_matrix, 3, 3, 2, 2) = matrix_overlap;
    const ::LinearSolver::Schwarz::OverlapId<1> overlap_id{
        Direction<1>::lower_xi(), ElementId<1>{0}};
    const std::array<std::reference_wrapper<const Matrix>, 1> matrices_element{
        matrix_element};
    const std::array<std::reference_wrapper<const Matrix>, 1> matrices_overlap{
        matrix_overlap};
    const auto linear_operator = [&matrices_element, &matrices_overlap,
                                  &overlap_id](
                                     const gsl::not_null<SubdomainData*> result,
                                     const SubdomainData& operand) {
      apply_matrices(make_not_null(&result->element_data), matrices_element,
                     operand.element_data, Index<1>{3});
      apply_matrices(make_not_null(&result->overlap_data.at(overlap_id)),
                     matrices_overlap, operand.overlap_data.at(overlap_id),
                     Index<1>{2});
    };

    blaze::DynamicMatrix<double> matrix_representation(5, 5);
    SubdomainData operand_buffer{3};
    get(get<ScalarFieldTag>(operand_buffer.element_data)) = DataVector(3, 0.);
    operand_buffer.overlap_data.emplace(overlap_id,
                                        typename SubdomainData::OverlapData{2});
    get(get<ScalarFieldTag>(operand_buffer.overlap_data.at(overlap_id))) =
        DataVector(2, 0.);
    auto result_buffer = make_with_value<SubdomainData>(operand_buffer, 0.);

    build_matrix(make_not_null(&matrix_representation),
                 make_not_null(&operand_buffer), make_not_null(&result_buffer),
                 linear_operator);
    CHECK_ITERABLE_APPROX(matrix_representation, expected_matrix);
  }
}

}  // namespace LinearSolver::Serial
