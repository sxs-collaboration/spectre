\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Writing Unit Tests {#writing_unit_tests}

## Testing Failure Cases

Adding the "attribute" `// [[OutputRegex, Regular expression to match]]`
before the `SPECTRE_TEST_CASE` macro will force ctest to only pass the particular test
if the regular expression is found. This can be used to test error handling.
When testing `ASSERT`s you must mark the `SPECTRE_TEST_CASE` as `[[noreturn]]`,
add the macro `ASSERTION_TEST();` to the beginning of the test, and also have
the test call `ERROR("Failed to trigger ASSERT in an assertion test");` at the
end of the test body.
For example,

```cpp
// [[OutputRegex, Must copy into same size]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.DataStructures.DataVector.ref_diff_size",
                               "[DataStructures][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  DataVector data{1.43, 2.83, 3.94, 7.85};
  DataVector data_ref;
  data_ref.set_data_ref(data);
  DataVector data2{1.43, 2.83, 3.94};
  data_ref = data2;
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
```

If the `ifdef SPECTRE_DEBUG` is omitted then compilers will correctly flag
the code as being unreachable which results in warnings.
