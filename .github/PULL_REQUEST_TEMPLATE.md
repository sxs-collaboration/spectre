## Proposed changes

<!--
At a high level, describe what this PR does.
-->

### Types of changes:

- [ ] Bugfix
- [ ] New feature

### Component:

- [ ] Code
- [ ] Documentation
- [ ] Build system
- [ ] Continuous integration

### Code review checklist

- [ ] Follows [code review guidelines](https://sxs-collaboration.github.io/spectre/code_review_guide.html)
- [ ] Code has documentation and unit tests
- [ ] Private member variables have a trailing underscore
- [ ] Do not use [Hungarian notation](https://en.wikipedia.org/wiki/Hungarian_notation), e.g. `double* pd_blah` is bad
- [ ] Header order:
  1. hpp corresponding to cpp (only in cpp files)
  2. Blank line (only in cpp files)
  3. STL and externals (in alphabetical order)
  4. Blank line
  5. SpECTRE includes (in alphabetical order)
- [ ] File lists in CMake are alphabetical
- [ ] Correct `noexcept` specification for functions (if unsure, mark `noexcept`)
- [ ] Mark objects `const` whenever possible
- [ ] Almost always `auto`, except with expression templates, i.e. `DataVector`
- [ ] All commits for performance changes provide quantitative evidence and the tests used to obtain said evidence.
- [ ] Make sure error messages are helpful, e.g. "The number of grid points in the matrix 'F' is not the same as the number of grid points in the determinant."
- [ ] Prefix commits addressing PR requests with `fixup`


### Further comments

<!--
If this is a relatively large or complex change, kick off the discussion by explaining why you chose the solution you did and what alternatives you considered, etc...
-->
