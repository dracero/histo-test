# Implementation Plan

- [ ] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Uvicorn Module Not Accessible via uv run
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate the bug exists
  - **Scoped PBT Approach**: For deterministic bugs, scope the property to the concrete failing case(s) to ensure reproducibility
  - Test that executing `npm run dev` successfully starts the uvicorn server without ModuleNotFoundError
  - Test that `uv run uvicorn server:app --reload --host 0.0.0.0 --port 10005` successfully imports and executes the uvicorn module
  - Test that `uv run python -c "import uvicorn; print(uvicorn.__file__)"` successfully imports uvicorn and prints its module path
  - The test assertions should verify: no ModuleNotFoundError occurs, uvicorn module is importable, server starts on port 10005
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS with ModuleNotFoundError (this is correct - it proves the bug exists)
  - Document counterexamples found: specific error messages, module resolution paths, environment state
  - Analyze whether `uv sync` alone resolves the issue by testing before and after sync
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3_

- [ ] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Alternative Execution Methods Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for alternative execution methods that do NOT use `uv run uvicorn`
  - Test 1: Run `npm run start` (uses `uv run python server.py`) on unfixed code, observe it works correctly, document the behavior
  - Test 2: Run `docker compose up -d` on unfixed code, verify containers start successfully, document the behavior
  - Test 3: Run `uv run python server.py` directly on unfixed code, observe server starts, document the behavior
  - Test 4: Run `uv run python -c "import torch; import neo4j; import fastapi; print('OK')"` on unfixed code, verify other dependencies are importable, document the behavior
  - Write property-based tests capturing these observed behavior patterns: alternative startup methods work, docker compose works, other dependencies resolve correctly
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (this confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 3. Fix for uvicorn dependency resolution

  - [ ] 3.1 Implement the fix
    - Add explicit `uv sync` command before running uvicorn in the dev script to ensure virtual environment is up-to-date
    - Modify package.json dev script to: `"dev": "docker compose up -d && uv sync && uv run uvicorn server:app --reload --host 0.0.0.0 --port 10005"`
    - Alternative approach: Use module invocation `uv run python -m uvicorn` instead of `uv run uvicorn` for more reliable module resolution
    - If needed, verify `.venv` directory exists in project root or that uv creates one automatically
    - Ensure the npm script executes in the correct project root context where pyproject.toml is located
    - _Bug_Condition: isBugCondition(input) where input.command == "uv run uvicorn server:app --reload --host 0.0.0.0 --port 10005" AND "uvicorn>=0.30.0" IN input.projectDependencies AND NOT moduleIsAccessible("uvicorn", input.uvEnvironment) AND errorMessage CONTAINS "ModuleNotFoundError: No module named 'uvicorn'"_
    - _Expected_Behavior: For any execution context where npm run dev is invoked with uvicorn declared in pyproject.toml, the fixed configuration SHALL ensure uvicorn is installed in the uv-managed virtual environment, is importable as a Python module, and executes successfully_
    - _Preservation: Alternative execution methods (npm run start, docker compose up, direct Python execution) and dependency resolution for other packages SHALL remain unchanged_
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4_

  - [ ] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Uvicorn Module Accessible via uv run
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - Verify: npm run dev starts server successfully, no ModuleNotFoundError occurs, uvicorn is importable, server listens on port 10005
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Alternative Execution Methods Still Work
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - Verify: npm run start works identically, docker compose works identically, direct Python execution works identically, other dependencies still resolve correctly
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm all alternative execution methods still work as before

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Verify the server starts successfully via `npm run dev` and responds on port 10005
  - Verify alternative methods (npm run start, docker compose) still work correctly
  - Verify other dependencies (torch, neo4j, fastapi) are still resolved correctly
