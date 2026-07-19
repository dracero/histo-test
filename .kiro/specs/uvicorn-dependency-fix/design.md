# Uvicorn Dependency Fix Bugfix Design

## Overview

This bugfix addresses a dependency resolution issue where the `npm run dev` command fails to start the FastAPI development server. Although `uvicorn>=0.30.0` is correctly declared in `pyproject.toml`, the module is not accessible when executing `uv run uvicorn server:app --reload --host 0.0.0.0 --port 10005`. The bug manifests as a ModuleNotFoundError, preventing the development server from starting. The fix will ensure that uvicorn is properly installed and accessible in the uv-managed virtual environment without disrupting other dependency resolution mechanisms or alternative startup methods.

## Glossary

- **Bug_Condition (C)**: The condition where `uv run uvicorn` fails to find the uvicorn module despite it being declared in pyproject.toml dependencies
- **Property (P)**: The desired behavior where uvicorn is successfully installed, resolvable, and executable within the uv-managed environment
- **Preservation**: Existing docker compose startup, alternative start script (`npm run start` with `python server.py`), and dependency resolution for other packages must remain unchanged
- **uv**: A fast Python package installer and resolver used in this project to manage dependencies
- **pyproject.toml**: Python project configuration file containing dependency declarations
- **package.json**: Node.js package file containing npm scripts for development workflow
- **Virtual Environment**: Python isolated environment where project dependencies are installed

## Bug Details

### Bug Condition

The bug manifests when the development script attempts to start the FastAPI server using uvicorn via the uv package manager. The `uv run uvicorn` command fails to locate the uvicorn module even though the dependency is correctly specified in the project configuration. The error occurs during the npm script execution, specifically when uv attempts to run uvicorn after docker compose has started successfully.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type ExecutionContext
  OUTPUT: boolean
  
  RETURN input.command == "uv run uvicorn server:app --reload --host 0.0.0.0 --port 10005"
         AND "uvicorn>=0.30.0" IN input.projectDependencies
         AND NOT moduleIsAccessible("uvicorn", input.uvEnvironment)
         AND errorMessage CONTAINS "ModuleNotFoundError: No module named 'uvicorn'"
END FUNCTION
```

### Examples

- **Example 1**: User runs `npm run dev` → docker compose starts successfully → uv run uvicorn attempts to execute → **Actual**: "ModuleNotFoundError: No module named 'uvicorn'" | **Expected**: Server starts and listens on port 10005

- **Example 2**: User runs `uv run uvicorn server:app --reload --host 0.0.0.0 --port 10005` directly → uv reinstalls packages → **Actual**: uvicorn not found in /home/dracero/.local/bin/uvicorn | **Expected**: uvicorn is found in the uv virtual environment and executes successfully

- **Example 3**: User checks uv environment → pyproject.toml contains uvicorn>=0.30.0 → **Actual**: Module is not accessible at runtime | **Expected**: Module is installed and importable

- **Edge case**: User has uvicorn installed globally on the system → **Expected**: The uv-managed project environment should have its own isolated uvicorn installation, not relying on global packages

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Docker compose startup via `docker compose up -d` must continue to work exactly as before
- The alternative start script `npm run start` using `uv run python server.py` must continue to function
- Resolution of other dependencies from pyproject.toml (torch, neo4j, fastapi, etc.) must remain unchanged
- Custom index configuration for pytorch packages (`tool.uv.index` pointing to pytorch-cu128) must continue to work

**Scope:**
All execution contexts that do NOT involve the specific `uv run uvicorn` command should be completely unaffected by this fix. This includes:
- Direct Python execution via `python server.py` or `uv run python server.py`
- Docker container operations
- Installation and usage of other project dependencies
- Alternative uvicorn invocation methods (if added in the future)

## Hypothesized Root Cause

Based on the bug description and error logs, the most likely issues are:

1. **Incorrect uv Sync State**: The uv virtual environment may not be properly synchronized with the pyproject.toml dependencies
   - The uvicorn package might be installed in an unexpected location
   - The virtual environment might not be activated or recognized when `uv run` executes
   - There may be a stale lock file or cache causing resolution issues

2. **CLI Binary vs Module Import Issue**: uv might be looking for the uvicorn CLI binary instead of ensuring the module is importable
   - The error suggests a module import failure rather than a binary not found error
   - The uvicorn package may be installed but not properly registered in the environment's site-packages

3. **Path Resolution Problem**: The command `uv run uvicorn` may not be correctly resolving the module path
   - uv might be searching in /home/dracero/.local/bin/ (global user location) instead of the project's virtual environment
   - The virtual environment location might not be correctly configured in the project

4. **Package.json Script Execution Context**: The script may be executing in an environment where uv doesn't have proper context
   - npm script execution may not preserve necessary environment variables
   - The working directory or PATH might not be correctly set when uv run executes

## Correctness Properties

Property 1: Bug Condition - Uvicorn Module Accessibility

_For any_ execution context where the `npm run dev` command is invoked with uvicorn declared in pyproject.toml dependencies, the fixed configuration SHALL ensure that uvicorn is installed in the uv-managed virtual environment, is importable as a Python module, and executes successfully when invoked via `uv run uvicorn server:app --reload --host 0.0.0.0 --port 10005`.

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Preservation - Alternative Execution Methods

_For any_ execution context that does NOT use the `uv run uvicorn` command (including `npm run start`, `docker compose up`, direct Python execution, and other dependency installations), the fixed configuration SHALL produce exactly the same behavior as the original configuration, preserving all existing functionality for alternative startup methods and dependency resolution mechanisms.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `package.json`

**Script**: `dev`

**Specific Changes**:
1. **Ensure Virtual Environment Sync**: Modify the dev script to explicitly sync dependencies before running uvicorn
   - Add `uv sync` before the uvicorn command to ensure the virtual environment is up-to-date
   - This ensures all pyproject.toml dependencies are installed in the project's virtual environment

2. **Verify Environment Context**: Ensure uv uses the project's virtual environment rather than global paths
   - Verify that a `.venv` directory exists in the project root or that uv creates one
   - Consider adding explicit environment activation if needed

3. **Alternative: Use Module Invocation**: Change from `uv run uvicorn` to `uv run python -m uvicorn`
   - This explicitly invokes uvicorn as a Python module rather than as a CLI tool
   - Ensures the module import path is used, which is more reliable than binary path resolution

4. **Add Dependency Cache Refresh**: If the issue is related to stale cache or lock files
   - Consider adding `uv lock --refresh` to ensure dependencies are correctly resolved
   - This may be necessary if the uv.lock file is out of sync with pyproject.toml

5. **Verify Working Directory**: Ensure the npm script executes in the correct project root context
   - npm scripts should already run in the package.json directory, but verify no cd commands are interfering
   - Ensure uv can find pyproject.toml from the execution directory

**Most Likely Fix Approach**:
The primary fix will likely be adding an explicit `uv sync` step before running uvicorn, and potentially switching to module invocation (`python -m uvicorn`) for more reliable module resolution:

```json
"dev": "docker compose up -d && uv sync && uv run python -m uvicorn server:app --reload --host 0.0.0.0 --port 10005"
```

Or if the CLI invocation must be preserved:

```json
"dev": "docker compose up -d && uv sync && uv run uvicorn server:app --reload --host 0.0.0.0 --port 10005"
```

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate the bug on unfixed code, then verify the fix works correctly and preserves existing behavior.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate the bug BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Execute the development command and related uv operations on the UNFIXED code to observe failures and understand the exact failure mode. Document the specific error messages, module resolution paths, and environment state.

**Test Cases**:
1. **Basic Dev Script Execution**: Run `npm run dev` and observe the ModuleNotFoundError (will fail on unfixed code)
2. **Direct uv run uvicorn**: Execute `uv run uvicorn server:app --reload --host 0.0.0.0 --port 10005` directly (will fail on unfixed code)
3. **Environment Inspection**: Run `uv run python -c "import uvicorn; print(uvicorn.__file__)"` to check if module is importable (will fail on unfixed code)
4. **Sync State Check**: Run `uv sync` then immediately `uv run python -c "import uvicorn"` to verify if sync alone resolves the issue (may pass, indicating root cause)

**Expected Counterexamples**:
- The ModuleNotFoundError indicates uvicorn is not in the Python module search path when uv run executes
- Possible causes: missing sync step, incorrect environment activation, wrong module resolution path, stale cache

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL executionContext WHERE isBugCondition(executionContext) DO
  result := executeDevScript_fixed(executionContext)
  ASSERT result.serverStarted == true
  ASSERT result.uvicornImportable == true
  ASSERT result.listenPort == 10005
  ASSERT result.reloadEnabled == true
END FOR
```

**Concrete Test Cases**:
1. **Fresh Environment Test**: Delete `.venv` directory, run `npm run dev`, verify server starts successfully
2. **Repeated Execution Test**: Run `npm run dev` multiple times, verify consistent success
3. **Module Import Test**: After fix, run `uv run python -c "import uvicorn; print('OK')"`, verify "OK" is printed
4. **Server Functionality Test**: After server starts, send HTTP request to `http://localhost:10005/api/status`, verify response

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL executionContext WHERE NOT isBugCondition(executionContext) DO
  ASSERT executeScript_original(executionContext) == executeScript_fixed(executionContext)
END FOR
```

**Testing Approach**: Manual testing is sufficient for preservation checking in this case because:
- The number of alternative execution methods is small and well-defined
- Each method can be tested individually with clear success criteria
- The changes are limited to the dev script and shouldn't affect other scripts
- Docker compose and python execution are isolated from uv's uvicorn invocation

**Test Plan**: Verify behavior of alternative methods on UNFIXED code works correctly, then verify the same behavior continues after fix.

**Test Cases**:
1. **Alternative Start Script Preservation**: Run `npm run start` (uses `uv run python server.py`) on unfixed code, observe it works. Then run on fixed code, verify identical behavior.
2. **Docker Compose Preservation**: Run `docker compose up -d` on unfixed code, verify containers start. Then run on fixed code, verify identical container startup.
3. **Direct Python Execution Preservation**: Run `uv run python server.py` on unfixed code, observe behavior. Then run on fixed code, verify identical behavior.
4. **Other Dependencies Preservation**: Run `uv run python -c "import torch; import neo4j; import fastapi; print('OK')"` on both unfixed and fixed code, verify identical success.
5. **PyTorch Custom Index Preservation**: Verify torch is still resolved from the pytorch-cu128 index after fix (check `uv.lock` file or package metadata).

### Unit Tests

- Test that `uv sync` successfully installs uvicorn in the virtual environment
- Test that uvicorn module is importable after sync
- Test that the uvicorn CLI is accessible via `uv run uvicorn --version`
- Test that the server module (server:app) is correctly loaded by uvicorn

### Property-Based Tests

Property-based testing is not necessary for this bugfix because:
- The bug is deterministic and environment-specific, not input-driven
- There are no variable inputs to generate; the command and environment are fixed
- Manual and integration tests provide sufficient coverage for this type of configuration bug

### Integration Tests

- Test full dev workflow: `npm run dev` → verify docker containers start → verify uvicorn starts → verify server responds on port 10005
- Test server reload functionality: start dev server → modify server.py → verify uvicorn auto-reloads
- Test full start workflow: `npm run start` → verify docker containers start → verify python server starts
- Test API endpoints: start server via dev script → test `/api/status` → test `/api/temario` → verify all responses are correct
