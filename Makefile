PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=vss
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Stabilize all tests in CI
ifdef CI
TEST_FLAGS:=--stabilize-tests
endif
T ?= $(TEST_FLAGS) "test/*"

# extension-ci-tools does not forward DuckDB's FORCE_DEBUG make variable.
# Translate it to the CMake flag consumed by its release-style build targets.
ifeq (${FORCE_DEBUG}, 1)
EXT_RELEASE_FLAGS += -DFORCE_DEBUG=1
endif

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

unittest_relassert:
	python3 duckdb/scripts/ci/run_tests.py build/relassert/test/unittest $(T)
