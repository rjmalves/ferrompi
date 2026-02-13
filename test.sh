#!/bin/bash
# FerroMPI Test Suite — Legacy Wrapper
#
# This script is deprecated. Use tests/run_mpi_tests.sh instead.
#
# Usage: ./test.sh [num_procs]

set -e

NUM_PROCS=${1:-4}

echo "NOTE: test.sh is deprecated. Delegating to tests/run_mpi_tests.sh"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MPI_NP="$NUM_PROCS"
exec "${SCRIPT_DIR}/tests/run_mpi_tests.sh" "${2:-}"
