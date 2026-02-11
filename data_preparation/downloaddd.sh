#!/usr/bin/env bash
set -euo pipefail

ITERATIONS=25

for ((i=1; i<=ITERATIONS; i++)); do

    echo "Iteration $i of $ITERATIONS"

    sleep 1    

    # Keep looping even if tokenization fails.
    if python tokenize_jagged.py; then
        :
    else
        status=$?
        echo "WARN: tokenize_jagged.py failed with exit code ${status} (iteration ${i}/${ITERATIONS}); continuing..." >&2
    fi

    sleep 900

done
