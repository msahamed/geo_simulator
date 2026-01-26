#!/bin/bash

###############################################################################
# Geodynamic Simulation Runner
#
# Usage: ./run_simulation.sh [config_file] [options]
#
# Options:
#   --no-viz        Skip visualization step
#   --clean-only    Only clean output directory, don't run simulation
#   --continue      Continue from existing output (don't clean)
###############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
CONFIG_FILE="${1:-inputs/core_complex_2d/config_des3d_style.toml}"
RUN_VIZ=true
CLEAN_OUTPUT=true
RUN_SIM=true

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --no-viz)
            RUN_VIZ=false
            shift
            ;;
        --clean-only)
            RUN_SIM=false
            shift
            ;;
        --continue)
            CLEAN_OUTPUT=false
            shift
            ;;
    esac
done

# Extract output directory from config file
if [ -f "$CONFIG_FILE" ]; then
    OUTPUT_DIR=$(grep -h 'output_dir' "$CONFIG_FILE" | head -1 | sed 's/.*"\(.*\)".*/\1/')
    if [ -z "$OUTPUT_DIR" ]; then
        OUTPUT_DIR="./output/default"
    fi
else
    echo -e "${RED}ERROR: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Geodynamic Simulation Runner${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "Config file:   ${GREEN}$CONFIG_FILE${NC}"
echo -e "Output dir:    ${GREEN}$OUTPUT_DIR${NC}"
echo -e "Run viz:       ${GREEN}$RUN_VIZ${NC}"
echo -e "Clean output:  ${GREEN}$CLEAN_OUTPUT${NC}"
echo ""

# Step 1: Create directory structure
echo -e "${YELLOW}[1/4] Setting up directories...${NC}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/visualizations"
echo -e "  ✓ Created: $OUTPUT_DIR"
echo -e "  ✓ Created: $OUTPUT_DIR/visualizations"
echo ""

# Step 2: Clean previous files
if [ "$CLEAN_OUTPUT" = true ]; then
    echo -e "${YELLOW}[2/4] Cleaning previous output...${NC}"

    # Remove VTU files
    VTU_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.vtu" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$VTU_COUNT" -gt 0 ]; then
        find "$OUTPUT_DIR" -maxdepth 1 -name "*.vtu" -delete
        echo -e "  ✓ Removed $VTU_COUNT VTU files"
    else
        echo -e "  ✓ No VTU files to remove"
    fi

    # Remove old visualizations
    VIZ_COUNT=$(find "$OUTPUT_DIR/visualizations" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$VIZ_COUNT" -gt 0 ]; then
        rm -f "$OUTPUT_DIR/visualizations"/*.png
        echo -e "  ✓ Removed $VIZ_COUNT visualization files"
    else
        echo -e "  ✓ No visualization files to remove"
    fi

    # Archive old log if it exists
    if [ -f "$OUTPUT_DIR/simulation.log" ]; then
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        mv "$OUTPUT_DIR/simulation.log" "$OUTPUT_DIR/logs/simulation_$TIMESTAMP.log"
        echo -e "  ✓ Archived previous log to logs/simulation_$TIMESTAMP.log"
    fi
else
    echo -e "${YELLOW}[2/4] Skipping cleanup (--continue mode)${NC}"
fi
echo ""

# Step 3: Run simulation
if [ "$RUN_SIM" = true ]; then
    echo -e "${YELLOW}[3/4] Running simulation...${NC}"

    # Check if binary exists
    if [ ! -f "./target/release/core_complex" ]; then
        echo -e "${RED}ERROR: Binary not found. Building...${NC}"
        cargo build --release --bin core_complex
    fi

    # Run simulation with logging
    LOG_FILE="$OUTPUT_DIR/simulation.log"
    START_TIME=$(date +%s)

    echo -e "  → Running: ${GREEN}./target/release/core_complex $CONFIG_FILE${NC}"
    echo -e "  → Logging to: ${GREEN}$LOG_FILE${NC}"
    echo ""

    # Run with output to both terminal and log file
    if ./target/release/core_complex "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        HOURS=$((ELAPSED / 3600))
        MINUTES=$(((ELAPSED % 3600) / 60))
        SECONDS=$((ELAPSED % 60))

        echo ""
        echo -e "${GREEN}✓ Simulation completed successfully${NC}"
        echo -e "  Wall time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
        echo -e "  Log saved: $LOG_FILE"
    else
        echo -e "${RED}✗ Simulation failed (exit code: $?)${NC}"
        echo -e "  Check log: $LOG_FILE"
        exit 1
    fi
else
    echo -e "${YELLOW}[3/4] Skipping simulation (--clean-only mode)${NC}"
fi
echo ""

# Step 4: Run visualization scripts
if [ "$RUN_VIZ" = true ] && [ "$RUN_SIM" = true ]; then
    echo -e "${YELLOW}[4/4] Running visualization scripts...${NC}"

    # Check if Python visualization scripts exist
    VIZ_SCRIPTS=(
        "scripts/visualize_results.py"
        "scripts/plot_2d_slices.py"
        "scripts/plot_evolution.py"
    )

    FOUND_SCRIPTS=false
    for script in "${VIZ_SCRIPTS[@]}"; do
        if [ -f "$script" ]; then
            FOUND_SCRIPTS=true
            echo -e "  → Running: ${GREEN}python3 $script $OUTPUT_DIR${NC}"
            if python3 "$script" "$OUTPUT_DIR"; then
                echo -e "  ${GREEN}✓ $script completed${NC}"
            else
                echo -e "  ${YELLOW}⚠ $script failed (non-fatal)${NC}"
            fi
        fi
    done

    if [ "$FOUND_SCRIPTS" = false ]; then
        echo -e "  ${YELLOW}⚠ No visualization scripts found in scripts/directory${NC}"
        echo -e "  ${YELLOW}  Expected: visualize_results.py, plot_2d_slices.py, plot_evolution.py${NC}"
    fi
else
    echo -e "${YELLOW}[4/4] Skipping visualization${NC}"
fi
echo ""

# Summary
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Run Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "Output location: ${GREEN}$OUTPUT_DIR${NC}"
echo ""

# Count output files
VTU_FILES=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.vtu" 2>/dev/null | wc -l | tr -d ' ')
PNG_FILES=$(find "$OUTPUT_DIR/visualizations" -name "*.png" 2>/dev/null | wc -l | tr -d ' ')

echo -e "Files generated:"
echo -e "  VTU files:      $VTU_FILES"
echo -e "  Visualizations: $PNG_FILES"
echo ""

# Show quick commands
echo -e "${BLUE}Next steps:${NC}"
echo -e "  View log:        ${GREEN}less $OUTPUT_DIR/simulation.log${NC}"
if [ "$VTU_FILES" -gt 0 ]; then
    echo -e "  Open in ParaView: ${GREEN}paraview $OUTPUT_DIR/step_*.vtu${NC}"
fi
if [ "$PNG_FILES" -gt 0 ]; then
    echo -e "  View plots:       ${GREEN}open $OUTPUT_DIR/visualizations/${NC}"
fi
echo ""
