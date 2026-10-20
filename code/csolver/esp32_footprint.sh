#!/usr/bin/env bash
# Static memory footprint of the solver and the costate law on the CCU target (ESP32, Xtensa LX6),
# compiled with the ESP32 Arduino core toolchain (esp32:esp32 3.3.7, xtensa-esp-elf-gcc 14.2).
# Writes ../../results/e6/esp32_footprint.json. No board is needed; on-target timing still requires
# flashing esp32_bench/ (see README.md).
set -e
cd "$(dirname "$0")"
BIN="${XTENSA_BIN:-/c/Users/Asus ROG Strix G16/AppData/Local/Arduino15/packages/esp32/tools/esp-x32/2511/bin}"
CLI="${ARDUINO_CLI:-/c/Users/Asus ROG Strix G16/AppData/Local/Programs/Arduino IDE/resources/app/lib/backend/resources/arduino-cli.exe}"
T="$(mktemp -d)"
CC="$BIN/xtensa-esp-elf-gcc.exe"; SZ="$BIN/xtensa-esp-elf-size.exe"; NM="$BIN/xtensa-esp-elf-nm.exe"
"$CC" -DMPC_DEPLOYED_DIMS -mlongcalls -Os -std=c99 -c mpc_solver.c -o "$T/ms.o"
solver_text=$("$SZ" "$T/ms.o" | awk 'NR==2{print $1}')
printf '#include "mpc_solver.h"\nmpc_work_t wsp;\n' > "$T/ws.c"
"$CC" -DMPC_DEPLOYED_DIMS -mlongcalls -Os -std=c99 -I. -c "$T/ws.c" -o "$T/ws.o"
ws=$((16#$("$NM" -S "$T/ws.o" | awk '/ wsp$/{print $2}')))
printf '#include "costate_law.h"\nint cs_entry(const costate_params_t *c, const float x[CS_NP][CS_NZ], float occ, float rho){ return costate_duty(c, x, occ, rho); }\n' > "$T/cs.c"
"$CC" -mlongcalls -Os -std=c99 -I. -c "$T/cs.c" -o "$T/cs.o"
cs_text=$("$SZ" "$T/cs.o" | awk 'NR==2{print $1}')
printf '#include "rollout_law.h"
int rl_entry(const rollout_params_t *c, const float x[RL_NP][RL_NZ], const float d[RL_NP][RL_NZ], int k, float rem){ return rollout_duty(c, x, d, k, rem); }
const unsigned rl_param_bytes = sizeof(rollout_params_t);
' > "$T/rl.c"
"$CC" -mlongcalls -Os -std=c99 -I. -c "$T/rl.c" -o "$T/rl.o"
rl_text=$("$SZ" "$T/rl.o" | awk 'NR==2{print $1}')
printf '#include "rollout_law.h"
unsigned char probe[sizeof(rollout_params_t)];
' > "$T/rp.c"
"$CC" -mlongcalls -Os -std=c99 -I. -c "$T/rp.c" -o "$T/rp.o"
rl_par=$((16#$("$NM" -S "$T/rp.o" | awk '/ probe$/{print $2}')))
# whole benchmark sketch (Arduino core + solver + five instances), as reported by arduino-cli
mkdir -p "$T/esp32_bench" && cp esp32_bench/esp32_bench.ino esp32_bench/instances.h mpc_solver.c mpc_solver.h "$T/esp32_bench/"
rep=$("$CLI" compile --fqbn esp32:esp32:esp32:CPUFreq=240 "$T/esp32_bench" 2>&1 | grep -E "Sketch uses|Global variables")
flash=$(echo "$rep" | sed -n 's/Sketch uses \([0-9]*\) bytes.*/\1/p')
ram=$(echo "$rep" | sed -n 's/Global variables use \([0-9]*\) bytes.*/\1/p')
mkdir -p ../../results/e6
cat > ../../results/e6/esp32_footprint.json <<EOF
{"solver_text_bytes_Os": $solver_text, "workspace_bytes_xtensa": $ws, "costate_text_bytes_Os": $cs_text,
 "rollout_text_bytes_Os": $rl_text, "rollout_param_bytes": $rl_par,
 "bench_sketch_flash_bytes": $flash, "bench_sketch_static_ram_bytes": $ram,
 "toolchain": "esp32:esp32 3.3.7, xtensa-esp-elf-gcc 14.2.0"}
EOF
cat ../../results/e6/esp32_footprint.json
