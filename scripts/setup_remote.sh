#!/usr/bin/env bash
# =============================================================================
# setup_remote.sh — One-shot infrastructure setup for VecAdvisor++ experiments
#
# Run this script on the remote server from the project root:
#   cd /tmp2/b11902156/VecAdvisor-pp
#   bash scripts/setup_remote.sh
#
# What it does:
#   1. Initialises a personal PostgreSQL 18 cluster at $PGDATA (port 15432)
#   2. Compiles and installs pgvector from source
#   3. Installs Rust toolchain + maturin, builds vecadvisor_rs extension
#   4. Downloads SIFT1M and GIST1M datasets
#
# No credentials are stored in this script. The PostgreSQL cluster uses
# OS-level peer authentication (your Unix user = DB superuser).
# =============================================================================
set -euo pipefail

REMOTE_BASE=/tmp2/b11902156
PGDATA=$REMOTE_BASE/pgdata
PGPORT=15432
PGDB=vecadvisor
PROJECT_DIR=$REMOTE_BASE/VecAdvisor-pp
DATA_DIR=$REMOTE_BASE/data

echo "==================================================================="
echo " VecAdvisor++ Remote Setup"
echo " REMOTE_BASE : $REMOTE_BASE"
echo " PGDATA      : $PGDATA"
echo " PGPORT      : $PGPORT"
echo " PROJECT_DIR : $PROJECT_DIR"
echo "==================================================================="

# ---------------------------------------------------------------------------
# Step 1: User-level PostgreSQL cluster
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: PostgreSQL cluster ---"

if [ -d "$PGDATA" ] && [ -f "$PGDATA/PG_VERSION" ]; then
    echo "Cluster already exists at $PGDATA — skipping initdb."
else
    initdb -D "$PGDATA" --encoding=UTF8 --locale=C
    echo "Cluster initialised at $PGDATA."
fi

# Patch postgresql.conf (idempotent — uses sed to replace or append)
patch_conf() {
    local key="$1" value="$2"
    local conf="$PGDATA/postgresql.conf"
    if grep -q "^${key}" "$conf" 2>/dev/null; then
        sed -i "s|^${key}.*|${key} = ${value}|" "$conf"
    else
        echo "${key} = ${value}" >> "$conf"
    fi
}

patch_conf "port"                       "$PGPORT"
patch_conf "listen_addresses"           "'localhost'"
patch_conf "unix_socket_directories"    "'$PGDATA'"
patch_conf "shared_buffers"             "32GB"
patch_conf "effective_cache_size"       "200GB"
patch_conf "maintenance_work_mem"       "4GB"
patch_conf "work_mem"                   "256MB"
patch_conf "max_wal_size"               "4GB"
patch_conf "checkpoint_completion_target" "0.9"
patch_conf "max_connections"            "50"

echo "postgresql.conf patched."

# Start cluster (or do nothing if already running)
if pg_ctl -D "$PGDATA" status > /dev/null 2>&1; then
    echo "PostgreSQL already running."
else
    pg_ctl -D "$PGDATA" -l "$REMOTE_BASE/pg.log" start
    echo "PostgreSQL started. Logs: $REMOTE_BASE/pg.log"
fi

# Create database (idempotent)
createdb -h localhost -p "$PGPORT" "$PGDB" 2>/dev/null \
    && echo "Database '$PGDB' created." \
    || echo "Database '$PGDB' already exists."

# ---------------------------------------------------------------------------
# Step 2: pgvector from source
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: pgvector ---"

PGVECTOR_DIR=$REMOTE_BASE/pgvector

if [ ! -d "$PGVECTOR_DIR" ]; then
    # Try v0.8.0 first; fall back to main if PG 18 requires a newer version
    git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git "$PGVECTOR_DIR" \
        || git clone https://github.com/pgvector/pgvector.git "$PGVECTOR_DIR"
fi

cd "$PGVECTOR_DIR"
# Attempt standard install (writes to /usr/lib/postgresql — may need sudo)
if make PG_CONFIG=/usr/bin/pg_config 2>&1; then
    if make install PG_CONFIG=/usr/bin/pg_config 2>&1; then
        echo "pgvector installed to system PG lib dir."
    else
        # Fall back to DESTDIR install + dynamic_library_path
        PGVEC_INSTALL=$REMOTE_BASE/pgvector_install
        make install DESTDIR="$PGVEC_INSTALL" PG_CONFIG=/usr/bin/pg_config
        LIB_PATH=$PGVEC_INSTALL/usr/lib/postgresql

        patch_conf "dynamic_library_path" "'$LIB_PATH:\$libdir'"
        echo "pgvector installed to $LIB_PATH and dynamic_library_path patched."

        # Reload config so the new path is active
        pg_ctl -D "$PGDATA" reload
    fi
else
    echo "ERROR: pgvector 'make' failed. Check that postgresql-devel headers are installed."
    echo "You may need: pacman -S postgresql"
    exit 1
fi

cd "$PROJECT_DIR"

psql -h localhost -p "$PGPORT" "$PGDB" \
    -c "CREATE EXTENSION IF NOT EXISTS vector;" \
    && echo "pgvector extension enabled." \
    || echo "WARNING: Could not create pgvector extension. Check the error above."

# ---------------------------------------------------------------------------
# Step 3: Rust toolchain + vecadvisor_rs
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: Rust + vecadvisor_rs ---"

# uv (Python package manager) — should already be installed
if ! command -v uv &>/dev/null; then
    source "$HOME/.local/bin/env" 2>/dev/null || true
fi

# Rust
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --quiet
    source "$HOME/.cargo/env"
    echo "Rust installed."
else
    source "$HOME/.cargo/env" 2>/dev/null || true
    echo "Rust already installed: $(rustc --version)"
fi

source "$PROJECT_DIR/.venv/bin/activate"
uv pip install maturin --quiet

maturin develop --release --manifest-path "$PROJECT_DIR/vecadvisor_rs/Cargo.toml"
echo "vecadvisor_rs built and installed."

# Verify
python -c "import vecadvisor_rs; fns=dir(vecadvisor_rs); print('vecadvisor_rs functions:', [f for f in fns if not f.startswith('_')])"

# ---------------------------------------------------------------------------
# Step 4: Download datasets
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4: Datasets ---"

mkdir -p "$DATA_DIR/sift1m" "$DATA_DIR/gist1m"

# SIFT1M — delegated to Python loader
python - <<'PYEOF'
import sys; sys.path.insert(0, ".")
from src.data.loader import download_sift1m
download_sift1m("/tmp2/b11902156/data/sift1m")
PYEOF

# GIST1M — ~3.6 GB
GIST_DIR=$DATA_DIR/gist1m
if [ -f "$GIST_DIR/gist/gist_base.fvecs" ]; then
    echo "GIST1M already present at $GIST_DIR/gist/"
else
    echo "Downloading GIST1M (~3.6 GB)..."
    wget --progress=dot:giga \
        ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz \
        -O "$GIST_DIR/gist.tar.gz"
    tar -xf "$GIST_DIR/gist.tar.gz" -C "$GIST_DIR/"
    echo "GIST1M extracted to $GIST_DIR/gist/"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "==================================================================="
echo " Setup complete!"
echo ""
echo " PostgreSQL: localhost:$PGPORT  db=$PGDB"
echo " PGDATA    : $PGDATA"
echo " Data      : $DATA_DIR"
echo ""
echo " To run a smoke test:"
echo "   export PGDATA=$PGDATA"
echo "   source $PROJECT_DIR/.venv/bin/activate"
echo "   python scripts/run_benchmark.py --config config/remote.yaml \\"
echo "       --n-base 10000 --n-queries 100 --k 10"
echo "==================================================================="
