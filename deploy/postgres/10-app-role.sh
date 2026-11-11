#!/bin/sh
# Runs once, on first start of an empty data directory. The schema is created later by the migrate
# service as the owner; the API only ever connects as this role, which receives DML grants and nothing else.
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v app_password="$RAGTR_APP_PASSWORD" <<'SQL'
CREATE ROLE ragtr_app LOGIN PASSWORD :'app_password';
GRANT CONNECT ON DATABASE ragtr TO ragtr_app;
GRANT USAGE ON SCHEMA public TO ragtr_app;
SQL
