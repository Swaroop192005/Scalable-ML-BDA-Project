#!/usr/bin/env bash
#
# Test scripts/download_data.py, and say which layer broke when it fails.
#
# A download can fail at four separate places, and the fixes are all
# different. This runs them in dependency order so the first FAIL names the
# real problem instead of leaving you with a traceback:
#
#   DNS            the host name does not resolve        -> network / VPN
#   TLS + HTTP     curl cannot fetch it                  -> firewall / UCI down
#   Python HTTPS   curl works but Python does not        -> missing CA certs
#   the download   everything reachable, script failed   -> a real bug
#
# The Python check matters on macOS especially: the python.org installer
# ships its own certificate store, so curl can succeed while Python fails
# with a certificate error. Testing only with curl would miss that entirely.
#
# Usage:
#   bash scripts/check_download.sh                  full test
#   bash scripts/check_download.sh --no-download    diagnostics only
#   PYTHON=python3.12 bash scripts/check_download.sh
#
# Exit status is 0 when every check passed, 1 otherwise.

# Deliberately no `set -e`: a failing check must be recorded and reported,
# not abort the run before the summary prints.
set -uo pipefail

HOST="archive.ics.uci.edu"
URL="https://archive.ics.uci.edu/static/public/352/online+retail.zip"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CSV="$ROOT/data/raw/online_retail.csv"
PY="${PYTHON:-python3}"

DO_DOWNLOAD=1
if [ "${1:-}" = "--no-download" ]; then DO_DOWNLOAD=0; fi

if [ -t 1 ]; then
  RED=$(printf '\033[31m'); GRN=$(printf '\033[32m'); YEL=$(printf '\033[33m')
  DIM=$(printf '\033[2m');  BLD=$(printf '\033[1m'); OFF=$(printf '\033[0m')
else
  RED=''; GRN=''; YEL=''; DIM=''; BLD=''; OFF=''
fi

PASSED=0; FAILED=0; WARNED=0
step() { printf '\n%s%s%s\n' "$BLD" "$1" "$OFF"; }
ok()   { printf '  %sPASS%s  %s\n' "$GRN" "$OFF" "$1"; PASSED=$((PASSED + 1)); }
no()   { printf '  %sFAIL%s  %s\n' "$RED" "$OFF" "$1"; FAILED=$((FAILED + 1)); }
hm()   { printf '  %sWARN%s  %s\n' "$YEL" "$OFF" "$1"; WARNED=$((WARNED + 1)); }
fix()  { printf '        %s%s%s\n' "$DIM" "$1" "$OFF"; }

printf '%sTesting the Online Retail downloader%s\n' "$BLD" "$OFF"
printf '%srepo   %s%s\n' "$DIM" "$ROOT" "$OFF"

# ---------------------------------------------------------------- 1. Python
step "1/6  Python interpreter"
if command -v "$PY" >/dev/null 2>&1; then
  ok "$($PY --version 2>&1) at $(command -v "$PY")"
else
  no "'$PY' not found on PATH"
  fix "activate your virtualenv, or set PYTHON=... to name one"
  exit 1
fi

# ------------------------------------------------------------------- 2. DNS
step "2/6  DNS resolution of $HOST"
RESOLVED=$("$PY" - "$HOST" <<'EOF' 2>&1
import socket, sys
try:
    info = socket.getaddrinfo(sys.argv[1], 443)
    print(sorted({i[4][0] for i in info})[0])
except Exception as exc:
    print(f"ERROR {exc}")
    sys.exit(1)
EOF
)
if [ $? -eq 0 ]; then
  ok "resolves to $RESOLVED"
else
  no "cannot resolve $HOST  ($RESOLVED)"
  fix "This is the 'nodename nor servname provided' error."
  fix "The machine cannot reach the internet for this request at all:"
  fix "check wifi, VPN, or a campus/corporate network blocking it."
  fix "Workaround: download by hand (see the end of this output)."
fi

# ------------------------------------------------------- 3. TLS + HTTP reach
step "3/6  HTTPS reachability (curl)"
if command -v curl >/dev/null 2>&1; then
  ERRFILE=$(mktemp 2>/dev/null || echo "/tmp/cd_curl_err.$$")
  # A reachability check must not download the file. Range is only a
  # request -- UCI ignores it and streams all 24 MB -- so this caps the
  # time instead and reads the status code. Any status at all proves the
  # connection succeeded, so a timeout part-way through the body is a PASS
  # here: getting bytes is the whole question.
  CODE=$(curl -sS -L --max-time 20 -o /dev/null -w '%{http_code}' "$URL" 2>"$ERRFILE")
  CURL_RC=$?
  CURL_ERR=$(tr '\n' ' ' < "$ERRFILE"); rm -f "$ERRFILE"

  if [ "$CODE" = "200" ] || [ "$CODE" = "206" ]; then
    if [ $CURL_RC -eq 28 ]; then
      ok "HTTP $CODE from UCI (transfer capped at 20s, which is expected here)"
    else
      ok "HTTP $CODE from UCI"
    fi
  elif [ "$CODE" = "000" ]; then
    no "never connected: $CURL_ERR"
    case "$CURL_ERR" in
      *ertificate*|*SSL*) fix "Certificate problem — your system trust store." ;;
      *resolve*)          fix "DNS — see check 2." ;;
      *imed\ out*)        fix "Timed out before any response: firewall, or UCI is down." ;;
      *)                  fix "Check your connection and any proxy settings." ;;
    esac
  else
    hm "HTTP $CODE (expected 200 or 206)"
    fix "UCI's archive is intermittently down or rate-limiting. Retry shortly."
  fi
else
  hm "curl not installed; skipping (the Python check below still covers it)"
fi

# ------------------------------------------------- 4. Python's own TLS stack
step "4/6  Python HTTPS (catches the macOS certificate problem)"
PYNET=$("$PY" - "$URL" <<'EOF' 2>&1
import sys, urllib.request, urllib.error
request = urllib.request.Request(sys.argv[1], headers={"User-Agent": "scalable-mba/1.0"})
try:
    with urllib.request.urlopen(request, timeout=45) as response:
        response.read(1024)
        print(f"OK status={response.status}")
except Exception as exc:
    print(f"ERROR {type(exc).__name__}: {exc}")
    sys.exit(1)
EOF
)
if [ $? -eq 0 ]; then
  ok "Python can fetch over HTTPS  ($PYNET)"
else
  no "Python cannot fetch  ($PYNET)"
  case "$PYNET" in
    *ertificate*|*SSL*|*ssl*)
      fix "Python's CA certificates are not installed. On a python.org build:"
      fix "  /Applications/Python\\ 3.x/Install\\ Certificates.command"
      fix "substituting your version. curl may still work — it uses a different store."
      ;;
    *gaierror*|*nodename*|*"Name or service"*)
      fix "DNS again — see check 2."
      ;;
    *)
      fix "If curl passed but this failed, the problem is Python's networking,"
      fix "not your connection: proxy env vars (http_proxy/https_proxy) are likely."
      ;;
  esac
fi

# -------------------------------------------------------- 5. run the script
step "5/6  scripts/download_data.py"
if [ -f "$CSV" ]; then
  ok "already downloaded: $CSV ($(du -h "$CSV" | cut -f1))"
  fix "the script would no-op; pass --force to re-download"
elif [ "$DO_DOWNLOAD" -eq 0 ]; then
  hm "skipped (--no-download)"
else
  printf '        %srunning (this takes a minute: ~24 MB, then an xlsx->csv convert)%s\n' "$DIM" "$OFF"
  if "$PY" "$ROOT/scripts/download_data.py"; then
    ok "the script completed"
  else
    no "the script failed — see its message above"
  fi
fi

# ------------------------------------------------------- 6. verify the file
step "6/6  Verify the dataset"
if [ ! -f "$CSV" ]; then
  no "$CSV does not exist"
else
  VERIFY=$("$PY" - "$CSV" <<'EOF' 2>&1
import sys
import pandas as pd

path = sys.argv[1]
expected = {"InvoiceNo", "StockCode", "Description", "Quantity",
            "InvoiceDate", "UnitPrice", "CustomerID", "Country"}
frame = pd.read_csv(path, dtype={"InvoiceNo": str, "StockCode": str})
missing = expected - set(frame.columns)
if missing:
    print(f"ERROR missing columns: {sorted(missing)}")
    sys.exit(1)
if len(frame) < 500_000:
    print(f"ERROR only {len(frame):,} rows; the full file has 541,909 "
          "(did you point this at the sample?)")
    sys.exit(1)
print(f"OK {len(frame):,} rows x {frame.shape[1]} columns, "
      f"{frame['InvoiceNo'].nunique():,} invoices, "
      f"{frame['Country'].nunique()} countries")
EOF
)
  if [ $? -eq 0 ]; then
    ok "${VERIFY#OK }"
  else
    no "${VERIFY#ERROR }"
    fix "Remove data/raw/online_retail.csv and run this again."
  fi
fi

# ------------------------------------------------------------------ summary
printf '\n%s%s%s\n' "$BLD" "----------------------------------------------------------" "$OFF"
printf '%s%d passed%s' "$GRN" "$PASSED" "$OFF"
[ "$WARNED" -gt 0 ] && printf ', %s%d warned%s' "$YEL" "$WARNED" "$OFF"
[ "$FAILED" -gt 0 ] && printf ', %s%d failed%s' "$RED" "$FAILED" "$OFF"
printf '\n'

if [ "$FAILED" -eq 0 ]; then
  printf '\nDataset is ready. Next:\n'
  printf '  pytest        %s269 passed, nothing skipped%s\n' "$DIM" "$OFF"
  printf '  mba run       %sfull pipeline over 541,909 rows%s\n' "$DIM" "$OFF"
  exit 0
fi

cat <<'EOT'

Could not get the dataset automatically. Download it by hand:

  1. Open  https://archive.ics.uci.edu/dataset/352/online+retail
     (Kaggle mirror: https://www.kaggle.com/datasets/carrie1/ecommerce-data)
  2. Save the file into  data/raw/
  3. Run  python scripts/download_data.py
     It picks up a file already sitting there and just converts it.

You do not need the dataset to use the project. Everything runs on the
committed 2,500-invoice sample:

  mba run -i data/sample/sample_retail.csv
  pytest                 249 pass, the 7 full-dataset tests skip
EOT
exit 1
