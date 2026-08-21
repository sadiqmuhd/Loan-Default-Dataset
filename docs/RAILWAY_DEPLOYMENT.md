# Deploying to Railway

The whole pipeline is:

```
GitHub repository → Railway (Nixpacks) → Python 3.11 → FastAPI → model artifact
```

There is no container registry, no artifact store and no separate training step
in the deploy path. Railway builds from the repository and the model ships with
it.

---

## 1. Connect the repository

1. Push this repository to GitHub.
2. In Railway, choose **New Project → Deploy from GitHub repo** and pick it.
3. Railway detects Python and reads `nixpacks.toml`. No template selection
   needed.

Railway redeploys on every push to the default branch. If you would rather it
only deployed when CI passes, turn off automatic deploys in the service settings
and trigger them from the GitHub Actions workflow instead.

---

## 2. How the build works

`nixpacks.toml` overrides the default Python install phase:

```toml
[phases.install]
cmds = [
  "python -m venv --copies /opt/venv",
  ". /opt/venv/bin/activate && pip install --upgrade pip setuptools wheel",
  ". /opt/venv/bin/activate && pip install -r requirements.txt",
  ". /opt/venv/bin/activate && pip install --no-deps -e ."
]
```

The last line is the one that matters and the one Nixpacks would not do on its
own. The application package lives under `src/`, so installing
`requirements.txt` alone leaves `loan_default` unimportable and
`uvicorn loan_default.api.main:app` fails at startup with a `ModuleNotFoundError`.

`requirements.txt` carries the pinned runtime dependencies. `pyproject.toml`
remains the source of truth, and `tests/unit/test_dependency_pins.py` fails if
the two disagree — otherwise the deployed environment could quietly drift from
the tested one.

Python version comes from `.python-version` (3.11).

---

## 3. How the app starts

From `railway.toml`:

```toml
startCommand = "uvicorn loan_default.api.main:app --host 0.0.0.0 --port $PORT --workers 2"
```

Two things are non-negotiable here:

- **Bind `0.0.0.0`, not `127.0.0.1`.** Binding to loopback makes the service
  unreachable from outside the container, and the deploy will look healthy while
  timing out.
- **Use `$PORT`.** Railway assigns the port at runtime. Hardcoding 8000 means
  the router cannot reach the app.

`Procfile` carries the same command as a fallback for any platform that reads it.

---

## 4. Model artifacts

The trained model is committed to the repository under `artifacts/`, and the
deploy uses it directly.

That is a deliberate choice. The artifact is 880KB — well inside what belongs in
git — and bundling it means the build is fully reproducible from a single
commit, with no external storage to configure, no credentials to manage, and no
possibility of the service booting against a model nobody can identify. Pulling
a model from object storage at boot would be the right call at a few hundred
megabytes, or if models were retrained independently of code. Neither applies
here, and adding S3 to this project would be complexity without a reason.

`artifacts/LATEST` names the version to serve. Each version directory holds:

```
artifacts/v20260821T011832Z/
  model.joblib      the fitted, calibrated pipeline
  metadata.json     version, data hash, feature contract, assumptions
  metrics.json      the metrics accepted at approval
```

To roll back, set `LOAN_DEFAULT_MODEL_VERSION` to an older version directory
that is still committed. To roll forward, retrain locally, commit the new
artifact, and push.

The 28MB training dataset is **not** deployed. `data/portfolio_sample.csv` is —
5,000 stratified rows, about 1MB — so the portfolio and stress endpoints work on
a fresh deploy without shipping the full file.

---

## 5. Environment variables

None are required. Every setting has a working default, which is why the service
deploys with no configuration at all.

Optional overrides, all prefixed `LOAN_DEFAULT_`:

| Variable | Default | Purpose |
|---|---|---|
| `LOAN_DEFAULT_LOG_LEVEL` | `INFO` | Log verbosity |
| `LOAN_DEFAULT_JSON_LOGS` | `true` | Structured JSON logs (keep on for Railway) |
| `LOAN_DEFAULT_MODEL_VERSION` | `latest` | Pin a specific artifact, or roll back |
| `LOAN_DEFAULT_CORS_ORIGINS` | localhost | Comma-separated allowed origins |
| `LOAN_DEFAULT_MAX_BATCH_SIZE` | `1000` | Batch endpoint ceiling |

Set them under **Variables** in the Railway service. `PORT` is injected by
Railway and must not be set by hand.

No secrets are needed by this service, and none are in the repository. If that
changes, use Railway variables and never commit them.

---

## 6. Health checks

`railway.toml` sets:

```toml
healthcheckPath = "/health/ready"
healthcheckTimeout = 300
```

`/health/ready` returns 200 only when the model has loaded **and** successfully
scored a canary record at startup. A deploy carrying a corrupt or missing
artifact fails the health check and Railway keeps the previous version serving,
rather than routing traffic to an instance that returns 500 on every request.

`/health/live` is separate and only reports that the process is up. Use it for
uptime monitoring; use `/health/ready` for traffic gating.

The 300 second timeout is generous on purpose: first boot loads the model and
builds the SHAP explainer, which takes a few seconds on a cold container.

---

## 7. Make the service public

Under **Settings → Networking**, click **Generate Domain**. Railway issues a
`*.up.railway.app` URL.

Then check:

- `https://<your-app>.up.railway.app/docs` — interactive API docs
- `https://<your-app>.up.railway.app/health/ready` — should report `ready`
- `https://<your-app>.up.railway.app/v1/model/metadata` — model provenance

---

## 8. Logs

**Deployments → View Logs** in the Railway dashboard, or `railway logs` with the
CLI.

Logs are JSON, one object per line, so they filter cleanly. Every scoring
request emits an audit record:

```json
{"timestamp":"2026-08-21T01:18:32Z","level":"INFO","event":"credit_decision",
 "request_id":"3f8a1c2e...","model_version":"v20260821T011832Z",
 "pd":0.057793,"risk_grade":"C","decision":"APPROVE","latency_ms":92.1}
```

Raw applicant field values are deliberately not logged. The request id links an
operational log line to a decision without putting personal data in the log.

---

## 9. Verify a deployment

```bash
APP=https://<your-app>.up.railway.app

curl -fsS $APP/health/ready
curl -fsS $APP/v1/model/metadata | head -c 400

curl -fsS -X POST $APP/v1/risk/assess \
  -H 'Content-Type: application/json' \
  -d @tests/fixtures/example_application.json
```

A healthy deployment returns `"status":"ready"`, a model version matching
`artifacts/LATEST`, and a scored assessment with a PD between 0 and 1.

The same checks run in CI on every push, against a locally booted instance, so a
broken start command is caught before it reaches Railway.

---

## 10. Custom domain

**Settings → Networking → Custom Domain**, enter the domain, and add the CNAME
record Railway shows you at your DNS provider. TLS is provisioned automatically
once the record resolves.

If the API will be called from a browser on that domain, add it to
`LOAN_DEFAULT_CORS_ORIGINS`. The default allows localhost only, and credentials
are disabled — a wildcard origin combined with credentials is rejected by
browsers anyway.

---

## Troubleshooting

**Build fails on `pip install --no-deps -e .`** — usually a `pyproject.toml`
syntax error. Reproduce locally with the exact commands in §2.

**Deploy is healthy but requests time out** — the app is bound to the wrong
interface or port. Check the start command uses `0.0.0.0` and `$PORT`.

**`/health/ready` returns 503** — the model did not load or the canary failed.
The response body carries the reason, and the boot logs carry the traceback.
Most often `artifacts/` was not committed, or `LOAN_DEFAULT_MODEL_VERSION` names
a version that is not in the repository.

**`ModuleNotFoundError: loan_default`** — the editable install step was skipped.
Confirm `nixpacks.toml` is at the repository root and Railway is reading it.
