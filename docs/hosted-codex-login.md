# Hosted Codex subscription login

The benchmark site's Google login and the server's Codex/ChatGPT login are
separate. A `codex_authentication_required` error means the server login needs
operator renewal; signing into the website again will not repair it.

Use a dedicated Codex login for the server, not a copy of a credential cache
that is still being used by a desktop or benchmark runner. Create that login
with `codex login --device-auth`, using a separate `CODEX_HOME` directory and
`cli_auth_credentials_store="file"`. Transfer its `auth.json` privately to the
Render `CODEX_AUTH_JSON_B64` secret. Never print, commit, or paste its contents
into chat. Do not subsequently run local jobs with that transferred login.

When replacing the secret, set a new non-secret `CODEX_AUTH_REVISION` value in
the same update. On startup, a changed revision atomically installs the new
credentials. Ordinary restarts with the same revision preserve the CLI's
refreshed credentials. Without a revision, existing disk credentials are
preserved for backward compatibility.

The live service must actually have the `codex-auth` disk mounted at `/var/data`
(as declared in `render.yaml`), with `CODEX_HOME=/var/data/codex`. Merely declaring
the disk in YAML does not attach it to an independently configured service.
Without it, redeployments discard rotated credentials and restore a stale seed.
Render one-off jobs do not share this disk: do not seed diagnostic jobs from the
server's login secret. Verify moves through the running service instead.

Authentication errors fail without retrying the same invalid login. The UI
reports the required action and a reference ID; server logs retain the reference,
game/model/effort, position, error category, and stack locations, but not raw
provider output, exception text, or credentials. These errors are not forfeits.

Sources:
- https://learn.chatgpt.com/docs/auth
- https://render.com/docs/disks
