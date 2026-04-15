import importlib
import os
import re
import select
import shutil
import socket
import subprocess
import time
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from config import (
    DEFAULT_BACKEND_HOST,
    DEFAULT_BACKEND_PORT,
    DEFAULT_QUICK_TUNNEL_ATTEMPTS,
    DEFAULT_QUICK_TUNNEL_EDGE_IP_VERSION,
    DEFAULT_QUICK_TUNNEL_PROTOCOL,
    DEFAULT_QUICK_TUNNEL_STARTUP_TIMEOUT_SEC,
    DEFAULT_REMOTE_API_KEY_QUERY_PARAM,
)


def discover_local_ipv4_addresses():
    ips = {"127.0.0.1"}

    probe = None
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        primary_ip = probe.getsockname()[0]
        if primary_ip:
            ips.add(str(primary_ip))
    except Exception:
        pass
    finally:
        if probe is not None:
            try:
                probe.close()
            except Exception:
                pass

    try:
        hostname = socket.gethostname()
        _host, _aliases, host_ips = socket.gethostbyname_ex(hostname)
        for ip in host_ips:
            if ip:
                ips.add(str(ip))
    except Exception:
        pass

    def _sort_key(ip_addr):
        if ip_addr.startswith("127."):
            return (1, ip_addr)
        return (0, ip_addr)

    return sorted(ips, key=_sort_key)


def print_backend_endpoints(host, port):
    host_str = str(host).strip() or DEFAULT_BACKEND_HOST
    port_int = int(port)

    if host_str in ("0.0.0.0", "::"):
        addresses = discover_local_ipv4_addresses()
    else:
        addresses = [host_str]

    print("Remote control backend endpoints:")
    print(f"  bind: {host_str}:{port_int}")
    for ip_addr in addresses:
        print(f"  backend target: {ip_addr}:{port_int}")
        print(f"    ws: ws://{ip_addr}:{port_int}/ws")
        print(f"    offer: http://{ip_addr}:{port_int}/offer")

    print("For gdog-remote backend input, use one of the 'backend target' values above.")


def summarize_neofetch_output(raw_output):
    lines = []
    for raw_line in str(raw_output or "").splitlines():
        cleaned = raw_line.strip()
        if cleaned:
            lines.append(cleaned)

    fields = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key_text = key.strip().lower()
        value_text = value.strip()
        if key_text and value_text:
            fields[key_text] = value_text

    summary = {}
    field_mapping = {
        "os": "os",
        "host": "host",
        "kernel": "kernel",
        "cpu": "cpu",
        "gpu": "gpu",
        "memory": "memory",
    }
    for output_key, summary_key in field_mapping.items():
        value = fields.get(output_key)
        if value:
            summary[summary_key] = value

    return {
        "line_count": len(lines),
        "raw": "\n".join(lines),
        "summary": summary,
    }


def collect_runtime_system_info():
    cmd = ["neofetch", "--stdout"]
    result_payload = {
        "source": "neofetch",
        "command": " ".join(cmd),
        "ok": False,
        "error": "",
        "raw": "",
        "summary": {},
        "line_count": 0,
    }

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except Exception as exc:
        result_payload["error"] = str(exc)
        return result_payload

    stdout = str(completed.stdout or "")
    stderr = str(completed.stderr or "").strip()
    parsed = summarize_neofetch_output(stdout)

    result_payload["ok"] = bool(completed.returncode == 0 and parsed["raw"])
    result_payload["error"] = "" if result_payload["ok"] else (stderr or f"exit code {completed.returncode}")
    result_payload["raw"] = parsed["raw"]
    result_payload["summary"] = parsed["summary"]
    result_payload["line_count"] = int(parsed["line_count"])

    return result_payload


def render_neofetch_banner():
    cmd = ["neofetch"]
    env = os.environ.copy()
    env["CLICOLOR_FORCE"] = "1"
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            timeout=10.0,
            env=env,
        )
    except Exception as exc:
        return False, str(exc)

    if completed.returncode != 0:
        return False, f"exit code {completed.returncode}"

    return True, ""


def terminate_process(process, process_label):
    if process is None:
        return
    if process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=3.0)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=2.0)
        except Exception:
            pass


def normalize_remote_url(remote_url):
    value = str(remote_url or "").strip()
    if not value:
        return ""
    if "://" not in value:
        value = f"https://{value}"
    return value


def extract_openai_api_key(raw_text):
    text = str(raw_text or "")
    match = re.search(r"(sk-[A-Za-z0-9_-]+)", text)
    if match:
        return match.group(1).strip()
    return ""


def load_remote_api_key(explicit_key, key_file):
    from_arg = extract_openai_api_key(explicit_key)
    if from_arg:
        return from_arg, "--remote-api-key"

    file_path = str(key_file or "").strip()
    if not file_path:
        return "", ""

    expanded_path = os.path.expanduser(file_path)
    try:
        with open(expanded_path, "r", encoding="utf-8") as handle:
            file_contents = handle.read()
    except FileNotFoundError:
        return "", ""
    except Exception as exc:
        print(f"Failed to read remote API key file '{expanded_path}': {exc}")
        return "", ""

    from_file = extract_openai_api_key(file_contents)
    if from_file:
        return from_file, expanded_path

    return "", ""


def build_remote_prefilled_link(
    remote_url,
    backend_target,
    remote_api_key="",
    remote_api_key_param=DEFAULT_REMOTE_API_KEY_QUERY_PARAM,
):
    base_url = normalize_remote_url(remote_url)
    backend_value = str(backend_target or "").strip()
    api_key_value = str(remote_api_key or "").strip()
    api_key_param = str(remote_api_key_param or "").strip() or DEFAULT_REMOTE_API_KEY_QUERY_PARAM
    if not base_url or not backend_value:
        return ""

    parts = urlsplit(base_url)
    if not parts.scheme or not parts.netloc:
        return ""

    query_map = dict(parse_qsl(parts.query, keep_blank_values=True))
    query_map["backend"] = backend_value
    if api_key_value:
        query_map[api_key_param] = api_key_value
    new_query = urlencode(query_map, doseq=True)

    path = parts.path or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, new_query, parts.fragment))


def print_ascii_qr(payload):
    text = str(payload or "").strip()
    if not text:
        return False

    try:
        qrcode_mod = importlib.import_module("qrcode")
    except Exception:
        print("Terminal QR rendering unavailable (optional dependency missing: qrcode).")
        print("Install with: python -m pip install qrcode")
        qr_image_url = f"https://api.qrserver.com/v1/create-qr-code/?size=360x360&data={quote(text, safe='')}"
        print("QR image URL:")
        print(f"  {qr_image_url}")
        return False

    try:
        qr = qrcode_mod.QRCode(border=1)
        qr.add_data(text)
        qr.make(fit=True)
        print("Scan this QR with your phone camera:")
        qr.print_ascii(invert=True)
        return True
    except Exception as exc:
        print(f"Failed to render terminal QR: {exc}")
        return False


def print_remote_controller_shortcut(
    remote_url,
    backend_target,
    print_qr=True,
    remote_api_key="",
    remote_api_key_param=DEFAULT_REMOTE_API_KEY_QUERY_PARAM,
):
    link = build_remote_prefilled_link(
        remote_url=remote_url,
        backend_target=backend_target,
        remote_api_key=remote_api_key,
        remote_api_key_param=remote_api_key_param,
    )
    if not link:
        return ""

    if str(remote_api_key or "").strip():
        print(f"Remote controller link (backend + {remote_api_key_param} prefilled):")
    else:
        print("Remote controller link (backend prefilled):")
    print(f"  {link}")
    if bool(print_qr):
        print_ascii_qr(link)
    return link


def start_cloudflare_quick_tunnel(
    bind_host,
    bind_port,
    startup_timeout_sec=DEFAULT_QUICK_TUNNEL_STARTUP_TIMEOUT_SEC,
    attempts=DEFAULT_QUICK_TUNNEL_ATTEMPTS,
    protocol=DEFAULT_QUICK_TUNNEL_PROTOCOL,
    edge_ip_version=DEFAULT_QUICK_TUNNEL_EDGE_IP_VERSION,
):
    cloudflared_bin = shutil.which("cloudflared")
    if cloudflared_bin is None:
        print("Cloudflare quick tunnel requested, but 'cloudflared' is not installed.")
        print("Install on macOS: brew install cloudflared")
        return None, None

    host_str = str(bind_host).strip() or DEFAULT_BACKEND_HOST
    tunnel_target_host = "127.0.0.1" if host_str in ("0.0.0.0", "::") else host_str
    tunnel_target_url = f"http://{tunnel_target_host}:{int(bind_port)}"

    cmd = [
        cloudflared_bin,
        "tunnel",
        "--url",
        tunnel_target_url,
        "--no-autoupdate",
    ]

    protocol_value = str(protocol or "").strip().lower() or DEFAULT_QUICK_TUNNEL_PROTOCOL
    if protocol_value != DEFAULT_QUICK_TUNNEL_PROTOCOL:
        cmd.extend(["--protocol", protocol_value])

    edge_ip_version_value = str(edge_ip_version or "").strip().lower() or DEFAULT_QUICK_TUNNEL_EDGE_IP_VERSION
    if edge_ip_version_value in {"4", "6"}:
        cmd.extend(["--edge-ip-version", edge_ip_version_value])

    attempts_int = max(int(attempts), 1)
    timeout_sec = max(float(startup_timeout_sec), 1.0)
    tunnel_pattern = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com", re.IGNORECASE)

    for attempt_idx in range(attempts_int):
        attempt_num = attempt_idx + 1
        print(
            "Starting Cloudflare quick tunnel for "
            f"{tunnel_target_url} (protocol={protocol_value}, edge_ip={edge_ip_version_value}, "
            f"attempt={attempt_num}/{attempts_int}) ..."
        )

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            print(f"Failed to start cloudflared: {exc}")
            return None, None

        tunnel_url = None
        recent_output_lines = []

        def _remember_output_line(raw_line):
            line = str(raw_line or "").rstrip("\n")
            if not line:
                return
            recent_output_lines.append(line)
            if len(recent_output_lines) > 12:
                del recent_output_lines[:-12]

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if process.poll() is not None:
                break
            if process.stdout is None:
                break

            ready, _, _ = select.select([process.stdout], [], [], 0.25)
            if not ready:
                continue

            line = process.stdout.readline()
            if not line:
                continue

            _remember_output_line(line)
            match = tunnel_pattern.search(line)
            if match:
                tunnel_url = match.group(0).rstrip("/")
                break

        if tunnel_url is not None:
            if tunnel_url.startswith("https://"):
                ws_base = f"wss://{tunnel_url[len('https://') :]}"
            else:
                ws_base = tunnel_url

            print("Cloudflare quick tunnel ready:")
            print(f"  backend target: {tunnel_url}")
            print(f"    ws: {ws_base}/ws")
            print(f"    offer: {tunnel_url}/offer")
            print("Use the 'backend target' URL above in the gdog-remote backend input.")
            return process, tunnel_url

        if process.stdout is not None:
            while True:
                try:
                    line = process.stdout.readline()
                except Exception:
                    break
                if not line:
                    break
                _remember_output_line(line)

        if process.poll() is None:
            print("Timed out waiting for Cloudflare quick tunnel URL.")
        else:
            print(f"cloudflared exited before tunnel became ready (code {process.returncode}).")

        if recent_output_lines:
            print("Recent cloudflared output:")
            for log_line in recent_output_lines[-8:]:
                print(f"  {log_line}")

        terminate_process(process, "cloudflared")

        if attempt_num < attempts_int:
            retry_delay = min(2.0 * attempt_num, 8.0)
            print(f"Retrying quick tunnel in {retry_delay:.1f}s...")
            time.sleep(retry_delay)
            continue

        print("You can also start tunnel manually in another terminal:")
        print(f"  {' '.join(cmd)}")
        print(
            "If this is restrictive guest Wi-Fi, retry with "
            "--quick-tunnel-protocol http2 --quick-tunnel-edge-ip-version 4."
        )
        return None, None

    return None, None
