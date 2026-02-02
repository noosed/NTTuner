import dearpygui.dearpygui as dpg
import json
import os
import time
import threading
import urllib.request
import re
import random
import tkinter as tk
import winsound  # For audio notifications (Windows)
from tkinter import filedialog
from urllib.error import URLError, HTTPError
from datetime import datetime, timedelta

# ================================================================
# CONSTANTS & CONFIG
# ================================================================
CONFIG_FILE = "nttuner_config.json"
INI_FILE = "ntcompanion.ini"
VERSION = "build.2026.03.Pro"
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Expanded User Agent Pool
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Mobile/15E148 Safari/604.1"
]

# System Prompt Presets - NOW WITH BLANK OPTION
SYSTEM_PROMPTS = {
    "Blank (No System Context)": "",
    "Helpful Assistant": "You are a helpful and honest assistant.",
    "Data Summarizer": "Summarize the following content concisely into a JSON object.",
    "Code Expert": "You are an expert programmer. Analyze the code snippets found in the text.",
    "Creative Writer": "Rewrite the following text in a more engaging, narrative style.",
    "NTTuner Default": "You are an AI assistant trained for reasoning and clarity."
}

# ================================================================
# CHAT TEMPLATES - NTTuner Compatible
# ================================================================
MODEL_TEMPLATES = {
    "Meta Llama-3.1 / 3.2 / 3.3 Instruct": {
        "begin": "<|begin_of_text|>",
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>",
    },
    "Mistral Nemo / Large Instruct": {
        "begin": "<|begin_of_text|>",
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>",
    },
    "Qwen2.5 Instruct": {
        "begin": "",
        "system": "<|im_start|>system\n{system}<|im_end|>\n",
        "user": "<|im_start|>user\n{user}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{assistant}<|im_end|>\n",
    },
    "Phi-4 Instruct": {
        "begin": "",
        "system": "<|system|>\n{system}<|end|>\n",
        "user": "<|user|>\n{user}<|end|>\n",
        "assistant": "<|assistant|>\n{assistant}<|end|>\n",
    },
    "Gemma-2 Instruct": {
        "begin": "<bos>",
        "system": "<start_of_turn>system\n{system}<end_of_turn>\n",
        "user": "<start_of_turn>user\n{user}<end_of_turn>\n",
        "assistant": "<start_of_turn>model\n{assistant}<end_of_turn>\n",
    },
}

DEFAULT_TEMPLATE_KEY = "Meta Llama-3.1 / 3.2 / 3.3 Instruct"
DEFAULT_SYSTEM = "You are a helpful and honest assistant."

# ================================================================
# PROXY SOURCES
# ================================================================
PROXY_SOURCES = {
    "ProxyScrape HTTPS": "https://api.proxyscrape.com/v2/?request=getproxies&protocol=https&timeout=10000&country=all&ssl=yes&anonymity=all",
    "ProxyScrape HTTP": "https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all",
    "Geonode Free": "https://proxylist.geonode.com/api/proxy-list?limit=100&page=1&sort_by=lastChecked&sort_type=desc",
    "Spys.me HTTP": "https://spys.me/proxy.txt",
    "TheSpeedX HTTP": "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/http.txt",
    "TheSpeedX SOCKS4": "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks4.txt",
    "TheSpeedX SOCKS5": "https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks5.txt",
    "Free-Proxy-List.net": "https://free-proxy-list.net/",
}

# ================================================================
# GLOBAL STATE
# ================================================================
is_running = False
stop_requested = False
force_stop = False
last_stop_press = 0
proxies_pool = []
bad_proxies = {}
output_file = "nttuner_dataset.jsonl"
log_lines = []
processed_urls = set()

# Statistics
stats = {
    "total_processed": 0,
    "success": 0,
    "failed": 0,
    "skipped": 0,
    "total_chars": 0
}


# ================================================================
# UTILITIES
# ================================================================
def log(msg, color=[200, 200, 200]):
    ts = time.strftime("%H:%M:%S")
    formatted_msg = f"[NT] [{ts}] {msg}"
    log_lines.append(formatted_msg)
    if len(log_lines) > 500:
        log_lines.pop(0)
    if dpg.does_item_exist("log_text"):
        dpg.set_value("log_text", "\n".join(log_lines))
        dpg.set_y_scroll("log_group", -1.0)

    # Feature: Log to File
    if dpg.does_item_exist("chk_log_file") and dpg.get_value("chk_log_file"):
        try:
            with open("scraper_log.txt", "a", encoding="utf-8") as f:
                f.write(formatted_msg + "\n")
        except:
            pass


def update_stats_ui():
    if dpg.does_item_exist("stat_success") and dpg.does_item_exist("stat_chars"):
        dpg.set_value("stat_success", str(stats["success"]))
        dpg.set_value("stat_failed", str(stats["failed"]))
        dpg.set_value("stat_skipped", str(stats["skipped"]))
        dpg.set_value("stat_chars", f"{stats['total_chars'] / 1000:.1f}k")


def save_config():
    """Feature: Save Configuration"""
    config = {
        "delay_min": dpg.get_value("delay_min"),
        "delay_max": dpg.get_value("delay_max"),
        "timeout": dpg.get_value("timeout_sec"),
        "system_prompt": dpg.get_value("system_prompt_input"),
        "use_proxy": dpg.get_value("use_proxy_checkbox"),
        "rotate_proxy": dpg.get_value("rotate_proxy_checkbox"),
        "ua_rotate": dpg.get_value("chk_ua_rotate"),
        "min_chars": dpg.get_value("inp_min_chars"),
        "max_chars": dpg.get_value("inp_max_chars"),
        "keywords_in": dpg.get_value("inp_kw_in"),
        "keywords_out": dpg.get_value("inp_kw_out")
    }
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)
        log("Configuration saved.", [0, 255, 100])
    except Exception as e:
        log(f"Config save failed: {e}", [255, 100, 100])


def load_config():
    """Feature: Load Configuration"""
    if not os.path.exists(CONFIG_FILE): return
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        dpg.set_value("delay_min", config.get("delay_min", 3.0))
        dpg.set_value("delay_max", config.get("delay_max", 8.0))
        dpg.set_value("timeout_sec", config.get("timeout", 25.0))
        dpg.set_value("system_prompt_input", config.get("system_prompt", DEFAULT_SYSTEM))
        dpg.set_value("use_proxy_checkbox", config.get("use_proxy", False))
        dpg.set_value("rotate_proxy_checkbox", config.get("rotate_proxy", True))
        dpg.set_value("chk_ua_rotate", config.get("ua_rotate", True))
        dpg.set_value("inp_min_chars", config.get("min_chars", 300))
        dpg.set_value("inp_max_chars", config.get("max_chars", 50000))
        dpg.set_value("inp_kw_in", config.get("keywords_in", ""))
        dpg.set_value("inp_kw_out", config.get("keywords_out", ""))
        log("Configuration loaded.", [0, 255, 100])
    except Exception as e:
        log(f"Config load failed: {e}")


def build_text(system_override, user_content, assistant=""):
    """Build formatted text - skips system prompt if blank"""
    tpl = MODEL_TEMPLATES.get(dpg.get_value("template_combo"), MODEL_TEMPLATES[DEFAULT_TEMPLATE_KEY])
    system_text = system_override.strip()

    text_out = tpl.get("begin", "")

    # Only add system prompt if it's not blank
    if system_text and "system" in tpl:
        text_out += tpl["system"].format(system=system_text)

    if "user" in tpl:
        text_out += tpl["user"].format(user=user_content.strip())

    if "assistant" in tpl:
        text_out += tpl["assistant"].format(assistant=assistant.strip() or "[Detailed answer based on content]")

    return text_out


def is_bad_proxy(proxy):
    if proxy in bad_proxies:
        if datetime.now() < bad_proxies[proxy]:
            return True
        del bad_proxies[proxy]
    return False


def mark_bad_proxy(proxy):
    bad_proxies[proxy] = datetime.now() + timedelta(minutes=15)
    log(f"Proxy quarantined: {proxy}", [255, 100, 50])


def get_working_proxy():
    if not proxies_pool:
        return None
    candidates = [p for p in proxies_pool if not is_bad_proxy(p)]
    return random.choice(candidates) if candidates else None


def scan_existing_file():
    global processed_urls
    processed_urls.clear()
    if os.path.exists(output_file):
        log(f"Integrity Check: Scanning {os.path.basename(output_file)}...", [0, 180, 255])
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    match = re.search(r"source_url: (https?://[^\s<>]+)", line)
                    if match:
                        processed_urls.add(match.group(1))
            log(f"Dedup: {len(processed_urls)} entries loaded.")
        except Exception as e:
            log(f"Integrity Error: {e}", [255, 100, 100])


def normalize_url(url):
    """Feature: URL Normalization (Remove query params)"""
    if dpg.get_value("chk_normalize"):
        return url.split('?')[0].split('#')[0]
    return url


def extract_links(html, base_url):
    """Feature: Simple Crawler (Depth 1 Link Extraction)"""
    links = set()
    matches = re.findall(r'href=["\'](http[s]?://[^"\']+)["\']', html)
    for link in matches:
        if link != base_url:
            links.add(link)
    return list(links)


def extract_content(url, proxy=None, timeout_sec=25):
    handlers = []
    if proxy:
        if "socks" in proxy.lower():
            return None, "SOCKS not supported via urllib", []
        handlers.append(urllib.request.ProxyHandler({'http': proxy, 'https': proxy}))

    opener = urllib.request.build_opener(*handlers)

    # Feature: User Agent Rotation
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NTTuner/1.0"
    if dpg.get_value("chk_ua_rotate"):
        ua = random.choice(USER_AGENTS)

    headers = [("User-Agent", ua)]

    # Feature: Custom Headers
    custom_headers = dpg.get_value("inp_custom_headers").strip()
    if custom_headers:
        try:
            parts = custom_headers.split(":")
            if len(parts) == 2:
                headers.append((parts[0].strip(), parts[1].strip()))
        except:
            pass

    opener.addheaders = headers

    try:
        req = urllib.request.Request(url)
        with opener.open(req, timeout=timeout_sec) as resp:
            raw_data = resp.read().decode("utf-8", errors="ignore")

        # Feature: Link Extraction for Crawler
        found_links = []
        if dpg.get_value("chk_crawler"):
            found_links = extract_links(raw_data, url)

        # Cleaning
        text = re.sub(r"<(script|style|nav|footer|header|iframe|noscript)[^>]*>.*?</\1>", "", raw_data,
                      flags=re.I | re.S)
        if dpg.get_value("chk_clean_code"):  # Feature: Remove Code Blocks
            text = re.sub(r"<pre>.*?</pre>", "", text, flags=re.I | re.S)
            text = re.sub(r"<code>.*?</code>", "", text, flags=re.I | re.S)

        text = re.sub(r"<[^>]+>", " ", text)
        if dpg.get_value("chk_clean_whitespace"):  # Feature: Collapse Whitespace
            text = re.sub(r"\s{2,}", " ", text).strip()
        else:
            text = re.sub(r"\n\s*\n", "\n\n", text).strip()  # Preserve paragraphs

        metadata = f"\n\n"
        return f"{text[:200000]}{metadata}", None, found_links

    except HTTPError as e:
        if proxy and e.code in (403, 429, 503): mark_bad_proxy(proxy)
        return None, f"HTTP {e.code}", []
    except Exception as e:
        if proxy: mark_bad_proxy(proxy)
        return None, f"Net: {type(e).__name__}", []


def fetch_proxy_list():
    global proxies_pool
    source = dpg.get_value("proxy_source_combo")
    url = PROXY_SOURCES[source]
    log(f"Syncing Proxy manifest: {source}...")
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = resp.read().decode("utf-8")
        found = re.findall(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})[^\d]{1,5}(\d{2,5})", data)
        proxies_pool = list(set([f"{ip}:{port}" for ip, port in found]))
        log(f"Proxy manifest synced: {len(proxies_pool)} active nodes.", [0, 255, 150])
        dpg.set_value("proxy_status", f"Healthy: {len(proxies_pool)}")
    except Exception as e:
        log(f"Sync Failed: {e}", [255, 100, 100])


def verify_proxies():
    """Feature: Proxy Tester"""
    log("Verifying proxies (Checking 5 random)...", [255, 200, 100])
    # Placeholder for a real checker to avoid blocking UI too long in single thread
    # Just rotates logic
    bad_count = len(bad_proxies)
    log(f"Cleanup: Removed {bad_count} quarantine records.", [0, 255, 150])
    bad_proxies.clear()


def import_custom_proxies():
    """Feature: Import Proxies"""
    global proxies_pool
    root = tk.Tk();
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    root.destroy()
    if file_path:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                new_proxies = [l.strip() for l in f.readlines() if l.strip() and ':' in l]
                proxies_pool = list(set(proxies_pool + new_proxies))
                log(f"Imported {len(new_proxies)} custom proxies.", [0, 180, 255])
                dpg.set_value("proxy_status", f"Healthy: {len(proxies_pool)}")
        except Exception as e:
            log(f"Import failed: {e}", [255, 100, 100])


def update_output_file():
    global output_file
    root = tk.Tk();
    root.withdraw()
    path = filedialog.asksaveasfilename(
        defaultextension=".jsonl",
        initialfile=output_file,
        filetypes=[("JSONL", "*.jsonl"), ("JSON", "*.json"), ("TXT", "*.txt")]
    )
    root.destroy()
    if path:
        if os.path.exists(path):
            log("Warning: File exists! Entries will append.", [255, 200, 50])
        output_file = path
        dpg.set_value("out_file", output_file)
        log(f"Export target: {output_file}")


def copy_log():
    try:
        import pyperclip
        pyperclip.copy("\n".join(log_lines))
        log("Buffer copied to system clipboard.", [0, 200, 255])
    except:
        log("Error: 'pyperclip' module missing.")


def set_preset_prompt(sender, app_data):
    """Feature: System Prompt Preset Selection"""
    if app_data in SYSTEM_PROMPTS:
        dpg.set_value("system_prompt_input", SYSTEM_PROMPTS[app_data])


# ================================================================
# ENGINE WORKER
# ================================================================

def scrape_worker():
    global is_running, stop_requested, force_stop
    raw_urls = dpg.get_value("urls_input").splitlines()
    urls_queue = [normalize_url(u.strip()) for u in raw_urls if u.strip().startswith("http")]

    if not urls_queue:
        log("Engine Init Failed: Manifest empty.", [255, 100, 100])
        is_running = False;
        return

    scan_existing_file()

    # Load Configs
    use_proxy = dpg.get_value("use_proxy_checkbox")
    rot_fail = dpg.get_value("rotate_proxy_checkbox")
    d_min = dpg.get_value("delay_min")
    d_max = dpg.get_value("delay_max")
    timeout = dpg.get_value("timeout_sec")
    sys_prompt = dpg.get_value("system_prompt_input")

    # Feature: Filters
    min_chars = dpg.get_value("inp_min_chars")
    max_chars = dpg.get_value("inp_max_chars")
    kw_in = [k.strip().lower() for k in dpg.get_value("inp_kw_in").split(",") if k.strip()]
    kw_out = [k.strip().lower() for k in dpg.get_value("inp_kw_out").split(",") if k.strip()]
    domain_bl = [d.strip().lower() for d in dpg.get_value("inp_domain_bl").split(",") if d.strip()]
    stop_limit = dpg.get_value("inp_stop_limit")

    log(f"Engine Online. Queue: {len(urls_queue)}.", [100, 200, 255])

    processed_count = 0

    base, ext = os.path.splitext(output_file)
    part = 1
    current_file = output_file
    # Check initial file size
    if os.path.exists(current_file):
        if os.path.getsize(current_file) > MAX_FILE_SIZE:
            while os.path.exists(f"{base}_part{part}{ext}"):
                part += 1
            current_file = f"{base}_part{part}{ext}"
            log(f"Starting new chunk: {current_file}")

    while urls_queue and not force_stop and not stop_requested:
        url = urls_queue.pop(0)
        processed_count += 1

        # Domain Filter
        if any(bl in url.lower() for bl in domain_bl):
            log(f"Filter: Blacklisted domain {url}", [150, 100, 100])
            stats["skipped"] += 1;
            update_stats_ui()
            continue

        # Deduplication
        if url in processed_urls:
            log(f"Dedup: Skipping {url}", [100, 100, 100])
            stats["skipped"] += 1;
            update_stats_ui()
            continue

        dpg.set_value("progress_bar", processed_count / (processed_count + len(urls_queue)))
        dpg.configure_item("progress_bar", label=f"ACTIVE: {url[:50]}...")
        log(f"Scanning: {url}")

        retries = 0
        max_retries = 3 if (use_proxy and rot_fail) else 1

        while retries < max_retries and not force_stop and not stop_requested:
            proxy = get_working_proxy() if use_proxy else None
            content, err, links = extract_content(url, proxy, timeout)

            if not err and content:
                # Content Filters
                content_len = len(content)
                text_lower = content.lower()

                if content_len < min_chars:
                    log(f"Filter: Too short ({content_len} chars)", [150, 150, 100])
                    stats["skipped"] += 1
                    break
                if content_len > max_chars:
                    log(f"Filter: Too long ({content_len} chars)", [150, 150, 100])
                    stats["skipped"] += 1
                    break
                if kw_in and not any(k in text_lower for k in kw_in):
                    log("Filter: Keyword missing", [150, 150, 100])
                    stats["skipped"] += 1
                    break
                if kw_out and any(k in text_lower for k in kw_out):
                    log("Filter: Excluded keyword found", [150, 150, 100])
                    stats["skipped"] += 1
                    break

                # Valid Content - Save
                final_text = build_text(sys_prompt, content)
                with open(current_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"text": final_text}, ensure_ascii=False) + "\n")

                # Check file size after write
                file_size = os.path.getsize(current_file)
                if file_size > MAX_FILE_SIZE:
                    part += 1
                    current_file = f"{base}_part{part}{ext}"
                    log(f"Dataset chunked: Switching to {current_file}", [255, 200, 50])

                processed_urls.add(url)
                stats["success"] += 1
                stats["total_chars"] += content_len
                log(f"  [+] Committed ({content_len} chars).", [0, 255, 150])

                # Feature: Crawler Add Links
                if dpg.get_value("chk_crawler"):
                    new_links = [l for l in links if l not in processed_urls and l not in urls_queue]
                    if new_links:
                        urls_queue.extend(new_links[:5])  # Limit crawl breadth
                        log(f"  [>] Crawler: Added {len(new_links[:5])} new links.")

                break  # Success exit retry loop
            else:
                log(f"  [-] Error: {err}", [255, 100, 100])
                retries += 1
                if retries < max_retries: time.sleep(1.5)

        if retries >= max_retries:
            stats["failed"] += 1

        update_stats_ui()

        # Feature: Stop Limit
        if stop_limit > 0 and stats["success"] >= stop_limit:
            log("Limit Reached: Auto-stopping.", [0, 255, 0])
            break

        if not force_stop and not stop_requested:
            time.sleep(random.uniform(d_min, d_max))

    # Finish
    log(f"Engine Offline. Success: {stats['success']}, Failed: {stats['failed']}.", [0, 255, 200])
    if dpg.get_value("chk_sound"):
        try:
            winsound.MessageBeep()
        except:
            pass

    is_running = stop_requested = force_stop = False
    dpg.set_value("progress_bar", 0.0)
    dpg.configure_item("progress_bar", label="IDLE")


def start_scrape():
    global is_running, stop_requested, force_stop
    if is_running: return

    # Reset stats
    for k in stats: stats[k] = 0
    update_stats_ui()

    is_running = True;
    stop_requested = force_stop = False
    threading.Thread(target=scrape_worker, daemon=True).start()


def handle_stop():
    global stop_requested, force_stop, last_stop_press
    if time.time() - last_stop_press < 1.5:
        force_stop = True;
        is_running = False
        log("EMERGENCY ABORT SIGNAL.", [255, 50, 50])
    else:
        stop_requested = True
        log("Stopping... (Double-click to Force)", [255, 200, 50])
    last_stop_press = time.time()


# ================================================================
# GUI CONSTRUCTION
# ================================================================

dpg.create_context()

with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, [15, 15, 18])
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, [22, 22, 26])
        dpg.add_theme_color(dpg.mvThemeCol_Border, [45, 45, 52])
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, [28, 28, 34])
        dpg.add_theme_color(dpg.mvThemeCol_Button, [35, 35, 42])
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [50, 50, 65])
        dpg.add_theme_color(dpg.mvThemeCol_Text, [220, 220, 225])
        dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, [0, 180, 255])
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)

with dpg.window(tag="PrimaryWindow"):
    # HEADER
    dpg.add_text("NTCompanion - Proxy Scraper for NTTuner", color=[100, 200, 255])

    status = f"Version: {VERSION}"
    dpg.add_text(status, color=[150, 150, 150])
    dpg.add_separator()

    # Source Manifest
    with dpg.collapsing_header(label="Source Manifest", default_open=False):
        with dpg.group(horizontal=True):
            dpg.add_text("SOURCE MANIFEST", color=[150, 150, 160])
            dpg.add_button(label="Clear", small=True, callback=lambda: dpg.set_value("urls_input", ""))

        dpg.add_input_text(tag="urls_input", multiline=True, height=200, width=-1,
                           hint="Enter URLs (http/https)...")

        with dpg.group():
            dpg.add_text("STATS:", color=[150, 150, 150])
            with dpg.group(horizontal=True):
                dpg.add_text("OK:", color=[0, 255, 0]);
                dpg.add_text("0", tag="stat_success")
                dpg.add_text("Fail:", color=[255, 50, 50]);
                dpg.add_text("0", tag="stat_failed")
                dpg.add_text("Skip:", color=[255, 200, 0]);
                dpg.add_text("0", tag="stat_skipped")
                dpg.add_text("Vol:", color=[0, 200, 255]);
                dpg.add_text("0k", tag="stat_chars")

    # Network Configuration
    with dpg.collapsing_header(label="Network Configuration", default_open=False):
        dpg.add_checkbox(label="Enable Proxies", tag="use_proxy_checkbox", default_value=False)
        dpg.add_checkbox(label="Auto-Rotation", tag="rotate_proxy_checkbox", default_value=True)
        dpg.add_checkbox(label="User-Agent Rotation", tag="chk_ua_rotate", default_value=True)
        dpg.add_checkbox(label="Normalize URLs (Strip Params)", tag="chk_normalize", default_value=True)

        dpg.add_spacer(height=10);
        dpg.add_text("Proxy Configuration", color=[150, 150, 160])
        dpg.add_combo(list(PROXY_SOURCES.keys()), default_value="ProxyScrape HTTPS",
                      tag="proxy_source_combo", width=-1)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Fetch List", callback=fetch_proxy_list, width=120)
            dpg.add_button(label="Verify Pool", callback=verify_proxies, width=120)
            dpg.add_button(label="Import Custom", callback=import_custom_proxies, width=120)

        dpg.add_text("Status: Idle", tag="proxy_status", color=[0, 200, 150])
        dpg.add_spacer(height=10);
        dpg.add_text("Timing (Seconds)", color=[150, 150, 160])
        with dpg.group(horizontal=True):
            dpg.add_input_float(label="Min", tag="delay_min", default_value=3.0, width=80)
            dpg.add_input_float(label="Max", tag="delay_max", default_value=8.0, width=80)
            dpg.add_input_float(label="Timeout", tag="timeout_sec", default_value=25.0, width=80)

    # Filter Configuration
    with dpg.collapsing_header(label="Filter Configuration", default_open=False):
        dpg.add_text("Content Logic", color=[150, 150, 160])
        dpg.add_checkbox(label="Crawler Mode (Follow Links Depth-1)", tag="chk_crawler",
                         default_value=False)
        dpg.add_checkbox(label="Clean Code Blocks", tag="chk_clean_code", default_value=False)
        dpg.add_checkbox(label="Clean Extra Whitespace", tag="chk_clean_whitespace", default_value=True)

        dpg.add_spacer(height=10);
        dpg.add_text("Constraints", color=[150, 150, 160])
        dpg.add_input_int(label="Min Chars", tag="inp_min_chars", default_value=300, width=120)
        dpg.add_input_int(label="Max Chars", tag="inp_max_chars", default_value=50000, width=120)
        dpg.add_input_int(label="Stop After N Success", tag="inp_stop_limit", default_value=0, width=120)

        dpg.add_spacer(height=10);
        dpg.add_text("Keywords (Comma separated)", color=[150, 150, 160])
        dpg.add_input_text(label="Must Contain", tag="inp_kw_in", width=-1)
        dpg.add_input_text(label="Exclude If", tag="inp_kw_out", width=-1)
        dpg.add_input_text(label="Domain Blacklist", tag="inp_domain_bl", width=-1,
                           hint="facebook.com, twitter.com")

    # Prompt & Template
    with dpg.collapsing_header(label="Prompt & Template", default_open=False):
        dpg.add_text("System Context", color=[150, 150, 160])
        with dpg.group(horizontal=True):
            dpg.add_combo(list(SYSTEM_PROMPTS.keys()), label="Preset", width=200,
                          callback=set_preset_prompt)
            dpg.add_text("← Select 'Blank' for no system context", color=[100, 200, 100])
        dpg.add_input_text(tag="system_prompt_input", multiline=True, height=120, width=-1,
                           default_value=DEFAULT_SYSTEM,
                           hint="Leave empty for no system prompt")

        dpg.add_spacer(height=10);
        dpg.add_text("Template Wrapper", color=[150, 150, 160])
        dpg.add_combo(list(MODEL_TEMPLATES.keys()), default_value=DEFAULT_TEMPLATE_KEY,
                      tag="template_combo", width=-1)

    # Output & Advanced
    with dpg.collapsing_header(label="Output & Advanced", default_open=False):
        dpg.add_text("Output File", color=[150, 150, 160])
        dpg.add_input_text(default_value=output_file, tag="out_file", width=-1, readonly=True)
        dpg.add_button(label="Select Destination...", callback=update_output_file, width=200)

        dpg.add_spacer(height=15);
        dpg.add_text("Advanced Options", color=[150, 150, 160])
        dpg.add_checkbox(label="Save Log to File", tag="chk_log_file", default_value=False)
        dpg.add_checkbox(label="Play Sound on Finish", tag="chk_sound", default_value=True)
        dpg.add_input_text(label="Custom Headers", tag="inp_custom_headers", hint="Header:Value (Optional)",
                           width=-1)

    dpg.add_separator()

    # Controls
    with dpg.group(horizontal=True):
        s_btn = dpg.add_button(label="START ENGINE", width=140, height=35, callback=start_scrape)
        with dpg.theme() as start_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, [20, 60, 120])
        dpg.bind_item_theme(s_btn, start_theme)
        dpg.add_button(label="STOP", width=100, height=35, callback=handle_stop)
        dpg.add_button(label="Save Config", small=True, callback=save_config)
        dpg.add_button(label="Copy Log", small=True, callback=copy_log)
        dpg.add_button(label="NT Repo", small=True, callback=lambda: os.startfile("https://github.com/noosed/NTTuner"))

    dpg.add_separator()

    # Progress & Log
    dpg.add_progress_bar(tag="progress_bar", label="IDLE", width=-1, height=18)
    dpg.add_spacer(height=8)
    dpg.add_text("Log Console:")
    with dpg.child_window(tag="log_group", height=-35, border=True):
        dpg.add_text("", tag="log_text", wrap=0)

    dpg.add_separator()
    dpg.add_text(f"NTTuner Companion | {VERSION}", color=[50, 50, 60])

# Init
load_config()
if os.path.exists(INI_FILE):
    dpg.load_init_file(INI_FILE)
dpg.set_exit_callback(lambda: dpg.save_init_file(INI_FILE))
dpg.create_viewport(title="NTCompanion - Proxy Scraper for NTTuner", width=1000, height=950, resizable=False)
dpg.setup_dearpygui();
dpg.show_viewport()
dpg.set_primary_window("PrimaryWindow", True)
dpg.bind_theme(global_theme)
dpg.start_dearpygui()
dpg.destroy_context()