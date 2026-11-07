"""Download model files into ComfyUI's folders, one explicit request at a time."""
import base64
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import shutil
from threading import Event, Lock, Thread
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
import urllib.request
from uuid import uuid4

from .inventory import InventoryError, check_gguf, inspect_file
from .model_sources import INSTALLABLE_SUFFIXES, blocked_reason, file_url, guess_folder, huggingface_parts, install_note, load_index, template_thumbnail

API = "https://huggingface.co/api/models"
CHUNK = 1024 * 1024
MAX_JOBS = 32
MAX_PLAN_ITEMS = 32
MAX_RESULTS = 100
MAX_WORKFLOWS = 1000  # ComfyUI ships a few hundred; all of them are listed
MAX_THUMBNAILS = 400
THUMBNAIL_SIDE = 320
MAX_RAW_THUMBNAIL = 1024 * 1024
MAX_API_BYTES = 8 * 1024 * 1024
RESERVE_BYTES = 256 * 1024 * 1024
PART_SUFFIX = ".modelrun-part"
RESERVED = re.compile(r"(con|prn|aux|nul|com\d|lpt\d)(\..*)?", re.I)
ACTIVE = ("queued", "downloading", "verifying")


def _allowed_host(host):
    return host in ("huggingface.co", "hf.co") or host.endswith((".huggingface.co", ".hf.co"))


def _token():
    """A login the person already has; only ever sent to huggingface.co, never stored or returned."""
    value = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if value:
        return value.strip()
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    try:
        with open(os.path.join(home, "token"), encoding="utf-8") as stream:
            return stream.read(1024).strip() or None
    except OSError:
        return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Transport:
    """Opens https requests to allowed hosts only and checks every redirect itself."""
    def __init__(self):
        self._opener = urllib.request.build_opener(_NoRedirect)

    def open(self, url, *, method="GET", headers=None, timeout=60, redirects=True):
        for _ in range(6):
            parts = urlsplit(url)
            if parts.scheme != "https" or not _allowed_host(parts.hostname or "") or parts.port is not None:
                raise ValueError("A redirect left Hugging Face; the download was refused")
            sent = dict(headers or {})
            token = _token()
            if token and parts.hostname == "huggingface.co":
                sent["Authorization"] = f"Bearer {token}"
            try:
                response = self._opener.open(urllib.request.Request(url, method=method, headers=sent), timeout=timeout)
            except HTTPError as exc:
                response = exc
            if not redirects or response.code not in (301, 302, 303, 307, 308):
                return response
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise ValueError("Redirect without a location")
            url = location if "://" in location else f"{parts.scheme}://{parts.netloc}{location}"
        raise ValueError("Too many redirects")


def _safe_part(part):
    if (not part or len(part) > 200 or part in (".", "..") or part != part.strip(" .") or RESERVED.fullmatch(part)
            or re.search(r'[<>:"|?*\\/\x00-\x1f]', part)):
        raise ValueError(f"Unsafe file or folder name: {part[:64]!r}")
    return part


def _match_tier(terms, names, tags, prose):
    """0 when every term is in the name, 1 when each is in a name or tag, 2 when only the description has some; None when a term is missing."""
    if not terms:
        return 0
    names, tags, prose = (" ".join(parts).lower() for parts in (names, tags, prose))
    if all(term in names for term in terms):
        return 0
    if all(term in names or term in tags for term in terms):
        return 1
    if all(term in names or term in tags or term in prose for term in terms):
        return 2
    return None


class _Cancelled(Exception):
    pass


class ModelInstaller:
    def __init__(self, folders, *, comfy_dir=None, on_installed=None, transport=None, index=None):
        self._folders, self._comfy_dir, self._on_installed = folders, comfy_dir, on_installed
        self._transport = transport or Transport()
        self._index = index
        self._lock = Lock()
        self._jobs = {}
        self._thumbnails = {}
        self._queue = []
        self._wake = Event()
        self._closed = False
        self._rescan = False
        self._worker = None

    def index(self):
        # Reading the lists takes seconds, so status polls must not wait on it.
        if self._index is None:
            self._index = load_index(self._comfy_dir)
        return self._index

    def categories(self):
        grouped = {}
        for entry in self._folders():
            grouped.setdefault(entry["category"], [])
            if entry["path"] not in grouped[entry["category"]]:
                grouped[entry["category"]].append(entry["path"])
        return grouped

    def _is_installed(self, item, names, grouped):
        if (item["category"], item["filename"]) in names:
            return True
        return any((Path(folder) / item["subfolder"] / item["filename"]).exists() for folder in grouped.get(item["category"], []))

    def _public(self, item, names, grouped):
        reason = item["blocked"] or (None if item["category"] in grouped else f"ComfyUI has no {item['category']} folder here; a custom node may need to add it")
        return {**item, "blocked": reason, "installed": self._is_installed(item, names, grouped)}

    def search(self, query="", *, category=None, source=None, missing_only=False, names=frozenset()):
        if not isinstance(query, str) or len(query) > 200:
            raise ValueError("Search must be under 200 characters")
        index, grouped = self.index(), self.categories()
        terms = query.lower().split()
        found = []
        for item in index["models"].values():
            if category and item["category"] != category or source and source not in item["sources"]:
                continue
            tier = _match_tier(terms, [item["filename"], item.get("display", "")], [item["base"], item["category"]],
                               [item["description"], *[w["title"] for w in item["workflows"]]])
            if tier is not None:
                public = {**self._public(item, names, grouped), "_tier": tier}
                if not (missing_only and public["installed"]):
                    found.append(public)
        found.sort(key=lambda m: (m.pop("_tier"), bool(m["blocked"]), -len(m["workflows"]), m["filename"].lower()))
        return {"total": len(found), "models": found[:MAX_RESULTS], "categories": sorted(grouped), "notes": index["notes"], "logged_in": bool(_token())}

    def workflows(self, query="", *, kind=None, names=frozenset()):
        """Workflow templates nearest to ready first: everything installed, then the fewest files missing."""
        if not isinstance(query, str) or len(query) > 200:
            raise ValueError("Search must be under 200 characters")
        index, grouped = self.index(), self.categories()
        terms = query.lower().split()
        found, kinds = [], {}
        for workflow in index["workflows"]:
            files = [index["by_id"][i]["filename"] for i in workflow["models"]]
            tier = _match_tier(terms, [workflow["title"], workflow["name"]], [workflow.get("kind", ""), *files], [workflow["description"]])
            if tier is None:
                continue
            label = workflow.get("kind") or "Other"
            kinds[label] = kinds.get(label, 0) + 1
            if kind and label != kind:
                continue
            models = [self._public(index["by_id"][i], names, grouped) for i in workflow["models"]]
            missing = [m for m in models if not m["installed"]]
            sizes = [m["size_bytes"] for m in missing]
            found.append({**workflow, "_tier": tier, "kind": label, "models": models, "missing": len(missing),
                          "installable": sum(not m["blocked"] for m in missing),
                          "missing_bytes": sum(sizes) if missing and all(isinstance(s, int) for s in sizes) else None})
        found.sort(key=lambda w: (w.pop("_tier"), w["missing"] > w["installable"], w["missing"], -(w.get("rank") or 0), w["title"].lower()))
        return {"total": len(found), "workflows": found[:MAX_WORKFLOWS], "kinds": [{"kind": k, "count": n} for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])],
                "ready": sum(not w["missing"] for w in found), "notes": index["notes"]}

    def sizes(self, ids):
        """Ask Hugging Face for the size of listed files the lists leave unsized, and remember the answers."""
        if not isinstance(ids, list) or not 1 <= len(ids) <= MAX_PLAN_ITEMS:
            raise ValueError(f"Ask for 1 to {MAX_PLAN_ITEMS} files at a time")
        by_id = self.index()["by_id"]
        items = list({id(by_id[i]): by_id[i] for i in ids if isinstance(i, str) and i in by_id and not by_id[i]["blocked"]}.values())

        def size(item):
            if item["size_bytes"] is None:
                try:
                    item["size_bytes"] = self._remote(item["url"]).get("size_bytes")
                except (OSError, ValueError):
                    pass
            return item
        with ThreadPoolExecutor(max_workers=6) as pool:
            return {"sizes": {item["id"]: item["size_bytes"] for item in pool.map(size, items)}}

    def thumbnail(self, name):
        """A small still preview of a workflow template from ComfyUI's own media, or None. Needs Pillow."""
        with self._lock:
            if name in self._thumbnails:
                return self._thumbnails[name]
        path = template_thumbnail(name, self._comfy_dir)
        result = None
        if path is not None:
            try:
                from PIL import Image
                with Image.open(path) as image:
                    image.seek(0)  # an animated preview keeps its first frame
                    still = image.convert("RGB")
                    still.thumbnail((THUMBNAIL_SIDE, THUMBNAIL_SIDE))
                    buffer = io.BytesIO()
                    still.save(buffer, "JPEG", quality=80)
                result = {"mime": "image/jpeg", "base64": base64.b64encode(buffer.getvalue()).decode()}
            except ImportError:
                # Without Pillow a small original is sent as it is; a large one is left out.
                if os.path.getsize(path) <= MAX_RAW_THUMBNAIL:
                    result = {"mime": "image/webp", "base64": base64.b64encode(Path(path).read_bytes()).decode()}
            except (OSError, ValueError):
                result = None
        with self._lock:
            if len(self._thumbnails) >= MAX_THUMBNAILS:
                self._thumbnails.pop(next(iter(self._thumbnails)))
            self._thumbnails[name] = result
        return result

    def clear_finished(self):
        """Forget downloads that are no longer running; files on disk are not touched."""
        with self._lock:
            for key in [key for key, job in self._jobs.items() if job["status"] not in ACTIVE]:
                del self._jobs[key]
        return self.status()

    def _api(self, url):
        response = self._transport.open(url, timeout=20)
        try:
            if response.code != 200:
                raise RuntimeError(f"Hugging Face answered {response.code}")
            data = response.read(MAX_API_BYTES + 1)
        finally:
            response.close()
        if len(data) > MAX_API_BYTES:
            raise ValueError("Hugging Face answer was too large")
        return json.loads(data)

    def hub_search(self, query):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 200:
            raise ValueError("Enter something to search for")
        # Hugging Face filters by one tag at a time, so repos with GGUF files are asked for separately.
        found = {}
        for tag in ("safetensors", "gguf"):
            params = urlencode({"search": query.strip(), "filter": tag, "sort": "downloads", "direction": "-1", "limit": 24})
            for r in self._api(f"{API}?{params}"):
                if not isinstance(r, dict) or not isinstance(r.get("id"), str):
                    continue
                entry = found.setdefault(r["id"], {"repo": r["id"], "downloads": r.get("downloads"), "likes": r.get("likes"),
                                                   "gated": bool(r.get("gated")), "task": r.get("pipeline_tag"), "formats": []})
                entry["formats"].append(tag)
        ranked = sorted(found.values(), key=lambda r: -(r["downloads"] if isinstance(r["downloads"], int) else 0))
        return {"repos": ranked[:24]}

    def hub_files(self, repo, *, names=frozenset()):
        if not isinstance(repo, str) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or ".." in repo:
            raise ValueError("Invalid repository name")
        files = self._api(f"{API}/{quote(repo)}/tree/main?recursive=true")
        grouped = self.categories()
        found = []
        for item in files if isinstance(files, list) else []:
            path = item.get("path") if isinstance(item, dict) else None
            if not isinstance(path, str) or item.get("type") != "file" or not path.lower().endswith(INSTALLABLE_SUFFIXES):
                continue
            lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else {}
            oid = lfs.get("oid") if isinstance(lfs.get("oid"), str) else None
            filename = path.rsplit("/", 1)[-1]
            category = guess_folder(path)
            found.append({"id": sha256(f"{repo}/{path}".encode()).hexdigest()[:16], "name": path, "filename": filename, "url": file_url(repo, path),
                          "category": category, "guessed_folder": True, "subfolder": "", "size_bytes": item.get("size"), "sha256": oid,
                          "installed": (category, filename) in names, "blocked": None if category in grouped else f"ComfyUI has no {category} folder here",
                          "note": install_note(filename)})
        return {"repo": repo, "files": found[:MAX_RESULTS], "total": len(found), "categories": sorted(grouped)}

    def _target(self, request):
        """Check a request completely and work out where it would go."""
        if not isinstance(request, dict):
            raise ValueError("Each install must be an object")
        url, category, filename = request.get("url"), request.get("category"), request.get("filename")
        subfolder = request.get("subfolder") or ""
        if not all(isinstance(v, str) for v in (url, category, filename, subfolder)):
            raise ValueError("Install requests need url, category and filename")
        reason = blocked_reason(url, filename)
        if reason:
            raise ValueError(reason)
        folders = self.categories().get(category)
        if not folders:
            raise ValueError(f"ComfyUI has no {category!r} folder here")
        folder = request.get("folder")
        if folder is None:
            named = [f for f in folders if Path(f).name == category]
            folder = (named or folders)[0]
        if folder not in folders:
            raise ValueError("That folder is not one of ComfyUI's folders for this type")
        root = Path(folder).resolve()
        parts = [_safe_part(p) for p in subfolder.replace("\\", "/").split("/") if p] + [_safe_part(filename)]
        target = root.joinpath(*parts).resolve()
        if not target.is_relative_to(root):
            raise ValueError("That path is outside ComfyUI's folder")
        return url.split("?")[0].replace("https://hf.co/", "https://huggingface.co/", 1), target, folders

    def _remote(self, url):
        """Size, published hash and access for a file, without downloading it."""
        response = self._transport.open(url, method="HEAD", timeout=30, redirects=False, headers={"Accept-Encoding": "identity"})
        try:
            location = response.headers.get("Location", "")
            if response.code in (301, 307, 308) and location.startswith("/"):
                response.close()
                response = self._transport.open(f"https://huggingface.co{location}", method="HEAD", timeout=30, redirects=False, headers={"Accept-Encoding": "identity"})
            headers, status = response.headers, response.code
        finally:
            response.close()
        if status in (401, 403):
            repo = huggingface_parts(url)[0]
            return {"access": "gated", "reason": f"Accept the licence on huggingface.co/{repo} and log in with the Hugging Face CLI, then try again"}
        if status == 404:
            return {"access": "missing", "reason": "Hugging Face no longer has this file"}
        if status not in (200, 301, 302, 303, 307, 308):
            return {"access": "error", "reason": f"Hugging Face answered {status}"}
        size = headers.get("X-Linked-Size") or headers.get("Content-Length")
        linked = (headers.get("X-Linked-ETag") or "").replace("W/", "").strip('" ').lower()
        return {"access": "ok", "size_bytes": int(size) if size and size.isdigit() else None, "sha256": linked if re.fullmatch(r"[0-9a-f]{64}", linked) else None}

    def _plan_one(self, request):
        result = {k: request.get(k) for k in ("url", "category", "subfolder", "filename")} if isinstance(request, dict) else {}
        try:
            url, target, folders = self._target(request)
        except (ValueError, TypeError) as exc:
            return {**result, "ok": False, "reason": str(exc)[:400]}
        result.update(url=url, destination=str(target), folders=folders, exists=target.exists())
        try:
            remote = self._remote(url)
        except (OSError, ValueError) as exc:
            return {**result, "ok": False, "reason": f"Could not reach Hugging Face: {exc}"[:400]}
        result.update(remote)
        listed = request.get("sha256")
        result["sha256"] = remote.get("sha256") or (listed if isinstance(listed, str) and re.fullmatch(r"[0-9a-f]{64}", listed) else None)
        anchor = target.parent
        while not anchor.exists() and anchor.parent != anchor:
            anchor = anchor.parent
        result["free_bytes"] = shutil.disk_usage(anchor).free
        part = target.with_name(target.name + PART_SUFFIX)
        result["resume_bytes"] = part.stat().st_size if part.exists() else 0
        # ComfyUI finds a model in any of a type's folders, so a copy in another one is still installed.
        relative = Path((request.get("subfolder") or "").replace("\\", "/")) / request["filename"]
        places = [target] if request.get("folder") is not None else [target, *(Path(f) / relative for f in folders)]
        problem = remote.get("reason") or ("This file is already installed" if any(p.exists() for p in places) else None)
        size = remote.get("size_bytes")
        if not problem and size is None:
            problem = "Hugging Face did not report a size for this file"
        if not problem and size - result["resume_bytes"] + RESERVE_BYTES > result["free_bytes"]:
            problem = "Not enough free disk space"
        result.update(ok=not problem, reason=problem)
        return result

    def plan(self, requests):
        if not isinstance(requests, list) or not 1 <= len(requests) <= MAX_PLAN_ITEMS:
            raise ValueError(f"Plan 1 to {MAX_PLAN_ITEMS} files at a time")
        with ThreadPoolExecutor(max_workers=6) as pool:
            items = list(pool.map(self._plan_one, requests))
        return {"items": items, "total_bytes": sum(i.get("size_bytes") or 0 for i in items if i.get("ok"))}

    def start(self, requests):
        """Queue files the person confirmed. A file that changed since then is refused."""
        planned = self.plan(requests)["items"]
        refused = [p for p in planned if not p["ok"]]
        if refused:
            raise ValueError(f"{refused[0].get('filename')}: {refused[0]['reason']}")
        for request, item in zip(requests, planned):
            if request.get("confirmed_bytes") != item["size_bytes"]:
                raise ValueError(f"{item['filename']} changed size since you confirmed it; check it again")
        with self._lock:
            if self._closed:
                raise RuntimeError("Installer is closed")
            active = {j["destination"] for j in self._jobs.values() if j["status"] in ACTIVE}
            if len({i["destination"] for i in planned}) < len(planned) or any(item["destination"] in active for item in planned):
                raise RuntimeError("That file is already downloading")
            jobs = []
            for item in planned:
                job = {"id": uuid4().hex, "filename": item["filename"], "category": item["category"], "destination": item["destination"], "url": item["url"], "status": "queued",
                       "bytes": 0, "total_bytes": item["size_bytes"], "sha256": item.get("sha256"), "hash_checked": False, "error": None, "cancel": False, "discard": False}
                self._jobs[job["id"]] = job
                self._queue.append(job["id"])
                jobs.append(job["id"])
            self._trim()
            if self._worker is None or not self._worker.is_alive():
                self._worker = Thread(target=self._run, name="model-run-install", daemon=True)
                self._worker.start()
            self._wake.set()
        return {"jobs": jobs}

    def _trim(self):
        finished = [key for key, job in self._jobs.items() if job["status"] not in ACTIVE]
        for key in finished[:max(0, len(self._jobs) - MAX_JOBS)]:
            del self._jobs[key]

    def status(self):
        with self._lock:
            return {"jobs": [{k: v for k, v in job.items() if k not in ("cancel", "discard")} for job in self._jobs.values()]}

    def cancel(self, job_id, *, discard=False):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise LookupError("Unknown install")
            if job["status"] == "queued":
                self._queue.remove(job_id)
                job["status"] = "cancelled"
            elif job["status"] in ACTIVE:
                job.update(cancel=True, discard=bool(discard))
            elif discard:
                self._remove_part(job)
        return {"cancelled": job_id}

    @staticmethod
    def _remove_part(job):
        target = Path(job["destination"])
        target.with_name(target.name + PART_SUFFIX).unlink(missing_ok=True)

    def take_rescan(self):
        with self._lock:
            value, self._rescan = self._rescan, False
            return value

    def request_rescan(self):
        with self._lock:
            self._rescan = True

    def close(self):
        with self._lock:
            self._closed = True
            for job in self._jobs.values():
                if job["status"] in ACTIVE:
                    job["cancel"] = True
        self._wake.set()

    def _set(self, job, **values):
        with self._lock:
            job.update(values)

    def _run(self):
        while True:
            with self._lock:
                if self._closed:
                    return
                job = self._jobs[self._queue.pop(0)] if self._queue else None
                if job is None:
                    self._wake.clear()
            if job is None:
                self._wake.wait(30)
                continue
            try:
                self._download(job)
            except _Cancelled:
                if job["discard"]:
                    self._remove_part(job)
                self._set(job, status="cancelled")
            except (OSError, ValueError, URLError, InventoryError, RuntimeError) as exc:
                self._set(job, status="failed", error=f"{exc}"[:400])
            else:
                self._set(job, status="installed")
                self.request_rescan()
                if self._on_installed is not None:
                    self._on_installed()

    def _check_cancel(self, job):
        if job["cancel"]:
            raise _Cancelled()

    def _download(self, job):
        target = Path(job["destination"])
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + PART_SUFFIX)
        digest = sha256()
        have = 0
        if part.exists():
            # Hash what is already here, so a resumed file is still checked end to end.
            self._set(job, status="verifying", bytes=0)
            with part.open("rb") as stream:
                while chunk := stream.read(CHUNK):
                    self._check_cancel(job)
                    digest.update(chunk)
                    have += len(chunk)
                    self._set(job, bytes=have)
        self._set(job, status="downloading", bytes=have)
        if have < job["total_bytes"]:
            headers = {"Accept-Encoding": "identity"}
            if have:
                headers["Range"] = f"bytes={have}-"
            response = self._transport.open(job["url"], headers=headers, timeout=60)
            try:
                if response.code == 200 and have:
                    # The server ignored the range, so start over.
                    digest, have = sha256(), 0
                    part.unlink()
                elif response.code == 206:
                    match = re.match(r"bytes (\d+)-", response.headers.get("Content-Range", ""))
                    if not match or int(match.group(1)) != have:
                        raise ValueError("The server resumed from the wrong place")
                elif response.code != 200:
                    raise ValueError(f"Hugging Face answered {response.code}")
                with part.open("ab") as stream:
                    while chunk := response.read(CHUNK):
                        self._check_cancel(job)
                        stream.write(chunk)
                        digest.update(chunk)
                        have += len(chunk)
                        self._set(job, bytes=have)
                        if have > job["total_bytes"]:
                            part_too_long = True
                            break
                    else:
                        part_too_long = False
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                response.close()
            if part_too_long:
                part.unlink(missing_ok=True)
                raise ValueError("The file was larger than Hugging Face said; it was deleted")
        self._set(job, status="verifying")
        if have != job["total_bytes"]:
            if have > job["total_bytes"]:
                part.unlink(missing_ok=True)
            raise ValueError(f"Download stopped at {have:,} of {job['total_bytes']:,} bytes; retry to resume")
        checked = False
        if job["sha256"]:
            if digest.hexdigest() != job["sha256"]:
                part.unlink(missing_ok=True)
                raise ValueError("The file's SHA-256 does not match the published one; it was deleted")
            checked = True
        gguf = target.name.lower().endswith(".gguf")
        try:
            (check_gguf if gguf else inspect_file)(part)
        except (InventoryError, ValueError, OSError) as exc:
            part.unlink(missing_ok=True)
            raise ValueError(f"This is not a valid {'GGUF' if gguf else 'safetensors'} file, so it was deleted: {exc}") from exc
        if target.exists():
            raise ValueError("A file with this name appeared during the download; it was left alone")
        # Neither call replaces an existing file.
        if os.name == "nt":
            os.rename(part, target)
        else:
            os.link(part, target)
            part.unlink()
        self._set(job, hash_checked=checked)
