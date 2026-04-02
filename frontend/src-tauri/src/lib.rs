use std::io::Write;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

static FLASK_PROCESS: Mutex<Option<Child>> = Mutex::new(None);

// ── Logging ────────────────────────────────────────────────────────────────

fn get_log_path() -> std::path::PathBuf {
    std::env::temp_dir().join("hatsun_startup.log")
}

fn log(msg: &str) {
    let path = get_log_path();
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = writeln!(f, "[{}] {}", ts, msg);
    }
    eprintln!("[HATSUN] {}", msg);
}

// ── Strip Windows \\?\ long-path UNC prefix ────────────────────────────────
// CreateProcess / current_dir does NOT accept \\?\ paths on Windows.

fn strip_unc(path: &std::path::Path) -> std::path::PathBuf {
    let s = path.to_string_lossy();
    if s.starts_with(r"\\?\") {
        std::path::PathBuf::from(s[4..].to_string())
    } else {
        path.to_path_buf()
    }
}

// ── Writable data dir ──────────────────────────────────────────────────────

fn get_data_dir(app: &tauri::App) -> std::path::PathBuf {
    if let Ok(d) = app.path().app_local_data_dir() {
        return strip_unc(&d);
    }
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        if !local.is_empty() {
            return std::path::PathBuf::from(local).join("HatsunVRP");
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return std::path::PathBuf::from(home)
                .join("Library").join("Application Support").join("HatsunVRP");
        }
    }
    std::env::temp_dir().join("HatsunVRP")
}

// ── Find script dir ─────────────────────────────────────────────────────────
// Tauri v2 bundles resources depending on how the glob is written.
// The files may land flat in resource_dir/ OR in resource_dir/script/.
// We probe both.

fn find_script_dir(app: &tauri::App) -> Option<std::path::PathBuf> {
    let resource_dir = match app.path().resource_dir() {
        Ok(d)  => strip_unc(&d),
        Err(e) => { log(&format!("resource_dir() failed: {}", e)); return None; }
    };
    log(&format!("resource_dir (stripped) = {:?}", resource_dir));

    // Probe 1: resource_dir/script/server.py  (expected when glob preserves folder)
    let candidate_subdir = resource_dir.join("script");
    if candidate_subdir.join("server.py").exists() {
        log(&format!("script_dir found at subdir: {:?}", candidate_subdir));
        return Some(candidate_subdir);
    }
    log(&format!("  subdir {:?} — server.py not there", candidate_subdir));

    // Probe 2: resource_dir/server.py  (flat bundle)
    if resource_dir.join("server.py").exists() {
        log(&format!("script_dir found flat at: {:?}", resource_dir));
        return Some(resource_dir);
    }
    log(&format!("  flat {:?} — server.py not there either", resource_dir));

    // Probe 3: same dir as the executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let exe_dir = strip_unc(exe_dir);
            if exe_dir.join("server.py").exists() {
                log(&format!("script_dir found at exe_dir: {:?}", exe_dir));
                return Some(exe_dir);
            }
            let exe_sub = exe_dir.join("script");
            if exe_sub.join("server.py").exists() {
                log(&format!("script_dir found at exe_dir/script: {:?}", exe_sub));
                return Some(exe_sub);
            }
            log(&format!("  exe_dir {:?} — server.py not there", exe_dir));
        }
    }

    None
}

// ── Python finder ──────────────────────────────────────────────────────────

fn find_python() -> Option<String> {
    let mut candidates: Vec<String> = vec![
        "python".into(),
        "python3".into(),
        "python.exe".into(),
        "python3.exe".into(),
        r"C:\Python312\python.exe".into(),
        r"C:\Python311\python.exe".into(),
        r"C:\Python310\python.exe".into(),
        r"C:\Python39\python.exe".into(),
        "/usr/bin/python3".into(),
        "/usr/local/bin/python3".into(),
        "/opt/homebrew/bin/python3".into(),
    ];

    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let root = std::path::Path::new(&local).join("Programs").join("Python");
        if let Ok(entries) = std::fs::read_dir(&root) {
            for entry in entries.flatten() {
                candidates.push(entry.path().join("python.exe").to_string_lossy().to_string());
            }
        }
    }
    if let Ok(profile) = std::env::var("USERPROFILE") {
        let root = std::path::Path::new(&profile)
            .join("AppData").join("Local").join("Programs").join("Python");
        if let Ok(entries) = std::fs::read_dir(&root) {
            for entry in entries.flatten() {
                candidates.push(entry.path().join("python.exe").to_string_lossy().to_string());
            }
        }
    }

    log(&format!("searching {} python candidates", candidates.len()));
    for candidate in &candidates {
        match Command::new(candidate).args(["--version"]).output() {
            Ok(out) if out.status.success() => {
                let ver = String::from_utf8_lossy(&out.stdout).trim().to_string();
                log(&format!("  FOUND: {} -> {}", candidate, ver));
                return Some(candidate.clone());
            }
            Err(e) => log(&format!("  not found: {} -> {}", candidate, e)),
            _ => {}
        }
    }
    None
}

// ── Tauri app ──────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::new().build())
        .setup(|app| {
            let _ = std::fs::write(get_log_path(), "");
            log("=== HatsunVRP startup ===");

            // ── Data dir (writable, for uploads + log copy) ────────────────
            let data_dir = get_data_dir(app);
            match std::fs::create_dir_all(&data_dir) {
                Ok(_)  => log(&format!("data_dir OK: {:?}", data_dir)),
                Err(e) => log(&format!("data_dir FAILED: {:?} -> {}", data_dir, e)),
            }

            // ── Dev vs production script dir ───────────────────────────────
            let script_dir = if cfg!(debug_assertions) {
                let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                manifest_dir.join("..").join("..").join("script")
            } else {
                match find_script_dir(app) {
                    Some(d) => d,
                    None => {
                        log("ERROR: server.py not found anywhere — Flask cannot start");
                        // Copy log to data_dir for easy discovery
                        if let Ok(c) = std::fs::read_to_string(get_log_path()) {
                            let _ = std::fs::write(data_dir.join("startup.log"), c);
                        }
                        return Ok(());
                    }
                }
            };

            let server_path = script_dir.join("server.py");
            log(&format!("script_dir  = {:?}", script_dir));
            log(&format!("server_path = {:?}", server_path));

            // ── Find Python ────────────────────────────────────────────────
            let python_exe = match find_python() {
                Some(p) => p,
                None => {
                    log("ERROR: Python not found");
                    if let Ok(c) = std::fs::read_to_string(get_log_path()) {
                        let _ = std::fs::write(data_dir.join("startup.log"), c);
                    }
                    return Ok(());
                }
            };

            // ── Spawn Flask ────────────────────────────────────────────────
            // Use data_dir as current_dir (always writable, never UNC).
            // Pass script_dir via env var so server.py knows where to find itself.
            log(&format!("spawning: {} {:?}", python_exe, server_path));
            log(&format!("current_dir: {:?}", data_dir));

            match Command::new(&python_exe)
                .arg(&server_path)
                .current_dir(&data_dir)          // writable, no UNC prefix
                .env("HATSUN_DATA_DIR", &data_dir)
                .env("HATSUN_SCRIPT_DIR", &script_dir)
                .stdout(std::process::Stdio::inherit())
                .stderr(std::process::Stdio::inherit())
                .spawn()
            {
                Ok(child) => {
                    log(&format!("Flask started PID={}", child.id()));
                    if let Ok(mut guard) = FLASK_PROCESS.lock() {
                        *guard = Some(child);
                    }
                }
                Err(e) => log(&format!("ERROR spawning Flask: {}", e)),
            }

            if let Ok(c) = std::fs::read_to_string(get_log_path()) {
                let _ = std::fs::write(data_dir.join("startup.log"), c);
            }
            Ok(())
        })
        .on_window_event(|app, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Ok(mut guard) = FLASK_PROCESS.lock() {
                    if let Some(mut child) = guard.take() {
                        let _ = child.kill();
                    }
                }
                app.app_handle().exit(0);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
