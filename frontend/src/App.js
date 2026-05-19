import React, { useEffect, useState, useRef, useCallback } from 'react';
import 'leaflet/dist/leaflet.css';
import { MapContainer, TileLayer, Popup, Marker, Polyline, CircleMarker, useMap } from 'react-leaflet';
import L from 'leaflet';
import './App.css';

import iconImg from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

L.Marker.prototype.options.icon = L.icon({
  iconUrl: iconImg,
  shadowUrl: iconShadow,
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});
const API = 'http://127.0.0.1:5050';

const ROUTE_COLORS = [
  '#e6194b', '#3cb44b', '#4363d8', '#f58231',
  '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
];

/* ── Auto-fit map bounds ─────────────────────────────────────────────────── */
function FitBounds({ positions }) {
  const map = useMap();
  useEffect(() => {
    if (positions.length > 0) {
      map.fitBounds(positions, { padding: [30, 30] });
    }
  }, [map, positions]);
  return null;
}

/* ── Main App ────────────────────────────────────────────────────────────── */
export default function App() {
  const [lat, setLat] = useState('');
  const [lon, setLon] = useState('');
  const [milkQty, setMilkQty] = useState('');
  const [runType, setRunType] = useState('insertion');
  const [logs, setLogs] = useState([]);
  const [status, setStatus] = useState('idle');
  const [phase, setPhase] = useState('');
  const [routes, setRoutes] = useState([]);
  const [cc, setCc] = useState(null);
  const [fullResults, setFullResults] = useState(null);
  const [insertionResults, setInsertionResults] = useState(null);
  const [newHmb, setNewHmb] = useState(null);
  const [selectedInsertionIdx, setSelectedInsertionIdx] = useState(0);
  const [distanceMode, setDistanceMode] = useState('haversine');
  const [uploadedFileName, setUploadedFileName] = useState(null);
  // 'waiting' | 'ready' | 'failed'
  const [serverState, setServerState] = useState('waiting');
  const logRef = useRef(null);
  const eventSourceRef = useRef(null);

  /* scroll log to bottom */
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  /* fetch initial routes once server is ready */
  const fetchRoutes = () => {
    fetch(`${API}/api/routes?t=${Date.now()}`, { cache: 'no-store' })
      .then(async r => {
        if (!r.ok) throw new Error(`Server returned ${r.status}`);
        return r.json();
      })
      .then(d => { setRoutes(d.routes || []); setCc(d.cc); })
      .catch((err) => console.error('Error fetching routes:', err));
  };

  /* Poll Flask /api/health until it responds (max 20 s = 40 × 500 ms) */
  useEffect(() => {
    let attempts = 0;
    const MAX = 40;
    const poll = setInterval(async () => {
      attempts++;
      try {
        const r = await fetch(`${API}/api/health`, { cache: 'no-store' });
        if (r.ok) {
          clearInterval(poll);
          setServerState('ready');
          // Only load routes if the user has already uploaded a file.
          // This prevents the bundled static CSV data from appearing
          // on the map before any file has been provided.
          try {
            const dr = await fetch(`${API}/api/has-data`, { cache: 'no-store' });
            const dj = await dr.json();
            if (dj.has_uploaded) fetchRoutes();
          } catch (_) { /* has-data failing is non-fatal */ }
        }
      } catch (_) {
        // not ready yet
      }
      if (attempts >= MAX) {
        clearInterval(poll);
        setServerState('failed');
      }
    }, 500);
    return () => clearInterval(poll);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setStatus('running');
    setPhase('Uploading Data...');
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API}/api/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Upload failed (${res.status}): ${errText}`);
      }
      fetchRoutes();
      setStatus('idle');
      setPhase('');
      setUploadedFileName(file.name);
      setLogs(prev => [...prev, `Successfully uploaded and parsed: ${file.name}`]);
    } catch (err) {
      console.error(err);
      setStatus('error');
      setLogs(prev => [...prev, `Error uploading file: ${err.message}`]);
    }
  };

  /* SSE listener */
  const connectSSE = useCallback(() => {
    if (eventSourceRef.current) eventSourceRef.current.close();

    const es = new EventSource(`${API}/api/events`);
    eventSourceRef.current = es;

    es.onmessage = (e) => {
      const msg = JSON.parse(e.data);

      switch (msg.type) {
        case 'log':
          setLogs(prev => [...prev, msg.message]);
          break;
        case 'status':
          setStatus(msg.status);
          break;
        case 'phase':
          setPhase(`${msg.phase} — ${msg.status}`);
          break;
        case 'insertion_results':
          setInsertionResults(msg.data);
          if (msg.new_hmb) setNewHmb(msg.new_hmb);
          if (msg.cc) setCc(msg.cc);
          // Also set the routes map to display all routes (which includes the cc changes)
          fetchRoutes();
          break;
        case 'full_results':
          setFullResults(msg.data);
          if (msg.data?.new_hmb) setNewHmb(msg.data.new_hmb);
          if (msg.data?.cc) setCc(msg.data.cc);
          // Update map routes from full optimization
          if (msg.data?.routes) {
            setRoutes(msg.data.routes.map(r => ({
              route_code: r.route_code,
              route_name: r.route_name,
              hmbs: r.hmbs || [],
            })));
          }
          break;
        case 'error':
          setStatus('error');
          setLogs(prev => [...prev, `ERROR: ${msg.message}`]);
          break;
        default:
          break;
      }
    };

    es.onerror = () => { /* reconnects automatically */ };
    return es;
  }, []);

  /* kick off an optimization run */
  const handleRun = async () => {
    setLogs([]);
    setStatus('running');
    setPhase('');
    setInsertionResults(null);
    setFullResults(null);
    setNewHmb(null);
    setSelectedInsertionIdx(0);

    connectSSE();

    const body = { type: runType, mode: distanceMode };

    if (lat.trim() && lon.trim()) {
      body.lat = parseFloat(lat);
      body.lon = parseFloat(lon);
    }
    if (milkQty.trim()) {
      body.milk_qty = parseFloat(milkQty);
    }

    try {
      const res = await fetch(`${API}/api/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        throw new Error(`Run optimization failed: ${res.status}`);
      }
    } catch (err) {
      console.error(err);
      setStatus('error');
      setPhase('Error');
      setLogs(prev => [...prev, `Error running optimization: ${err.message}`]);
    }
  };

  /* ── derive map data ───────────────────────────────────────────────────── */
  const displayRoutes = fullResults?.routes ?? routes;
  const getRouteColor = (routeCode) => {
    const idx = displayRoutes.findIndex(r => r.route_code === routeCode);
    return idx !== -1 ? ROUTE_COLORS[idx % ROUTE_COLORS.length] : '#888888';
  };

  const allPositions = [];
  if (cc) allPositions.push([cc.lat, cc.lng]);
  displayRoutes.forEach(r => r.hmbs?.forEach(h => allPositions.push([h.lat, h.lng])));
  if (newHmb) allPositions.push([newHmb.lat, newHmb.lng]);

  const mapCenter = cc ? [cc.lat, cc.lng] : [12.3, 78.5];

  /* ── Server startup overlay ─────────────────────────────────── */
  if (serverState === 'waiting') {
    return (
      <div style={{
        display: 'flex', height: '100vh', alignItems: 'center', justifyContent: 'center',
        background: '#1e1e2e', color: '#cdd6f4', flexDirection: 'column', gap: 16
      }}>
        <div style={{ fontSize: 32 }}>⏳</div>
        <div style={{ fontSize: 18, fontWeight: 600, color: '#89b4fa' }}>Starting backend server…</div>
        <div style={{ fontSize: 13, color: '#6c7086' }}>This takes a few seconds on first launch.</div>
      </div>
    );
  }

  if (serverState === 'failed') {
    const isWin = navigator.userAgent.includes('Windows');
    const logPath = isWin
      ? '%TEMP%\\hatsun_startup.log'
      : '/tmp/hatsun_startup.log';
    const dataLog = isWin
      ? '%LOCALAPPDATA%\\HatsunVRP\\startup.log'
      : '~/Library/Application Support/HatsunVRP/startup.log';
    return (
      <div style={{
        display: 'flex', height: '100vh', alignItems: 'center', justifyContent: 'center',
        background: '#1e1e2e', color: '#cdd6f4', flexDirection: 'column', gap: 12, padding: 32
      }}>
        <div style={{ fontSize: 32 }}>❌</div>
        <div style={{ fontSize: 18, fontWeight: 600, color: '#f38ba8' }}>Backend server failed to start</div>
        <div style={{ fontSize: 13, color: '#a6adc8', textAlign: 'center', maxWidth: 520, lineHeight: 1.7 }}>
          The Python/Flask server could not start. Common causes:
          <ul style={{ textAlign: 'left', marginTop: 8 }}>
            <li><b>Python not found</b> — install Python 3.9+ and tick "Add to PATH"</li>
            <li><b>Missing packages</b> — open a terminal and run:<br />
              <code style={{ color: '#a6e3a1' }}>pip install flask flask-cors pandas openpyxl</code></li>
            <li><b>Antivirus</b> blocked python.exe</li>
            <li><b>Scripts not bundled</b> — rebuild the app</li>
          </ul>
          <b>Debug log (primary):</b><br />
          <code style={{ color: '#f9e2af', fontSize: 12 }}>{logPath}</code><br /><br />
          <b>Debug log (copy):</b><br />
          <code style={{ color: '#f9e2af', fontSize: 12 }}>{dataLog}</code>
        </div>
        <button onClick={() => window.location.reload()}
          style={{
            marginTop: 8, padding: '8px 20px', background: '#89b4fa', color: '#1e1e2e',
            border: 'none', borderRadius: 6, fontWeight: 600, cursor: 'pointer', fontSize: 14
          }}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'system-ui, sans-serif' }}>

      {/* ── Sidebar ─────────────────────────────────────────────────────── */}

      <div style={{
        width: 400, minWidth: 360, background: '#1e1e2e', color: '#cdd6f4',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }}>
        <div style={{ padding: '16px 16px 8px', borderBottom: '1px solid #313244', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ margin: 0, color: '#89b4fa' }}>Route Optimizer</h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 11, color: '#6c7086' }}>Mode:</span>
            <select value={distanceMode} onChange={e => setDistanceMode(e.target.value)}
              style={{ background: '#313244', color: '#cdd6f4', border: '1px solid #45475a', borderRadius: 4, fontSize: 11, padding: '3px 6px', cursor: 'pointer' }}>
              <option value="haversine">Haversine</option>
              <option value="osrm">OSRM</option>
            </select>
          </div>
        </div>

        {/* controls */}
        <div style={{ padding: 16, borderBottom: '1px solid #313244' }}>

          {/* Data Upload UI */}
          <div style={{ marginBottom: 16, padding: 12, border: '1px dashed #585b70', borderRadius: 4, background: '#181825' }}>
            <label style={{ fontSize: 13, display: 'block', marginBottom: 8, fontWeight: 'bold' }}>Data Source</label>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <label htmlFor="csv-upload" style={{ cursor: 'pointer', background: '#313244', padding: '6px 12px', borderRadius: 4, fontSize: 13, border: '1px solid #45475a', flex: 1, textAlign: 'center' }}>
                Upload CSV / Excel
              </label>
              <input id="csv-upload" type="file" accept=".csv,.xlsx,.xls" onChange={handleFileUpload} style={{ display: 'none' }} />
            </div>
            {uploadedFileName && (
              <div style={{ marginTop: 6, fontSize: 11, color: '#a6e3a1', display: 'flex', alignItems: 'center', gap: 4 }}>
                <span>📄</span>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{uploadedFileName}</span>
              </div>
            )}

            <details style={{ marginTop: 8, fontSize: 11, color: '#a6adc8' }}>
              <summary style={{ cursor: 'pointer', userSelect: 'none' }}>Expected Data Format</summary>
              <div style={{ marginTop: 6, lineHeight: 1.5 }}>
                Ensure your file includes these exact columns:<br />
                <code>CC Code, CC Name, CC Lat & Lon, Route Code, Route Name, Route Capacity, Route Milk Qty, Per Day KM, UTI Percent, Transporter Name, Vehicle Type, HMB Sequence, HMB SAP Code, HMB Name, HMB Lat & Lon, HMB Distance KM</code>
              </div>
            </details>
          </div>

          <label style={{ fontSize: 13 }}>Optimization Type</label>
          <select value={runType} onChange={e => setRunType(e.target.value)}
            style={inputStyle}>
            <option value="insertion">Insert Route (2-Opt)</option>
            <option value="full">Full Route Optimization (GA)</option>
          </select>

          <label style={{ fontSize: 13 }}>
            New HMB Latitude {runType === 'full' && <span style={{ color: '#6c7086' }}>(optional)</span>}
          </label>
          <input value={lat} onChange={e => setLat(e.target.value)}
            placeholder="e.g. 12.35" style={inputStyle} />

          <label style={{ fontSize: 13 }}>
            New HMB Longitude {runType === 'full' && <span style={{ color: '#6c7086' }}>(optional)</span>}
          </label>
          <input value={lon} onChange={e => setLon(e.target.value)}
            placeholder="e.g. 78.55" style={inputStyle} />

          <label style={{ fontSize: 13 }}>Expected Milk (litres/day)</label>
          <input value={milkQty} onChange={e => setMilkQty(e.target.value)}
            placeholder="e.g. 100" style={inputStyle} />

          <button onClick={handleRun} disabled={status === 'running'}
            style={{
              width: '100%', padding: '10px', marginTop: 8,
              background: status === 'running' ? '#585b70' : '#89b4fa',
              color: '#1e1e2e', border: 'none', borderRadius: 6,
              fontWeight: 600, fontSize: 14, cursor: status === 'running' ? 'wait' : 'pointer',
            }}>
            {status === 'running' ? `Running… ${phase}` : 'Run Optimization'}
          </button>
        </div>

        {/* ── Results Panel ─────────────────────────────────────────────── */}
        <div style={{ flex: 1, overflow: 'auto' }}>

          {/* Full optimization results */}
          {fullResults && (
            <div style={{ padding: '12px 16px', borderBottom: '1px solid #313244' }}>
              <div style={{ color: '#a6e3a1', fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
                Full Optimization Complete
              </div>
              <div style={statRow}>
                <span>Original</span><b>{fullResults.original_km} KM</b>
              </div>
              <div style={statRow}>
                <span>Optimized</span><b style={{ color: '#a6e3a1' }}>{fullResults.optimized_km} KM</b>
              </div>
              <div style={statRow}>
                <span>Saved</span>
                <b style={{ color: '#f9e2af' }}>{fullResults.saving_km} KM ({fullResults.saving_pct}%)</b>
              </div>

              {/* Per-route details */}
              <div style={{ marginTop: 12 }}>
                {fullResults.routes?.filter(r => r.hmb_count > 0).map((r, i) => (
                  <div key={r.route_code} style={{
                    padding: '8px', marginTop: 6, borderRadius: 6,
                    background: '#313244', fontSize: 12,
                    borderLeft: `4px solid ${getRouteColor(r.route_code)}`,
                  }}>
                    <div style={{ fontWeight: 600, marginBottom: 4 }}>
                      {r.route_code} ({r.route_name}) — {r.hmb_count} HMBs
                    </div>
                    <div>{r.distance_km} KM ({r.diff_km >= 0 ? '+' : ''}{r.diff_km}) | {r.est_time_h}h
                      {!r.time_ok && <span style={{ color: '#f38ba8' }}> ⚠ OVER TIME</span>}
                    </div>
                    <div style={{ color: '#a6adc8', marginTop: 4 }}>
                      CC → {r.sequence?.join(' → ')} → CC
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Insertion results */}
          {insertionResults && insertionResults.length > 0 && (
            <div style={{ padding: '12px 16px', borderBottom: '1px solid #313244' }}>
              <div style={{ color: '#fab387', fontWeight: 600, marginBottom: 8, fontSize: 14 }}>
                Insertion Results (Ranked)
              </div>

              {insertionResults.map((r, i) => {
                const isSelected = i === selectedInsertionIdx;
                const routeColor = getRouteColor(r.route_code);
                return (
                  <div key={r.route_code}
                    onClick={() => setSelectedInsertionIdx(i)}
                    style={{
                      padding: '8px', marginTop: 6, borderRadius: 6,
                      background: isSelected ? '#2a3a2a' : '#313244', fontSize: 12,
                      borderLeft: `4px solid ${routeColor}`,
                      cursor: 'pointer',
                      opacity: isSelected ? 1 : 0.6,
                    }}>
                    <div style={{ fontWeight: 600, marginBottom: 2 }}>
                      <span style={{ color: routeColor, marginRight: 6 }}>⬤</span>
                      {i === 0 ? '#1 ' : `#${i + 1} `}
                      {r.route_code} ({r.route_name})
                      {!r.feasible && <span style={{ color: '#f38ba8' }}> {r.reason}</span>}
                    </div>
                    <div>
                      +{r.extra_km} KM | Post-2opt: {r.post_2opt_km} KM | Time: {r.est_time_h}h | Score: {r.score}
                    </div>
                    <div style={{ color: '#a6adc8', marginTop: 4 }}>
                      Insert: {r.prev_stop} → <b style={{ color: '#f9e2af' }}>NEW</b> → {r.next_stop}
                    </div>
                    {r.sequence && (
                      <div style={{ color: '#a6adc8', marginTop: 2 }}>
                        CC → {r.sequence.join(' → ')} → CC
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* live log */}
          <div ref={logRef} style={{
            padding: '8px 12px',
            fontFamily: '"SF Mono", "Fira Code", monospace', fontSize: 11,
            lineHeight: 1.6, background: '#181825', minHeight: 100,
          }}>
            {logs.length === 0 && <span style={{ color: '#6c7086' }}>Logs will appear here…</span>}
            {logs.map((l, i) => (
              <div key={i} style={{ color: l.includes('ERROR') ? '#f38ba8' : '#a6adc8' }}>{l}</div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Map ─────────────────────────────────────────────────────────── */}
      <div style={{ flex: 1 }}>
        <MapContainer center={mapCenter} zoom={10} scrollWheelZoom style={{ height: '100%', width: '100%' }}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          {allPositions.length > 1 && <FitBounds positions={allPositions} />}

          {/* CC marker */}
          {cc && (
            <Marker position={[cc.lat, cc.lng]}>
              <Popup><b>CC (Uthangarai Chilling Center)</b></Popup>
            </Marker>
          )}

          {/* New HMB marker */}
          {newHmb && (
            <CircleMarker center={[newHmb.lat, newHmb.lng]} radius={10}
              pathOptions={{ color: '#f38ba8', fillColor: '#f38ba8', fillOpacity: 0.9 }}>
              <Popup><b>NEW HMB</b></Popup>
            </CircleMarker>
          )}

          {/* Route polylines + HMB markers */}
          {displayRoutes.map((route, idx) => {
            if (!route.hmbs || route.hmbs.length === 0) return null;
            const color = ROUTE_COLORS[idx % ROUTE_COLORS.length];

            // Build polyline points — if this route is the selected insertion route,
            // insert the new HMB at the correct sequence position
            const selResult = insertionResults && insertionResults[selectedInsertionIdx];
            const isSelectedRoute = selResult && route.route_code === selResult.route_code;
            const points = [];
            if (cc) points.push([cc.lat, cc.lng]);
            if (isSelectedRoute && selResult?.sequence && newHmb) {
              const hmbByName = {};
              route.hmbs.forEach(h => { hmbByName[h.name] = h; });
              selResult.sequence.forEach(name => {
                if (name === 'NEW HMB') {
                  points.push([newHmb.lat, newHmb.lng]);
                } else if (hmbByName[name]) {
                  points.push([hmbByName[name].lat, hmbByName[name].lng]);
                }
              });
            } else {
              route.hmbs.forEach(h => points.push([h.lat, h.lng]));
            }
            if (cc) points.push([cc.lat, cc.lng]);

            return (
              <React.Fragment key={route.route_code || idx}>
                <Polyline positions={points}
                  pathOptions={{ color, weight: 3, opacity: 0.8 }} />
                {route.hmbs.map((h, hi) => (
                  <CircleMarker key={hi} center={[h.lat, h.lng]} radius={5}
                    pathOptions={{ color, fillColor: color, fillOpacity: 0.8 }}>
                    <Popup>
                      <b>{h.name}</b><br />
                      Route: {route.route_code} ({route.route_name})
                      {h.from_route && <><br /><i>Moved from {h.from_route}</i></>}
                    </Popup>
                  </CircleMarker>
                ))}
              </React.Fragment>
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
}

const inputStyle = {
  width: '100%', padding: '8px', marginBottom: 8, marginTop: 2,
  background: '#313244', color: '#cdd6f4', border: '1px solid #45475a',
  borderRadius: 4, fontSize: 13, boxSizing: 'border-box',
};

const statRow = {
  display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 2,
};