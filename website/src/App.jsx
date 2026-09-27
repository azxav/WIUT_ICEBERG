import { useEffect, useMemo, useRef, useState } from 'react';
import { seedData } from './seed.js';

const CLASSES = ['congestion', 'failure_to_yield', 'red_light', 'jaywalking', 'stop_line'];
const CLASS_COLOR = {
  congestion: '#54c6df', failure_to_yield: '#e7a53a', red_light: '#ef6c5b',
  jaywalking: '#a48be4', stop_line: '#d7bc71',
};
const NAV = [
  ['Team', 'team'], ['Approach', 'approach'], ['EDA', 'eda'],
  ['Results', 'results'], ['Report', 'report'], ['Links', 'links'], ['Demo', 'demo'],
];
const TEAM_MEMBERS = [
  {
    name: 'Azizbek Xasanov',
    initials: 'AX',
    role: 'Scene & labels',
    responsibility: 'Scene-map and CVAT review, event-label QA, and release documentation.',
    profiles: [
      ['LinkedIn', 'https://www.linkedin.com/in/azizbek-xasanov/'],
      ['GitHub', 'https://github.com/azxav'],
    ],
  },
  {
    name: 'Dilyorbek Muhammadjonov',
    initials: 'DI',
    role: 'Computer vision · detection',
    responsibility: 'Own detection, tracking, and Part A event rules; review speed and reproducibility.',
    profiles: [
      ['LinkedIn', 'https://www.linkedin.com/in/dilyor/'],
      ['GitHub', 'https://github.com/dilyorm'],
    ],
  },
  {
    name: 'Davlat Mahmudov',
    initials: 'DM',
    role: 'Evaluation · risk & delivery',
    responsibility: 'Own metric review, Part B risk assessment, and website/demo preparation and delivery.',
    profiles: [],
  },
];
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function timeLabel(seconds) {
  const s = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function SectionTitle({ title, description, number }) {
  return <div className="section-title">
    {number && <span className="section-number">{number}</span>}
    <div><h2>{title}</h2>{description && <p>{description}</p>}</div>
  </div>;
}

function Topbar({ active }) {
  return <header className="topbar">
    <a className="brand" href="#overview" aria-label="ICEBERG traffic event detection overview">
      <span className="brand-mark">ICEBERG</span><span className="brand-divider" />
      <span className="brand-name">Traffic Events</span>
    </a>
    <nav aria-label="Main navigation">
      {NAV.map(([name, id]) => <a className={active === id ? 'active' : ''} href={`#${id}`} key={id}>{name}</a>)}
    </nav>
  </header>;
}

function CountBars({ counts }) {
  const max = Math.max(1, ...counts.map((d) => d.count));
  return <div className="count-bars" role="img" aria-label="Reviewed event counts by class">
    {counts.map((item) => <div className="count-row" key={item.label}>
      <span className="class-name">{item.label}</span>
      <span className="bar-track"><span className="bar-fill" style={{ width: `${(item.count / max) * 100}%`, background: CLASS_COLOR[item.label] }} /></span>
      <strong>{item.count}</strong>
    </div>)}
  </div>;
}

function EventsHistogram({ events, duration }) {
  const labels = CLASSES;
  const binWidth = duration > 240 ? 15 : 10;
  const binCount = Math.max(1, Math.ceil(duration / binWidth));
  const bins = Array.from({ length: binCount }, (_, i) => ({ start: i * binWidth, counts: Object.fromEntries(labels.map((label) => [label, 0])) }));
  events.forEach(([start, , label]) => {
    const index = Math.min(binCount - 1, Math.max(0, Math.floor(start / binWidth)));
    if (bins[index]?.counts[label] != null) bins[index].counts[label] += 1;
  });
  const totals = bins.map((bin) => labels.reduce((sum, label) => sum + bin.counts[label], 0));
  const max = Math.max(1, ...totals);
  const barWidth = 100 / binCount;
  return <div className="histogram-wrap">
    <svg className="event-histogram" viewBox="0 0 100 112" preserveAspectRatio="none" role="img" aria-label={`Reviewed event counts in ${binWidth}-second bins`}>
      {[25, 50, 75, 100].map((y) => <line key={y} x1="0" x2="100" y1={y} y2={y} className="chart-grid" />)}
      {bins.map((bin, i) => {
        let y = 104;
        return <g key={bin.start}>
          {labels.map((label) => {
            const amount = bin.counts[label];
            const h = amount / max * 76;
            y -= h;
            return amount ? <rect key={label} x={i * barWidth + barWidth * .18} y={y} width={Math.max(.25, barWidth * .64)} height={h} fill={CLASS_COLOR[label]}><title>{`${timeLabel(bin.start)}–${timeLabel(bin.start + binWidth)} · ${label}: ${amount}`}</title></rect> : null;
          })}
        </g>;
      })}
    </svg>
    <div className="histogram-axis"><span>00:00</span><span>{timeLabel(duration / 2)}</span><span>{timeLabel(duration)}</span></div>
    <div className="histogram-legend">{labels.filter((label) => events.some((event) => event[2] === label)).map((label) => <span key={label}><i style={{ background: CLASS_COLOR[label] }} />{label}</span>)}</div>
  </div>;
}

function EventTimeline({ events, duration, onSelect, compact = false }) {
  const grouped = CLASSES.map((label) => ({ label, items: events.filter((event) => event[2] === label) }));
  return <div className={`event-timeline ${compact ? 'compact' : ''}`}>
    {grouped.filter((row) => !compact || row.items.length).map(({ label, items }) => <div className="timeline-row" key={label}>
      <span className="timeline-label">{label}</span>
      <div className="timeline-track" aria-label={`${label}: ${items.length} events`}>
        {items.map(([start, end], i) => {
          const left = Math.max(0, Math.min(100, (start / Math.max(duration, 1)) * 100));
          const width = Math.max(0.7, Math.min(100 - left, ((end - start) / Math.max(duration, 1)) * 100));
          return <button key={`${start}-${i}`} className="timeline-event" title={`${label}: ${timeLabel(start)}–${timeLabel(end)}`} style={{ left: `${left}%`, width: `${width}%`, backgroundColor: CLASS_COLOR[label] }} onClick={() => onSelect?.(start, end, label)} aria-label={`Seek to ${label} at ${timeLabel(start)}`} />;
        })}
      </div>
      <span className="timeline-count">{items.length}</span>
    </div>)}
    <div className="timeline-axis"><span>00:00</span><span>{timeLabel(duration / 2)}</span><span>{timeLabel(duration)}</span></div>
  </div>;
}

function MiniCurve({ values, label, color = '#53c5df', height = 110 }) {
  if (!values?.length) return <div className="empty-curve">No curve available for this sample.</div>;
  const points = values.filter((p, i) => i % Math.max(1, Math.floor(values.length / 160)) === 0 || i === values.length - 1);
  const maxT = Math.max(1, points.at(-1)?.[0] || 1);
  const d = points.map(([t, v], i) => `${i ? 'L' : 'M'} ${(t / maxT) * 100} ${(1 - Math.max(0, Math.min(1, v))) * (height - 12) + 4}`).join(' ');
  return <svg className="curve" viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" role="img" aria-label={label}>
    {[0.25, 0.5, 0.75].map((y) => <line key={y} x1="0" x2="100" y1={y * height} y2={y * height} className="chart-grid" />)}
    <path d={d} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
  </svg>;
}

function SceneMap({ data }) {
  const scene = data?.scene || {};
  const poly = (coordinates) => (coordinates || []).map(([x, y]) => `${x},${y}`).join(' ');
  return <svg className="scene-map" viewBox="0 0 1920 1080" role="img" aria-label="Mapped junction and sampled vehicle trajectories">
    <rect width="1920" height="1080" fill="#182126" />
    <defs><clipPath id="road-density-clip">{Object.values(scene.carriageways || {}).map((road, i) => <polygon key={i} points={poly(road.polygon)} />)}</clipPath></defs>
    {Object.entries(scene.carriageways || {}).map(([name, road]) => <polygon key={name} points={poly(road.polygon)} fill={name === 'near' ? '#39464c' : '#303c42'} stroke="#89989e" strokeWidth="7" />)}
    <g clipPath="url(#road-density-clip)" aria-label="Vehicle detection density">
      {(data.density || []).flatMap((row, yi) => row.map((value, xi) => value > 0.03 ? <rect key={`${xi}-${yi}`} x={xi * 60} y={yi * 60} width="60" height="60" fill="#54c6df" opacity={Math.min(.48, value * .5)} /> : null))}
    </g>
    {(scene.median?.polyline?.length > 1) && <polyline points={poly(scene.median.polyline)} fill="none" stroke="#d2c9a2" strokeWidth="24" />}
    {Object.entries(scene.lanes || {}).map(([name, lane]) => <polygon key={name} points={poly(lane.polygon)} fill="none" stroke="#879398" strokeWidth="4" strokeDasharray="20 18" opacity=".66" />)}
    {Object.entries(scene.crosswalks || {}).map(([name, shape]) => <polygon key={name} points={poly(shape.polygon)} fill="#bec4c2" stroke="#12191c" strokeWidth="4" opacity=".55" />)}
    {(data.trajectories || []).map((line, i) => <polyline key={i} points={poly(line.points)} fill="none" stroke={line.color || '#57c6df'} strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" opacity=".62" />)}
    <text x="105" y="180" className="map-label">Far carriageway · up-left</text>
    <text x="1450" y="940" className="map-label">Near approach · down-right</text>
  </svg>;
}

function useSiteData() {
  const [data, setData] = useState(seedData);
  useEffect(() => {
    fetch('/site-data.json').then((r) => r.ok ? r.json() : null).then((value) => value && setData({ ...seedData, ...value }))
      .catch(() => {});
  }, []);
  return data;
}

function sampleVideoPath(video) {
  return video?.media || `/media/${(video?.id || '').replace(/\.MP4$/i, '').toLowerCase()}_annotated.mp4`;
}

function Overview({ data, selectedVideo, setSelectedVideo, onSeek }) {
  const videoRef = useRef(null);
  const current = data.videos.find((video) => video.id === selectedVideo) || data.videos[0];
  const hasMedia = current?.mediaReady !== false;
  const seek = (start) => {
    onSeek?.(start);
    if (videoRef.current && hasMedia) videoRef.current.currentTime = start;
  };
  return <section id="overview" className="hero-section">
    <div className="hero-copy">
      <div><h1>Traffic, mapped and measured</h1><p className="hero-subtitle">Four sample clips. One fixed camera. 118 reviewed events.</p>
        <p className="hero-description">A mapped-junction pipeline for reviewing road-user movement, event timing, and a causal risk baseline.</p></div>
      <div className="pipeline" aria-label="Analysis pipeline">
        {['Scene map', 'Track', 'Rules', 'Events & risk'].map((item, i) => <div className="pipeline-step" key={item}><strong>{item}</strong><span>{['Register the junction', 'Track road users', 'Apply reviewed rules', 'Inspect time and risk'][i]}</span>{i < 3 && <b aria-hidden="true">→</b>}</div>)}
      </div>
    </div>
    <div className="overview-grid">
      <div className="media-panel">
        {hasMedia ? <video ref={videoRef} src={sampleVideoPath(current)} poster="/scene-frame.jpg" controls preload="metadata" aria-label={`${current?.id} annotated sample video`} />
          : <div className="video-placeholder"><img src="/scene-frame.jpg" alt="A frame from the mapped junction" /><p>Annotated preview is being prepared.</p></div>}
        <div className="media-footer"><label htmlFor="overview-video">Sample clip</label><select id="overview-video" value={selectedVideo} onChange={(event) => setSelectedVideo(event.target.value)}>{data.videos.map((video) => <option value={video.id} key={video.id}>{video.id}</option>)}</select></div>
      </div>
      <div className="timeline-panel">
        <div className="panel-heading"><h2>Event timeline</h2><span>Reviewed development labels</span></div>
        <EventTimeline events={current?.events || []} duration={current?.duration || 1} onSelect={(start) => seek(start)} />
        <div className="gate-note"><span className="gate-rule" />
          <p>Current model emits <b>congestion</b>, <b>failure_to_yield</b>, and <b>red_light</b> under the saved 0.7 dev-precision gate.</p>
          <div><strong>118</strong><span>reviewed events<br />across four clips</span></div>
        </div>
      </div>
    </div>
    <div className="hero-bottom"><span>Rules tuned on reviewed development labels</span><a href="#results">Explore sample results <b>↓</b></a></div>
  </section>;
}

function TeamSection() {
  return <section id="team" className="text-section section-rule">
    <SectionTitle number="01" title="Team" description="ICEBERG · Traffic event detection project." />
    <div className="team-intro">
      <div><span className="team-kicker">ASSIGNED WORKSTREAMS</span><h3>Three owners, one submission.</h3></div>
      <p>Responsibilities are split across scene and annotation review, computer vision, and evaluation and delivery.</p>
    </div>
    <div className="team-grid">{TEAM_MEMBERS.map((member, index) => <article className="team-card" key={member.name}>
      <div className="team-card-meta"><span>{String(index + 1).padStart(2, '0')} / ICEBERG</span><span className="team-initials">{member.initials}</span></div>
      <p className="team-role">{member.role}</p>
      <h3>{member.name}</h3>
      <p className="team-responsibility">{member.responsibility}</p>
      <div className="team-links">{member.profiles.length ? member.profiles.map(([label, url]) => <a href={url} key={label} target="_blank" rel="noreferrer" aria-label={`${member.name} on ${label}`}>{label}<span aria-hidden="true"> ↗</span></a>) : <span className="profile-note">Profile links not provided</span>}</div>
    </article>)}</div>
    <p className="team-note">Workstream ownership is assigned for this submission. Only profile links supplied by the team are shown.</p>
  </section>;
}

function ApproachSection({ data }) {
  return <section id="approach" className="section-block section-rule">
    <SectionTitle number="02" title="Approach" description="A fixed-camera pipeline, calibrated against the junction and reviewed event intervals." />
    <div className="approach-grid">
      <div className="approach-copy"><h3>From pixels to time-localized events</h3><p>We register the scene map to the first frame, track road users with YOLO11s and ByteTrack, and derive motion, lane, crosswalk, queue, and signal features. Temporal rules turn those features into event intervals.</p><p>The sample labels are a manually reviewed development set. They support rule review and threshold selection; they are not a held-out test set.</p><a className="text-link" href="#eda">See the scene and data <b>→</b></a></div>
      <ol className="approach-steps">
        <li><span>01</span><div><strong>Register</strong><p>Align the hand-mapped road, lanes, crossings, stop line, and signal ROI.</p></div></li>
        <li><span>02</span><div><strong>Track</strong><p>Follow vehicles and pedestrians; smooth positions and estimate movement.</p></div></li>
        <li><span>03</span><div><strong>Apply rules</strong><p>Detect congestion, yielding conflicts, and red-light running. Gate low-precision classes.</p></div></li>
        <li><span>04</span><div><strong>Estimate risk</strong><p>Part B samples frames causally and combines TTC, braking, and signal cues. Risk is experimental.</p></div></li>
      </ol>
    </div>
    <div className="class-policy"><span>Emitted classes</span>{data.emittedClasses.map((name) => <strong key={name} style={{ color: CLASS_COLOR[name] }}>{name}</strong>)}<span className="policy-divider" /><span>Held back under the 0.7 precision threshold: jaywalking, stop_line.</span></div>
  </section>;
}

function EdaSection({ data }) {
  const [videoId, setVideoId] = useState(data.videos[0]?.id || '');
  const current = data.videos.find((video) => video.id === videoId) || data.videos[0];
  const counts = useMemo(() => data.classCounts || seedData.classCounts, [data.classCounts]);
  return <section id="eda" className="section-block section-rule">
    <SectionTitle number="03" title="Exploratory data analysis" description="Reviewed events, tracked movement, and the geometry used by the rules." />
    <p className="data-note">Four clips from one fixed camera. The 118 event intervals below are development labels for rule validation.</p>
    <div className="eda-top">
      <div className="viz-block"><div className="viz-heading"><h3>Reviewed event counts</h3><span>118 total</span></div><CountBars counts={counts} /></div>
      <div className="viz-block"><div className="viz-heading"><h3>Events over time</h3><select value={videoId} onChange={(e) => setVideoId(e.target.value)} aria-label="Select clip for event-count chart">{data.videos.map((video) => <option key={video.id} value={video.id}>{video.id}</option>)}</select></div>
        <EventsHistogram events={current?.events || []} duration={current?.duration || 1} /></div>
    </div>
    <div className="eda-bottom">
      <div className="viz-block map-block"><div className="viz-heading"><div><h3>Tracked movement and detection density</h3><span>Density uses road-user detections; traces are sampled trajectories, not event locations</span></div></div><SceneMap data={data} /><div className="map-legend"><span><i />Detection density</span><span>Track observations are not event coordinates</span></div></div>
      <div className="viz-block lane-block"><div className="viz-heading"><div><h3>Lane directions and track counts</h3><span>Mapped lanes with observed track samples</span></div></div>
        <div className="flow-list"><div><span className="direction-arrow">↖</span><div><strong>Far carriageway</strong><p>Traffic moves up-left, away from camera.</p></div></div><div><span className="direction-arrow">↘</span><div><strong>Near approach</strong><p>Four through lanes and one curbside right-turn lane move down-right. The right-turn branch leads lower-left.</p></div></div></div>
        {data.laneCounts?.length > 0 && <div className="lane-counts">{data.laneCounts.map((lane) => <div key={lane.lane}><span>{lane.lane}</span><i style={{ width: `${Math.max(2, lane.count / Math.max(...data.laneCounts.map((x) => x.count)) * 100)}%` }} /><b>{lane.count.toLocaleString()}</b></div>)}</div>}
      </div>
    </div>
  </section>;
}

function ResultsSection({ data, selectedVideo, setSelectedVideo }) {
  const current = data.videos.find((video) => video.id === selectedVideo) || data.videos[0];
  const videoRef = useRef(null);
  const seek = (start) => { if (videoRef.current && current?.mediaReady !== false) videoRef.current.currentTime = start; };
  return <section id="results" className="section-block section-rule">
    <SectionTitle number="04" title="Sample results" description="Model outputs next to reviewed intervals, with the limits visible." />
    <div className="result-toolbar"><label htmlFor="result-video">Video</label><select id="result-video" value={selectedVideo} onChange={(e) => setSelectedVideo(e.target.value)}>{data.videos.map((video) => <option key={video.id} value={video.id}>{video.id}</option>)}</select>
      {current?.runtime != null && <span className="runtime">Runtime {Number(current.runtime).toFixed(1)} s / {Number(current.duration).toFixed(1)} s clip</span>}</div>
    <div className="results-grid">
      <div className="media-panel result-player">{current?.mediaReady === false ? <div className="video-placeholder"><img src="/scene-frame.jpg" alt="Mapped junction" /><p>Annotated video export not available yet.</p></div> : <video ref={videoRef} src={sampleVideoPath(current)} poster="/scene-frame.jpg" controls preload="metadata" aria-label={`${current?.id} sample result video`} />}</div>
      <div className="result-timelines"><div className="panel-heading"><h3>Reviewed labels</h3><span>{current?.events?.length || 0} intervals</span></div><EventTimeline events={current?.events || []} duration={current?.duration || 1} onSelect={seek} />
        <div className="panel-heading prediction-heading"><h3>Predicted events</h3><span>{current?.predictions?.length || 0} intervals</span></div><EventTimeline events={current?.predictions || []} duration={current?.duration || 1} onSelect={seek} compact />
      </div>
    </div>
    <div className="result-bottom">
      <div className="viz-block"><div className="viz-heading"><h3>Accident risk (experimental)</h3><span>Causal per-frame estimates</span></div><MiniCurve values={current?.risk || []} label="Experimental accident risk curve" /></div>
      <div className="viz-block score-block"><div className="viz-heading"><h3>Development-set checks</h3><span>tIoU 0.5 · class gate 0.7</span></div>
        {data.metrics?.length ? <div className="metrics-table"><div className="metric-head"><span>Class</span><span>P</span><span>R</span><span>F1</span><span>TP/FP/FN · clip</span></div>{data.metrics.map((m) => <div key={m.label}><span>{m.label}</span><span>{m.precision?.toFixed(2) ?? '—'}</span><span>{m.recall?.toFixed(2) ?? '—'}</span><span>{m.f1?.toFixed(2) ?? '—'}</span><span className="metric-counts">{(current?.errorCounts?.[m.label] || [0, 0, 0]).join('/')}</span></div>)}</div> : <p>Comparison will appear after the full sample run.</p>}
        <p className="small-note">P/R/F1 pool all four clips. TP/FP/FN are for the selected clip. Different event classes may overlap; these dev labels are not a held-out benchmark.</p>
        {data.signalAblation && <div className="ablation-block"><div className="viz-heading"><h3>Signal-color ablation</h3><span>same dev set</span></div>
          <div className="metrics-table ablation-table"><div className="metric-head"><span>Signal rule</span><span>Score A</span><span>Red P</span><span>Red FP</span></div>
            {[data.signalAblation.before, data.signalAblation.after].map((row) => <div key={row.label}><span>{row.label}</span><span>{row.score?.toFixed(3) ?? '—'}</span><span>{row.precision?.toFixed(2) ?? '—'}</span><span>{row.fp}</span></div>)}
          </div><p className="small-note">{data.signalAblation.note} All 4 red-light labels remain detected.</p>
        </div>}
      </div>
    </div>
  </section>;
}

function DemoSection() {
  const [file, setFile] = useState(null);
  const [duration, setDuration] = useState(null);
  const [status, setStatus] = useState('idle');
  const [message, setMessage] = useState('Choose an MP4 to begin.');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const inputRef = useRef(null);
  const videoRef = useRef(null);
  const objectUrlRef = useRef(null);
  useEffect(() => () => { if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current); }, []);

  const chooseFile = (next) => {
    setError(''); setResult(null); setDuration(null); setFile(null); setStatus('idle'); setUploadProgress(0);
    if (!next) return;
    if (!/\.mp4$/i.test(next.name) || next.type && next.type !== 'video/mp4') { setError('Choose an MP4 video.'); return; }
    if (next.size > 200 * 1024 * 1024) { setError('Maximum file size is 200 MB.'); return; }
    const url = URL.createObjectURL(next);
    const probe = document.createElement('video');
    probe.preload = 'metadata'; probe.src = url;
    probe.onloadedmetadata = () => {
      URL.revokeObjectURL(url);
      if (!Number.isFinite(probe.duration) || probe.duration <= 0) { setError('Could not read this video.'); return; }
      if (probe.duration > 120) { setError('Maximum video length is 2 minutes.'); return; }
      setFile(next); setDuration(probe.duration); setMessage('Ready to upload.');
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = URL.createObjectURL(next);
    };
    probe.onerror = () => { URL.revokeObjectURL(url); setError('Could not read this video.'); };
  };

  const analyze = () => {
    if (!file || !duration) return;
    setStatus('uploading'); setMessage('Uploading video…'); setError(''); setResult(null); setUploadProgress(0);
    const form = new FormData(); form.append('file', file);
    const xhr = new XMLHttpRequest(); xhr.open('POST', `${API_BASE}/api/jobs`);
    xhr.upload.onprogress = (event) => { if (event.lengthComputable) setUploadProgress(Math.round(event.loaded / event.total * 100)); };
    xhr.onerror = () => { setStatus('failed'); setError(`Cannot reach the demo API at ${API_BASE}.`); };
    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) { setStatus('failed'); setError(xhr.responseText || `Upload failed (${xhr.status}).`); return; }
      let job;
      try { job = JSON.parse(xhr.responseText); } catch { setStatus('failed'); setError('The API returned an invalid response.'); return; }
      poll(job.job_id);
    };
    xhr.send(form);
  };

  const poll = async (jobId) => {
    setStatus('analyzing'); setMessage('Video uploaded. Analysis is running.');
    try {
      while (true) {
        const response = await fetch(`${API_BASE}/api/jobs/${encodeURIComponent(jobId)}`);
        if (!response.ok) throw new Error(`Status request failed (${response.status}).`);
        const job = await response.json(); setMessage(job.message || 'Analysis is running.');
        if (job.status === 'completed') { setResult(job.result); setStatus('completed'); setMessage('Results are ready.'); return; }
        if (job.status === 'failed') throw new Error(job.error || 'Analysis failed.');
        await new Promise((resolve) => setTimeout(resolve, 1500));
      }
    } catch (e) { setStatus('failed'); setError(e.message || 'Could not retrieve analysis.'); }
  };

  const demoSeek = (start) => { if (videoRef.current) videoRef.current.currentTime = start; };
  return <section id="demo" className="section-block demo-section section-rule">
    <SectionTitle number="07" title="Try the event detector" description="Upload a short clip from a camera view matching this mapped junction." />
    <div className="demo-workspace">
      <div className="demo-controls">
        <div className="upload-box" role="button" tabIndex="0" onClick={() => inputRef.current?.click()} onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); chooseFile(e.dataTransfer.files?.[0]); }}>
          <input ref={inputRef} type="file" accept="video/mp4,.mp4" hidden onChange={(e) => chooseFile(e.target.files?.[0])} />
          <span className="upload-icon" aria-hidden="true">↑</span><strong>{file ? file.name : 'Choose an MP4 or drop it here'}</strong>
          <span>MP4 only · up to 2 minutes · 200 MB</span><span>Video is scaled to 1280 × 720 for demo processing.</span>
        </div>
        {file && <div className="file-summary"><span>{file.name}</span><span>{(file.size / (1024 * 1024)).toFixed(1)} MB · {timeLabel(duration)}</span><button onClick={() => chooseFile(null)} aria-label="Remove selected video">Remove</button></div>}
        <button className="run-button" disabled={!file || !duration || ['uploading', 'analyzing'].includes(status)} onClick={analyze}>{status === 'uploading' ? `Uploading ${uploadProgress}%` : status === 'analyzing' ? 'Analyzing…' : 'Run analysis'}</button>
        <div className={`status-message ${status}`} aria-live="polite">{error || message}</div>
        <ol className="demo-steps"><li className={status !== 'idle' ? 'done' : 'current'}>Upload</li><li className={status === 'analyzing' ? 'current' : status === 'completed' ? 'done' : ''}>Analyze</li><li className={status === 'completed' ? 'current' : ''}>Results</li></ol>
        <div className="demo-note"><strong>Demo limits</strong><p>Best results with a similar camera angle and junction. Uploaded videos stay in a temporary session and are removed after processing.</p><p>Risk output is an experimental baseline.</p></div>
      </div>
      <div className="demo-output">
        {result && objectUrlRef.current ? <video ref={videoRef} src={objectUrlRef.current} controls preload="metadata" aria-label="Uploaded demo video" />
          : <div className="demo-empty"><img src="/scene-frame.jpg" alt="Mapped junction reference view" /><p>{file ? 'Video ready for analysis' : 'Video preview and results appear here'}</p></div>}
        {result ? <><div className="panel-heading"><h3>Detected events</h3><span>{result.events?.length || 0} intervals</span></div><EventTimeline events={result.events || []} duration={duration || 1} onSelect={demoSeek} />
          <div className="viz-heading risk-heading"><h3>Accident risk · experimental</h3><span>Only past frames used</span></div><MiniCurve values={result.risk || []} label="Uploaded video experimental risk curve" /></>
          : <div className="output-empty"><span>Upload → Analyze → Results</span><p>After processing, event intervals and the risk curve will be shown here.</p></div>}
      </div>
    </div>
  </section>;
}

function ReportSection({ data }) {
  return <section id="report" className="section-block section-rule">
    <SectionTitle number="05" title="Technical report" description="What was built, where it works, and what remains uncertain." />
    <div className="report-grid">
      <article><h3>What works</h3><p>Scene registration, vehicle and pedestrian tracking, signal-state sampling, and temporal rules produce events in the official format. Congestion, failure-to-yield, and red-light rules clear the selected 0.7 development-precision gate.</p></article>
      <article><h3>What is withheld</h3><p>Jaywalking and stop-line stay in the official class list, but the current scored run emits no predictions for them. Their zero precision is not a candidate-rule estimate; the dev labels include 13 jaywalking and 2 stop-line intervals. Other official classes remain disabled until a reliable rule is validated.</p></article>
      <article><h3>Risk baseline</h3><p>Part B uses causal sampled tracks, time-to-collision, hard braking, and signal cues. The development set has no accident ground-truth intervals, so this risk curve is exploratory rather than calibrated.</p></article>
    </div>
    <p className="data-note">{data.runtimeNote}</p><div className="runtime-table"><div><span>Sample</span><span>Duration</span><span>Last full run</span><span>Budget use</span></div>{data.videos.map((video) => <div key={video.id}><span>{video.id}</span><span>{Number(video.duration || 0).toFixed(1)} s</span><span>{video.runtime == null ? 'Pending full run' : `${Number(video.runtime).toFixed(1)} s`}</span><span>{video.runtime == null ? '—' : `${(100 * video.runtime / Math.max(1, video.duration * 3)).toFixed(0)}% of 3× budget`}</span></div>)}</div>
  </section>;
}

function LinksSection() {
  return <section id="links" className="section-block section-rule links-section">
    <SectionTitle number="06" title="Project links" description="Code and reproducibility files for review." />
    <div className="link-list"><a href="/predictions_samples.json" target="_blank" rel="noreferrer"><span>Sample predictions</span><b>Open JSON ↗</b></a><a href="/weights/yolo11s.pt" download><span>YOLO11s weights</span><b>Download ↗</b></a><a href="/README.md" target="_blank" rel="noreferrer"><span>Method and run instructions</span><b>Read README ↗</b></a><a href="#team"><span>ICEBERG team profiles</span><b>View team ↗</b></a><div><span>Public project repository</span><b>Not created yet</b></div></div>
    <p className="license-note">Model and repository licensing: AGPL-3.0. Sample-video rights remain with their source owners.</p>
  </section>;
}

export default function App() {
  const data = useSiteData();
  const [selectedVideo, setSelectedVideo] = useState(data.videos[0]?.id || 'C3896.MP4');
  const [active, setActive] = useState('overview');
  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => { if (entry.isIntersecting) setActive(entry.target.id); });
    }, { rootMargin: '-25% 0px -65% 0px' });
    ['overview', ...NAV.map(([, id]) => id)].forEach((id) => { const el = document.getElementById(id); if (el) observer.observe(el); });
    return () => observer.disconnect();
  }, []);
  useEffect(() => { if (!data.videos.some((video) => video.id === selectedVideo) && data.videos[0]) setSelectedVideo(data.videos[0].id); }, [data, selectedVideo]);
  return <>
    <Topbar active={active} />
    <main className="page-shell">
      <Overview data={data} selectedVideo={selectedVideo} setSelectedVideo={setSelectedVideo} />
      <TeamSection />
      <ApproachSection data={data} />
      <EdaSection data={data} />
      <ResultsSection data={data} selectedVideo={selectedVideo} setSelectedVideo={setSelectedVideo} />
      <ReportSection data={data} />
      <LinksSection />
      <DemoSection />
    </main>
    <footer className="footer"><span>ICEBERG · Traffic Event Detection</span><a href="#overview">Back to top ↑</a></footer>
  </>;
}
