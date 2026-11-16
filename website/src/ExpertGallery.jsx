import React, { useEffect, useMemo, useRef, useState } from 'react';
import catalog from '../public/data/expert-catalog.json';

const matches = (task, group, query) => (group === 'all' || task.group === group) &&
  `${task.label} ${task.detail}`.toLowerCase().includes(query.trim().toLowerCase());

function Metric({ label, value }) {
  if (!value) return <div><span>{label}</span><b>No matched metric available</b></div>;
  const score = Number.isFinite(value.rate) && Number.isInteger(value.successes)
    ? `${value.successes} / ${value.n}`
    : Number.isFinite(value.body_error_m) ? `${(value.body_error_m * 100).toFixed(2)} cm`
    : 'See recorded evaluation';
  return <div><span>{label}</span><b>{score}</b></div>;
}

export default function ExpertGallery({ onOpenFilm, Video }) {
  const [selectedId, setSelectedId] = useState('husky_skateboard');
  const [group, setGroup] = useState('all');
  const [query, setQuery] = useState('');
  const [enabled, setEnabled] = useState(true);
  const [afterEnd, setAfterEnd] = useState(false);
  const tabsRef = useRef(null);
  const [scrollEdges, setScrollEdges] = useState({ left: false, right: false });
  const visible = useMemo(() => catalog.tasks.filter(task => matches(task, group, query)), [group, query]);
  const task = catalog.tasks.find(item => item.id === selectedId) || catalog.tasks[0];
  const record = enabled ? task.on : task.off;
  const modeLabel = enabled ? 'Expert on' : 'Expert off';
  const visibleKey = visible.map(item => item.id).join('|');
  useEffect(() => {
    const tabs = tabsRef.current;
    const update = () => setScrollEdges({ left: tabs.scrollLeft > 2,
      right: tabs.scrollLeft + tabs.clientWidth < tabs.scrollWidth - 2 });
    const observer = new ResizeObserver(update);
    observer.observe(tabs); tabs.addEventListener('scroll', update, { passive: true }); update();
    return () => { observer.disconnect(); tabs.removeEventListener('scroll', update); };
  }, [visibleKey]);
  useEffect(() => {
    const tabs = tabsRef.current;
    const selected = tabs.querySelector('[aria-selected="true"]');
    if (!selected) return;
    // Scroll only the skill strip; selecting a task must not move the page.
    const left = selected.offsetLeft;
    if (left < tabs.scrollLeft || left + selected.offsetWidth > tabs.scrollLeft + tabs.clientWidth)
      tabs.scrollLeft = Math.max(0, left - (tabs.clientWidth - selected.offsetWidth) / 2);
  }, [selectedId, visibleKey]);
  function scrollTabs(direction) {
    const tabs = tabsRef.current;
    tabs.scrollBy({ left: direction * tabs.clientWidth * .8,
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth' });
  }
  function select(id) { setSelectedId(id); setAfterEnd(false); }
  function filter(nextGroup, nextQuery) {
    setGroup(nextGroup); setQuery(nextQuery);
    const next = catalog.tasks.filter(item => matches(item, nextGroup, nextQuery));
    if (next.length && !next.some(item => item.id === selectedId)) select(next[0].id);
  }
  function navigate(event, index) {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? visible.length - 1
      : (index + (event.key === 'ArrowRight' ? 1 : -1) + visible.length) % visible.length;
    select(visible[next].id);
    document.getElementById(`expert-tab-${visible[next].id}`)?.focus({ preventScroll: true });
  }
  function expand(full = false) {
    if (!record) return;
    onOpenFilm({ id: full ? record.fullId : record.id,
      title: `${task.label} · ${modeLabel}`,
      note: 'Actual policy recording at original speed, including the observed task outcome.',
      neutralRecording: true });
  }
  return <section className="skills section" id="skills">
    <div className="section-heading reveal"><p className="eyebrow">02 / SKILL LIBRARY</p>
      <div className="heading-row"><h2>Different skills.<br/><span>One policy.</span></h2>
        <p className="side-copy">{catalog.tasks.length} skills · Original-speed recordings<br/>One switch. Compare expert on and off.</p></div>
    </div>
    <div className="comparison-intro" id="expert-comparison">
      <p>Choose a skill and toggle its expert to compare the same shared policy. Recordings play at original speed and hold the last frame.</p>
    </div>
    <div className="catalog-toolbar">
      <div className="catalog-filters" aria-label="Filter by skill category">
        {[{ id: 'all', label: 'All skills' }, ...catalog.groups].map(item => <button
          key={item.id} type="button" aria-pressed={group === item.id}
          onClick={() => filter(item.id, query)}>{item.label}<span>{catalog.tasks.filter(t => item.id === 'all' || t.group === item.id).length}</span></button>)}
      </div>
      <label className="catalog-search"><span>Search skills</span><input type="search" value={query}
        placeholder="Try stairs, tennis, or climbing" onChange={e => filter(group, e.target.value)}/></label>
    </div>
    <div className="catalog-strip">
    <button className="catalog-scroll-arrow" type="button" aria-label="Previous skills" disabled={!scrollEdges.left} onClick={() => scrollTabs(-1)}>‹</button>
    <div ref={tabsRef} className="catalog-skill-tabs" role="tablist" aria-label="Browse and select a skill" aria-orientation="horizontal">
      {visible.map((item, index) => <button key={item.id} type="button" role="tab"
        id={`expert-tab-${item.id}`} aria-selected={task.id === item.id}
        aria-controls="skill-panel" tabIndex={task.id === item.id ? 0 : -1}
        onClick={() => select(item.id)} onKeyDown={e => navigate(e, index)}>
        <span>{item.label}</span><small>{item.detail}</small>
      </button>)}
      {!visible.length && <p className="catalog-empty">No matching skills. Try another search.</p>}
    </div>
    <button className="catalog-scroll-arrow" type="button" aria-label="Next skills" disabled={!scrollEdges.right} onClick={() => scrollTabs(1)}>›</button>
    </div>
    <div className="expert-switch-row gallery-expert-controls"><div>
      <span className="route-label">{task.label} / RECORDED COMPARISON</span>
      <strong aria-live="polite">{modeLabel}</strong></div>
      <button id="skill-expert-switch" type="button" className={`expert-toggle ${enabled ? 'enabled' : ''}`}
        role="switch" aria-checked={enabled} aria-label={`Enable ${task.label} expert`} aria-controls="skill-panel"
        onClick={() => { setEnabled(value => !value); setAfterEnd(false); }}>
        <span className="expert-toggle-track" aria-hidden="true"><i/></span><span>{enabled ? 'ON' : 'OFF'}</span>
      </button>
    </div>
    <div key={`${task.id}-${enabled ? 'on' : 'off'}`} className="skill-stage" id="skill-panel" role="tabpanel"
      aria-labelledby={visible.some(item => item.id === task.id) ? `expert-tab-${task.id}` : undefined}
      aria-label={visible.some(item => item.id === task.id) ? undefined : task.label} tabIndex="0">
      {record ? <><Video key={`${task.id}-${record.id}`} id={record.id} loop={false}
        title={`${task.label} · ${modeLabel}`}
        onTimeUpdate={time => setAfterEnd(!enabled && record.continuation > 0 && time >= record.nativeEnd)}/>
        {!enabled && afterEnd && <span className="continuation-badge">Post-evaluation continuation · No reset</span>}
        <button className="clip-expand" aria-label={`Expand ${task.label} recording`} onClick={() => expand(false)}><span aria-hidden="true">⛶</span></button>
        <div className="stage-label"><span className={enabled ? 'live-dot' : ''}/>{task.label} · {modeLabel}</div>
        <span className="stage-spec">{record.duration.toFixed(1)} s · Original speed</span>
      </> : <div className="catalog-recording-pending"><span>{task.label} · {modeLabel}</span>
        <h3>Recording not yet available</h3><p>The actual recording will appear here once archived.</p></div>}
    </div>
    <div className="catalog-caption"><p>{task.detail}. Actual task outcome retained. Playback holds the final frame.</p>
      {record && <button className="text-link" onClick={() => expand(true)}>Watch full recording <span aria-hidden="true">▷</span></button>}
    </div>
    <details className="catalog-evidence"><summary>View existing evaluation</summary>
      <p>{task.metricLabel}</p><div className="catalog-metrics"><Metric label="Expert on" value={task.metrics.on}/><Metric label="Expert off" value={task.metrics.off}/></div>
      <small>These batch evaluations are separate from the individual demonstration. Tasks use different metrics and should not be compared directly.</small>
    </details>
  </section>;
}
