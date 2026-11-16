import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { asset, project, heroPreview } from './content';
import ResearchFigures from './ResearchFigures';
import ExpertGallery from './ExpertGallery';
import './styles.css';

const Arrow = () => <span aria-hidden="true">↗</span>;
const Play = ({pause=false}) => <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">{pause ? <path d="M8 5v14M16 5v14" stroke="currentColor" strokeWidth="3"/> : <path d="m9 5 11 7-11 7z" fill="currentColor"/>}</svg>;
function useReducedMotion() {
  const [reduced, setReduced] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  useEffect(() => { const m = window.matchMedia('(prefers-reduced-motion: reduce)'); const update = () => setReduced(m.matches); m.addEventListener('change',update); return () => m.removeEventListener('change',update); },[]);
  return reduced;
}
function Video({id, title, className='', auto=true, loop=true, onReady, onTimeUpdate}) {
  const ref = useRef(null);
  const reduced = useReducedMotion();
  const [playing,setPlaying] = useState(false);
  const [loaded,setLoaded] = useState(false);
  const [error,setError] = useState(false);
  const manuallyPaused = useRef(false);
  const endedRef = useRef(false);const [ended,setEnded]=useState(false);
  useEffect(() => {
    const video = ref.current;
    const source = asset(`media/${id}.mp4`);
    // HMR and reused players can receive a new recording without remounting.
    // Always reset the native source and playback state together.
    if (video.getAttribute('src') && video.getAttribute('src') !== source) {
      video.pause(); video.removeAttribute('src'); video.load();
    }
    endedRef.current=false;manuallyPaused.current=false;
    setEnded(false);setError(false);setPlaying(false);
    setLoaded(video.getAttribute('src') === source);
    const observer = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) {
        if (video.getAttribute('src') !== source) { video.src = source; video.load(); setLoaded(true); }
        if (auto && !reduced && !manuallyPaused.current && !endedRef.current && !document.hidden) video.play().catch(()=>{});
      } else video.pause();
    }, {threshold: 0.15});
    observer.observe(video);
    const visibility = () => { if(document.hidden) video.pause(); };
    document.addEventListener('visibilitychange',visibility);
    return () => { observer.disconnect(); video.pause(); document.removeEventListener('visibilitychange',visibility); };
  },[id,auto,reduced]);
  useEffect(()=>{if(reduced)ref.current?.pause();},[reduced]);
  function toggle() {
    const video=ref.current;
    if(playing) { manuallyPaused.current=true; video.pause(); }
    else { manuallyPaused.current=false; if(endedRef.current){video.currentTime=0;endedRef.current=false;setEnded(false);} if(!loaded){video.src=asset(`media/${id}.mp4`);setLoaded(true);} video.play().catch(()=>setError(true)); }
  }
  return <div className={`video-wrap ${className}`}>
    <video ref={ref} poster={asset(`media/${id}.webp`)} muted loop={loop} playsInline preload="none" onEnded={()=>{endedRef.current=true;setEnded(true);setPlaying(false);}} aria-label={title} onPlay={()=>setPlaying(true)} onPause={()=>setPlaying(false)} onCanPlay={onReady} onTimeUpdate={e=>onTimeUpdate?.(e.currentTarget.currentTime)} onError={()=>setError(true)} />
    <button className="video-toggle" onClick={toggle} aria-label={`${ended?'Replay':playing?'Pause':'Play'} ${title}`}><Play pause={playing}/></button>
    {ended && <span className="recording-ended">Recording complete</span>}
    {error && <div className="media-error">Video playback unavailable. <a href={asset(`media/${id}.mp4`)}>Open the MP4 <Arrow/></a></div>}
  </div>;
}
function FilmDialog({film,onClose}) {
  const ref=useRef(null);
  useEffect(()=>{
    const previous=document.activeElement;
    const d=ref.current; d.showModal();
    const close=()=>onClose(); d.addEventListener('close',close);
    return ()=>{d.removeEventListener('close',close);d.close();previous?.focus();};
  },[onClose]);
  return <dialog className="film-dialog" aria-label={film.title} ref={ref} onClick={e=>{if(e.target===ref.current)onClose();}}>
    <div className="film-header"><span>{film.title}</span><button autoFocus onClick={onClose} aria-label="Close video">✕</button></div>
    <video src={asset(`media/${film.id}.mp4`)} poster={asset(`media/${film.id}.webp`)} controls autoPlay muted playsInline/>
    <p>{film.note} {film.designPreview?'One visual scene · animated joins · no policy executed in this film.':film.editedHighlights?'Selected excerpts.':film.neutralRecording?'':'Original speed · recorded robot motion.'} {film.fullId&&<a href={asset(`media/${film.fullId}.mp4`)} target="_blank" rel="noreferrer">Watch the complete source recording <Arrow/></a>} {film.fullRecordings&&<span className="film-sources">Full recordings: {film.fullRecordings.map((source,i)=><React.Fragment key={source.id}>{i>0?' · ':''}<a href={asset(`media/${source.id}.mp4`)} target="_blank" rel="noreferrer">{source.label}</a></React.Fragment>)}</span>}</p>
  </dialog>;
}
function Header(){
  const [open,setOpen]=useState(false);
  return <header className="header"><div className="nav-inner">
    <a className="wordmark" href="#top" aria-label="PERA home"><span className="brand-icon" aria-hidden="true"><i/><i/><i/></span>PERA<span className="brand-descriptor">Humanoid Learning</span></a>
    <button className="menu-toggle" aria-expanded={open} aria-controls="main-nav" onClick={()=>setOpen(!open)}>{open?'Close':'Menu'} <span aria-hidden="true">{open?'−':'+'}</span></button>
    <nav id="main-nav" className={open?'open':''} aria-label="Main navigation">{[['Overview','overview'],['Skills','skills'],['Results','results']].map(([label,id])=><a key={id} href={`#${id}`} onClick={()=>setOpen(false)}>{label}</a>)}</nav>
  </div></header>;
}
function Motivation(){
 return <section className="section motivation" id="overview">
   <div className="section-heading reveal"><p className="eyebrow">01 / THE IDEA</p><h2>One policy. <span>More skills.</span></h2></div>
   <figure className="policy-concept-figure reveal">
     <a href={asset('concept-one-policy.svg')} target="_blank" rel="noreferrer" aria-label="Enlarge the shared-policy and task-expert concept figure">
       <img src={asset('concept-one-policy.svg')} width="2048" height="1202" loading="lazy" alt="Task experts extend a frozen base model within one shared policy, with explicit expert on/off control."/>
     </a>
     <figcaption>Freeze the base. Add skills on demand. Execute them through one shared policy.</figcaption>
   </figure>
 </section>;
}
function Skills({onOpenFilm}) { return <ExpertGallery onOpenFilm={onOpenFilm} Video={Video}/>; }
function App(){
 const [film,setFilm]=useState(null);const closeFilm=React.useCallback(()=>setFilm(null),[]);

 useEffect(()=>{const elements=document.querySelectorAll('.reveal');const observer=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add('visible');observer.unobserve(e.target);}}),{threshold:0.06});elements.forEach(e=>observer.observe(e));return()=>observer.disconnect();},[]);
 return <><a className="skip-link" href="#main">Skip to content</a><Header/><main id="main"><section className="hero" id="top"><div className="hero-intro"><p className="eyebrow hero-eyebrow"><span className="blue-dot"/> PERA / HUMANOID SKILL LEARNING</p><h1>Skills as <span>Plug-ins.</span></h1><p className="hero-subtitle">{project.subtitle}</p><p className="hero-description">Extend the repertoire. Activate an expert.<br className="mobile-break"/> Keep the shared policy frozen.</p><div className="hero-resources" aria-label="Research links">{project.paper?<a className="resource-button" href={project.paper}>Paper <Arrow/></a>:<span className="resource-button pending" aria-disabled="true">Paper <small>Coming soon</small></span>}<a className="resource-button" href={project.code} target="_blank" rel="noreferrer">Code <Arrow/><small>{project.codeStatus}</small></a></div><div className="hero-actions"><a className="button primary" href="#skills">Explore the skills <span aria-hidden="true">↘</span></a><button className="film-link" onClick={()=>setFilm({...heroPreview})}>Watch the highlights <span className="play-circle"><Play/></span></button></div></div><div className="hero-visual"><Video id={heroPreview.id} title="PERA native-scene skill highlights" loop={false}/></div><p className="hero-preview-note"><b>Recorded highlights</b> Four native scenes · Same frozen Actor · Original speed. <a href="#skills">View recorded comparisons <Arrow/></a></p></section><Motivation/><Skills onOpenFilm={setFilm}/><ResearchFigures/></main><footer><a className="wordmark" href="#top">PERA</a><span>Skills as Plug-ins.</span><span>Research snapshot · November 2026</span><a href="#top">Back to top ↑</a></footer>{film&&<FilmDialog film={film} onClose={closeFilm}/>}</>;
}

createRoot(document.getElementById('root')).render(<App/>);
