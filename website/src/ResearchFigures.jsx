import React from 'react';
import { asset } from './content';

const figures = [
  {
    id: 'transfer',
    number: '01',
    source: 'PAPER FIGURE 3 · UNSEEN-MOTION TRANSFER',
    title: 'Transfer beyond the training motions.',
    description: 'On eight fast-dance references not used to train the expert, enabling it increases full completion from 59.38% to 71.88% and mean motion completion from 73.24% to 83.83%.',
    file: 'unseen-dance-transfer.png',
    width: 4320,
    height: 1512,
    alt: 'Paper Figure 3: Eight unseen dance references and three simulation repeats. Full completion increases from 114/192 to 138/192 episodes; mean completion rises from 73.24% to 83.83%. In one replay, joint and aligned-body RMSE decrease by 7.10% and 31.51% over the first 66 jointly surviving frames.',
    note: '8 references · 3 simulation repeats · 192 episodes per policy. The right panel uses 66 shared frames from one replay. No policy updates during testing.',
  },
  {
    id: 'teacher-comparison',
    number: '02',
    source: 'PAPER FIGURE 5 · TEACHER COMPARISON',
    title: 'Compare control accuracy after completion.',
    description: 'Across five tasks where both policies complete all 16 evaluation episodes, HoloMotion with an expert has 27.40% lower mean endpoint error and 14.70% lower mean root-position error than the task teachers.',
    file: 'expert-teacher-errors.png',
    width: 4320,
    height: 1596,
    alt: 'Paper Figure 5: Five-task comparison of HoloMotion with an expert and task teachers. Mean endpoint errors are 13.78 versus 18.98 cm; root-position errors are 9.73 versus 11.41 cm. Both errors are lower on four tasks and higher on S05.',
    note: 'Five tasks with 16/16 completions for both policies, averaged equally. Both errors are lower on four tasks and higher on one. Positive heatmap values indicate lower error.',
  },
];

export default function ResearchFigures() {
  return <section className="section paper-results" id="results" aria-labelledby="paper-results-title">
    <div className="paper-results-heading reveal">
      <div>
        <p className="eyebrow">03 / RESULTS</p>
        <h2 id="paper-results-title">Motion, backed by measurements.</h2>
      </div>
      <p>Two experiments from the paper.<br/>Select a figure to view the full-resolution image.</p>
    </div>
    <div className="paper-figures">
      {figures.map(figure => <figure className="paper-figure reveal" key={figure.id}>
        <div className="paper-figure-heading">
          <div>
            <p className="paper-figure-source"><span>{figure.number}</span>{figure.source}</p>
            <h3>{figure.title}</h3>
          </div>
          <p>{figure.description}</p>
        </div>
        <div className="paper-figure-scroll" role="region" tabIndex="0" aria-label={`${figure.source}; scroll horizontally on small screens`}>
          <a className="paper-figure-image" href={asset(`figures/paper/${figure.file}`)} target="_blank" rel="noreferrer" aria-label={`Enlarge ${figure.source}`}>
            <img src={asset(`figures/paper/${figure.file}`)} width={figure.width} height={figure.height} alt={figure.alt} loading="lazy"/>
          </a>
        </div>
        <figcaption>{figure.note}</figcaption>
      </figure>)}
    </div>
  </section>;
}
