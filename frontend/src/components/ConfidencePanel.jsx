import { ShieldCheck, ShieldAlert, CircleHelp } from 'lucide-react'
import { confidenceThresholds } from '../services/api.js'

function scoreLabel(score) {
  if (score == null) return 'Not scored'
  if (score >= confidenceThresholds.high) return 'High'
  if (score >= confidenceThresholds.moderate) return 'Moderate'
  return 'Low'
}

function ScoreBar({ label, value }) {
  return (
    <div className="score-row">
      <div className="score-label"><span>{label}</span><strong>{value == null ? '—' : `${value}%`}</strong></div>
      <div className="score-track"><span style={{ width: `${Math.max(0, Math.min(100, value ?? 0))}%` }} /></div>
    </div>
  )
}

export default function ConfidencePanel({ confidence, status }) {
  const verified = status === 'passed' && confidence?.status === 'passed' && confidence?.overall != null
  const value = confidence?.overall
  const label = verified ? scoreLabel(value) : 'Unverified'
  return (
    <section className={`confidence-card ${verified ? '' : 'confidence-unverified'}`} aria-label="Answer evaluation">
      <div className="confidence-top">
        <div className={`confidence-icon ${verified ? 'confidence-icon-good' : 'confidence-icon-muted'}`}>
          {verified ? <ShieldCheck size={17} /> : <ShieldAlert size={17} />}
        </div>
        <div className="confidence-title"><strong>{verified ? 'Grounded answer' : 'Verification status'}</strong><span>{verified ? 'Evaluated against your documentation' : 'This response was not verified'}</span></div>
        <div className="confidence-overall"><strong>{verified ? `${value}%` : '—'}</strong><span>{label} confidence</span></div>
      </div>
      {verified ? <div className="confidence-scores">
        <ScoreBar label="Faithfulness" value={confidence.faithfulness} />
        <ScoreBar label="Relevance" value={confidence.relevance} />
        <ScoreBar label="Correctness" value={confidence.correctness} />
      </div> : <div className="confidence-note"><CircleHelp size={14} /> {confidence?.status === 'insufficient_evidence' ? 'There was not enough relevant evidence to answer.' : 'No confidence score is shown for an unverified answer.'}</div>}
      {verified && confidence?.reasons && <details className="evaluation-details">
        <summary>Evaluation notes</summary>
        <div className="evaluation-reasons">{['faithfulness', 'relevance', 'correctness'].map((key) => confidence.reasons[key] && <p key={key}><b>{key[0].toUpperCase() + key.slice(1)}:</b> {confidence.reasons[key]}</p>)}</div>
      </details>}
    </section>
  )
}
