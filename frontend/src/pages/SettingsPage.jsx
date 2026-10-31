import { useEffect, useState } from 'react'
import { CheckCircle2, CircleAlert, ExternalLink, Globe2, Server, ShieldCheck } from 'lucide-react'
import { api, checkHealth, friendlyError } from '../services/api.js'

export default function SettingsPage() {
  const [health, setHealth] = useState({ status: 'checking' })
  const [healthError, setHealthError] = useState('')
  useEffect(() => {
    checkHealth().then(setHealth).catch((error) => { setHealth({ status: 'offline' }); setHealthError(friendlyError(error)) })
  }, [])
  const baseURL = api.defaults.baseURL
  return <div className="page-shell settings-page">
    <header className="page-heading-row"><div><div className="eyebrow">PREFERENCES</div><h1>Settings</h1><p>Application status and connection details.</p></div></header>
    <section className="settings-card"><div className="settings-section-heading"><div className="settings-icon"><Server size={17} /></div><div><h2>API connection</h2><p>DocuMind communicates with your local FastAPI service.</p></div></div>
      <div className="settings-detail"><span>Service status</span><strong className={`service-health health-${health.status}`}>{health.status === 'ok' ? <CheckCircle2 size={15} /> : health.status === 'checking' ? <span className="health-spinner" /> : <CircleAlert size={15} />}{health.status === 'ok' ? 'Connected' : health.status === 'checking' ? 'Checking…' : 'Unavailable'}</strong></div>
      {healthError && <div className="settings-error">{healthError}</div>}
      <div className="settings-detail"><span>API base URL</span><code>{baseURL}</code></div>
      <div className="settings-detail"><span>Frontend variable</span><code>VITE_API_BASE_URL</code></div>
      <div className="settings-detail"><span>Configuration file</span><code>frontend/.env</code></div>
    </section>
    <section className="settings-card settings-security"><div className="settings-section-heading"><div className="settings-icon"><ShieldCheck size={17} /></div><div><h2>Privacy and grounding</h2><p>Your answers are checked against indexed technical documentation.</p></div></div><div className="settings-callout"><Globe2 size={15} /><span>Frontend requests go only to the API base URL configured for this app. Gemini credentials stay on the backend.</span></div></section>
    <div className="settings-help">Need to change the API address? Set <code>VITE_API_BASE_URL</code> in <code>frontend/.env</code>, then restart the Vite server. <a href={`${baseURL}/docs`} target="_blank" rel="noreferrer">Open API docs <ExternalLink size={12} /></a></div>
  </div>
}
