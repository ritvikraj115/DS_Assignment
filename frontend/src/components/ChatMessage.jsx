import { useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { Bot, Check, Copy, ExternalLink, FileCode2, LoaderCircle, ThumbsDown, ThumbsUp } from 'lucide-react'
import ConfidencePanel from './ConfidencePanel.jsx'
import { submitFeedback } from '../services/api.js'

function CodeBlock({ children, ...props }) {
  const [copied, setCopied] = useState(false)
  const pre = useRef(null)
  async function copy() {
    try {
      await navigator.clipboard.writeText(pre.current?.innerText || '')
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch {
      setCopied(false)
    }
  }
  return <div className="code-block-wrap"><div className="code-block-toolbar"><span><FileCode2 size={13} /> Code</span><button onClick={copy} aria-label="Copy code">{copied ? <Check size={13} /> : <Copy size={13} />}{copied ? 'Copied' : 'Copy'}</button></div><pre ref={pre} {...props}>{children}</pre></div>
}

function SourceList({ sources = [] }) {
  if (!sources.length) return null
  return <div className="sources-wrap"><div className="sources-heading">Sources <span>{sources.length}</span></div><div className="source-list">{sources.map((source) => <details className="source-item" key={source.chunk_id}>
    <summary><span className="source-file-icon"><FileCode2 size={14} /></span><span className="source-main"><strong>{source.document}</strong><small>{source.section || 'Document'}</small></span><span className="source-chunk">{source.label || source.chunk_id}</span></summary>
    <div className="source-detail"><span>Chunk ID</span><code>{source.chunk_id}</code>{source.source_url && <a href={source.source_url} target="_blank" rel="noreferrer">Open source <ExternalLink size={12} /></a>}</div>
  </details>)}</div></div>
}

function Feedback({ message, sessionId }) {
  const [rating, setRating] = useState(null)
  const [commentOpen, setCommentOpen] = useState(false)
  const [comment, setComment] = useState('')
  const [busy, setBusy] = useState(false)
  const [done, setDone] = useState(false)
  const [error, setError] = useState('')

  async function send(value) {
    setRating(value)
    setBusy(true)
    setError('')
    try {
      await submitFeedback({ question: message.question, sessionId, rating: value, comment })
      setDone(true)
      setCommentOpen(false)
    } catch {
      setError('Feedback could not be saved. Try again.')
    } finally {
      setBusy(false)
    }
  }
  if (done) return <div className="feedback-confirmation"><Check size={13} /> Thanks, your feedback was recorded.</div>
  return <div className="feedback-row">
    <span>Was this helpful?</span>
    <button className={`feedback-icon ${rating === 'up' ? 'feedback-selected' : ''}`} disabled={busy} onClick={() => commentOpen ? send('up') : (setRating('up'), setCommentOpen(true))} aria-label="Thumbs up"><ThumbsUp size={14} /></button>
    <button className={`feedback-icon ${rating === 'down' ? 'feedback-selected' : ''}`} disabled={busy} onClick={() => commentOpen ? send('down') : (setRating('down'), setCommentOpen(true))} aria-label="Thumbs down"><ThumbsDown size={14} /></button>
    {commentOpen && <div className="feedback-editor"><textarea value={comment} onChange={(event) => setComment(event.target.value)} placeholder="Add an optional comment" rows={2} /><div><button className="feedback-cancel" onClick={() => { setCommentOpen(false); setRating(null) }}>Cancel</button><button className="feedback-submit" disabled={busy} onClick={() => send(rating)}>{busy ? 'Sending…' : 'Send feedback'}</button></div>{error && <small className="form-error">{error}</small>}</div>}
  </div>
}

export default function ChatMessage({ message, sessionId }) {
  if (message.role === 'user') return <div className="message-row user-message"><div className="user-bubble">{message.content}</div><div className="user-avatar">You</div></div>
  return <div className="message-row assistant-message">
    <div className="assistant-avatar"><Bot size={17} /></div>
    <div className="assistant-content">
      <div className="assistant-name">DocuMind <span>Assistant</span></div>
      {message.progress?.length > 0 && <div className="pipeline-progress" aria-live="polite" aria-label="Answer pipeline progress">
        {message.progress.map((step, index) => <div className={`pipeline-step pipeline-${step.state}`} key={`${step.stage}-${index}`}>
          <span className="pipeline-marker">{step.state === 'complete' ? <Check size={11} /> : <LoaderCircle size={11} className="spin" />}</span>
          <span><strong>{step.label}</strong>{step.state === 'active' && step.detail && <small>{step.detail}</small>}</span>
        </div>)}
      </div>}
      <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]} components={{ pre: CodeBlock }}>{message.content}</ReactMarkdown></div>
      {message.confidence && <ConfidencePanel confidence={message.confidence} status={message.status} />}
      <SourceList sources={message.sources} />
      {message.status !== 'error' && message.status !== 'streaming' && message.question && <Feedback key={message.id} message={message} sessionId={sessionId} />}
    </div>
  </div>
}
