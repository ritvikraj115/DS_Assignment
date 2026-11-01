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
    <summary><span className="source-file-icon"><FileCode2 size={14} /></span><span className="source-main"><strong>{source.document}</strong><small>{source.section || 'Document'}</small></span><span className="source-chunk">{source.chunk_id}</span></summary>
    <div className="source-detail"><span>Chunk ID</span><code>{source.chunk_id}</code>{source.source_url && <a href={source.source_url} target="_blank" rel="noreferrer">Open source <ExternalLink size={12} /></a>}</div>
  </details>)}</div></div>
}

function Feedback({ message, sessionId }) {
  const [rating, setRating] = useState(null)
  const [busy, setBusy] = useState(false)
  const [done, setDone] = useState(false)
  const [error, setError] = useState('')

  async function send(value) {
    setRating(value)
    setBusy(true)
    setError('')
    try {
      await submitFeedback({ question: message.question, sessionId, rating: value,
        answer: message.content, confidence: message.confidence, sources: message.sources, status: message.status })
      setDone(true)
    } catch {
      setError('Feedback could not be saved. Try again.')
    } finally {
      setBusy(false)
    }
  }
  if (done) return <div className="feedback-confirmation"><Check size={13} /> Thanks, your feedback was recorded.</div>
  return <div className="feedback-row">
    <span>Was this helpful?</span>
    <button className={`feedback-icon ${rating === 'up' ? 'feedback-selected' : ''}`} disabled={busy} onClick={() => send('up')} aria-label="Thumbs up" title="Helpful"><ThumbsUp size={14} /></button>
    <button className={`feedback-icon ${rating === 'down' ? 'feedback-selected' : ''}`} disabled={busy} onClick={() => send('down')} aria-label="Thumbs down" title="Not helpful"><ThumbsDown size={14} /></button>
    {busy && <small>Saving…</small>}{error && <small className="form-error">{error}</small>}
  </div>
}

export default function ChatMessage({ message, sessionId }) {
  if (message.role === 'user') return <div className="message-row user-message"><div className="user-bubble">{message.content}</div><div className="user-avatar">You</div></div>
  const visibleContent = message.content.replace(/\s*\[S\d+\]/gi, '')
  return <div className="message-row assistant-message">
    <div className="assistant-avatar"><Bot size={17} /></div>
    <div className="assistant-content">
      <div className="assistant-name">DocuMind <span>Assistant</span></div>
      {message.progress?.length > 0 && (() => {
        const current = message.progress[message.progress.length - 1]
        const inProgress = message.status === 'streaming'
        return <details className="pipeline-progress" aria-live="polite">
          <summary className="pipeline-summary">
            <span className="pipeline-marker">{inProgress ? <LoaderCircle size={11} className="spin" /> : <Check size={11} />}</span>
            <span className="pipeline-current"><strong>{inProgress ? current.label : 'Answer complete'}</strong>{inProgress && current.stage === 'memory_match' && current.detail && <small className="pipeline-memory-current">{current.detail}</small>}<small>{message.progress.length} steps · {inProgress ? 'click to view progress' : 'click to view completed steps'}</small></span>
          </summary>
          <div className="pipeline-history" aria-label="Answer pipeline progress">
            {message.progress.map((step, index) => <div className={`pipeline-step pipeline-${step.state}`} key={`${step.stage}-${index}`}>
              <span className="pipeline-marker">{step.state === 'complete' || !inProgress ? <Check size={11} /> : <LoaderCircle size={11} className="spin" />}</span>
              <span><strong>{step.label}</strong>{step.detail && <small>{step.detail}</small>}
                {step.matches?.map((match, matchIndex) => <small className="memory-match" key={`${match.question}-${matchIndex}`}>
                  “{match.question}” · {Math.round(match.similarity * 100)}% similar · {match.rating === 'up' ? 'positive feedback' : match.rating === 'down' ? 'negative feedback' : match.rating === 'mixed' ? 'mixed feedback' : 'no feedback yet'}{match.response_count ? ` · ${match.response_count} saved answer${match.response_count === 1 ? '' : 's'}` : ''}
                </small>)}
              </span>
            </div>)}
          </div>
        </details>
      })()}
      {message.status === 'streaming' && message.content && <div className="draft-verification">
        {message.progress?.at(-1)?.stage === 'approved'
          ? <><Check size={12} /> Verified answer · streaming</>
          : <><LoaderCircle size={12} className="spin" /> Preparing verified answer</>}
      </div>}
      <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]} components={{ pre: CodeBlock }}>{visibleContent}</ReactMarkdown></div>
      {message.confidence && <ConfidencePanel confidence={message.confidence} status={message.status} />}
      <SourceList sources={message.sources} />
      {message.status !== 'error' && message.status !== 'streaming' && message.question && <Feedback key={message.id} message={message} sessionId={sessionId} />}
    </div>
  </div>
}
