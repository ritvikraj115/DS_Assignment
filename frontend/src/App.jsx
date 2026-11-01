import { useEffect, useMemo, useRef, useState } from 'react'
import { ArrowUp, BookOpenText, Check, ChevronDown, CircleHelp, FileText, LoaderCircle, Plus, Sparkles, WandSparkles } from 'lucide-react'
import Sidebar from './components/Sidebar.jsx'
import ChatMessage from './components/ChatMessage.jsx'
import DocumentsPage from './pages/DocumentsPage.jsx'
import SettingsPage from './pages/SettingsPage.jsx'
import { checkHealth, deleteConversation, friendlyError, streamQuestion } from './services/api.js'

const SUGGESTIONS = [
  { icon: FileText, title: 'Explain a concept', text: 'What are the key concepts in my documentation?' },
  { icon: BookOpenText, title: 'Find an example', text: 'Show me a code example from the docs.' },
  { icon: CircleHelp, title: 'Troubleshoot', text: 'How do I troubleshoot a common setup issue?' },
]

function newSession() {
  return { id: crypto.randomUUID(), title: 'New conversation', messages: [], updatedAt: Date.now() }
}

function readSavedChats() {
  try {
    const saved = JSON.parse(localStorage.getItem('documind-chats') || '[]')
    return Array.isArray(saved) ? saved : []
  } catch { return [] }
}

export default function App() {
  const [page, setPage] = useState('chat')
  const [chats, setChats] = useState(readSavedChats)
  const [activeChatId, setActiveChatId] = useState(() => localStorage.getItem('documind-active-chat') || null)
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [apiStatus, setApiStatus] = useState('checking')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const conversationEnd = useRef(null)
  const textarea = useRef(null)

  const activeChat = useMemo(() => chats.find((chat) => chat.id === activeChatId) || null, [chats, activeChatId])
  const messages = activeChat?.messages || []

  useEffect(() => {
    localStorage.setItem('documind-chats', JSON.stringify(chats))
  }, [chats])

  useEffect(() => {
    if (activeChatId) localStorage.setItem('documind-active-chat', activeChatId)
    else localStorage.removeItem('documind-active-chat')
  }, [activeChatId])

  useEffect(() => {
    let alive = true
    checkHealth().then((data) => alive && setApiStatus(data.status === 'ok' ? 'online' : 'offline'))
      .catch(() => alive && setApiStatus('offline'))
    const interval = window.setInterval(() => checkHealth().then((data) => alive && setApiStatus(data.status === 'ok' ? 'online' : 'offline')).catch(() => alive && setApiStatus('offline')), 45000)
    return () => { alive = false; window.clearInterval(interval) }
  }, [])

  useEffect(() => { conversationEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [messages.length, busy])

  useEffect(() => {
    function onKeyDown(event) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        startNewChat()
      }
      if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
        event.preventDefault()
        submitQuestion()
      }
      if (event.key === 'Escape') setSidebarOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  })

  function updateChat(id, updater) {
    setChats((current) => current.map((chat) => chat.id === id ? updater(chat) : chat).sort((a, b) => b.updatedAt - a.updatedAt))
  }

  function startNewChat() {
    const chat = newSession()
    setChats((current) => [chat, ...current])
    setActiveChatId(chat.id)
    setPage('chat')
    setDraft('')
    setError('')
    setSidebarOpen(false)
    window.setTimeout(() => textarea.current?.focus(), 50)
  }

  function selectChat(id) {
    setActiveChatId(id)
    setPage('chat')
    setError('')
    setSidebarOpen(false)
  }

  async function deleteChat(id) {
    const chat = chats.find((item) => item.id === id)
    if (!chat) return
    if (busy && activeChatId === id) {
      setError('Wait for the current answer to finish before deleting this conversation.')
      return
    }
    if (!window.confirm(`Delete “${chat.title || 'this conversation'}”, its conversation memory, and its feedback?`)) return
    try {
      await deleteConversation(id)
      setChats((current) => current.filter((item) => item.id !== id))
      if (activeChatId === id) {
        setActiveChatId(null)
        setPage('chat')
      }
      setError('')
    } catch (deleteError) {
      setError(friendlyError(deleteError))
    }
  }

  async function submitQuestion(explicitQuestion) {
    const question = (explicitQuestion ?? draft).trim()
    if (!question || busy) return
    let chat = activeChat
    if (!chat) {
      chat = newSession()
      setChats((current) => [chat, ...current])
      setActiveChatId(chat.id)
    }
    const sessionId = chat.id
    const userMessage = { id: crypto.randomUUID(), role: 'user', content: question }
    const assistantId = crypto.randomUUID()
    const assistantMessage = { id: assistantId, role: 'assistant', content: '', status: 'streaming', sources: [], question, progress: [] }
    updateChat(sessionId, (current) => ({
      ...current,
      title: current.messages.length ? current.title : question.length > 48 ? `${question.slice(0, 48)}…` : question,
      messages: [...current.messages, userMessage, assistantMessage], updatedAt: Date.now(),
    }))
    setDraft('')
    setBusy(true)
    setError('')
    try {
      const response = await streamQuestion({
        question, sessionId,
        onStage: (event) => updateChat(sessionId, (current) => ({
          ...current,
          messages: current.messages.map((message) => message.id !== assistantId ? message : {
            ...message,
            progress: [...(message.progress || []).map((step) => ({ ...step, state: 'complete' })),
              { stage: event.stage, label: event.label, detail: event.detail, matches: event.matches, state: 'active' }],
          }),
          updatedAt: Date.now(),
        })),
        onToken: (text) => updateChat(sessionId, (current) => ({
          ...current,
          messages: current.messages.map((message) => message.id === assistantId ? { ...message, content: text === null ? '' : message.content + text } : message),
          updatedAt: Date.now(),
        })),
      })
      updateChat(sessionId, (current) => ({
        ...current,
        messages: current.messages.map((message) => message.id !== assistantId ? message : {
          ...message, content: response.answer || 'The assistant returned an empty answer.', status: response.status,
          confidence: response.confidence, sources: response.sources || [],
          progress: [...(message.progress || []).map((step) => ({ ...step, state: 'complete' })),
            { stage: 'completed', label: 'Answer complete',
              detail: response.status === 'passed' ? 'The answer passed verification and streamed to the chat.' : 'The response is ready.',
              state: 'complete' }],
        }),
        updatedAt: Date.now(),
      }))
      if (response.status !== 'passed' && response.status !== 'insufficient_evidence') setError(response.answer || 'The answer could not be verified.')
    } catch (requestError) {
      const backend = requestError?.response?.data
      if (backend?.answer) {
        updateChat(sessionId, (current) => ({ ...current, messages: current.messages.map((message) => message.id === assistantId ? {
          ...message, content: backend.answer, status: backend.status || 'unverified', confidence: backend.confidence,
          sources: backend.sources || [], progress: [],
        } : message), updatedAt: Date.now() }))
      } else {
        setError(friendlyError(requestError))
        updateChat(sessionId, (current) => ({ ...current, messages: current.messages.map((message) => message.id === assistantId ? {
          ...message, content: friendlyError(requestError), status: 'error', progress: [],
        } : message), updatedAt: Date.now() }))
      }
    } finally {
      setBusy(false)
    }
  }

  function handleSubmit(event) {
    event.preventDefault()
    submitQuestion()
  }

  const pageLabel = page === 'chat' ? 'Assistant' : page === 'documents' ? 'Documents' : 'Settings'

  return <div className={`app-frame ${sidebarOpen ? 'mobile-sidebar-open' : ''}`}>
    <button className="mobile-scrim" aria-label="Close menu" onClick={() => setSidebarOpen(false)} />
    <Sidebar page={page} setPage={(value) => { setPage(value); setSidebarOpen(false) }} chats={chats.filter((chat) => chat.messages.length)} activeChatId={activeChatId} onNewChat={startNewChat} onSelectChat={selectChat} onDeleteChat={deleteChat} />
    <main className="main-area">
      <header className="topbar">
        <div className="topbar-left"><button className="mobile-menu-button" aria-label="Open menu" onClick={() => setSidebarOpen(true)}><ChevronDown size={17} /></button><span className="breadcrumb-muted">Workspace</span><span className="breadcrumb-separator">/</span><strong>{pageLabel}</strong></div>
        <div className="topbar-right"><span className={`api-pill api-${apiStatus}`}><span />{apiStatus === 'online' ? 'API connected' : apiStatus === 'checking' ? 'Connecting' : 'API offline'}</span></div>
      </header>

      {page === 'documents' ? <DocumentsPage /> : page === 'settings' ? <SettingsPage /> : <div className={`chat-page ${messages.length ? 'chat-page-active' : ''}`}>
        {messages.length === 0 ? <section className="welcome-view">
          <div className="welcome-kicker"><span className="welcome-kicker-icon"><WandSparkles size={14} /></span> YOUR DOCUMENTATION, IN FOCUS</div>
          <h1>Answers live in your docs.<br /><span>Let’s find them.</span></h1>
          <p className="welcome-subtitle">Ask a question about your technical documentation and get a clear, cited answer you can trust.</p>
          <form className="question-composer welcome-composer" onSubmit={handleSubmit}>
            <textarea ref={textarea} rows={2} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submitQuestion() } }} placeholder="Ask anything about your documentation…" aria-label="Ask a documentation question" />
            <div className="composer-bottom"><span><Sparkles size={13} /> Answers grounded in your indexed sources</span><button className="send-button" type="submit" disabled={!draft.trim() || busy} aria-label="Send question"><ArrowUp size={18} /></button></div>
          </form>
          {error && <div className="chat-error">{error}</div>}
          <div className="suggestion-heading">TRY ASKING <span>·</span> GET STARTED</div>
          <div className="suggestion-grid">{SUGGESTIONS.map(({ icon: Icon, title, text }) => <button className="suggestion-card" key={title} onClick={() => submitQuestion(text)} disabled={busy}><div className="suggestion-icon"><Icon size={15} /></div><div><strong>{title}</strong><span>{text}</span></div><ArrowUp size={14} className="suggestion-arrow" /></button>)}</div>
          <div className="welcome-footer"><span><Check size={13} /> Source-backed responses</span><span className="footer-divider" /><span>Private to your workspace</span></div>
        </section> : <>
          <div className="conversation-scroll">
            <div className="conversation-column">
              <div className="conversation-title"><div><span className="conversation-overline">DOCUMENTATION ASSISTANT</span><h1>{activeChat?.title || 'Conversation'}</h1></div><button className="new-chat-compact" onClick={startNewChat}><Plus size={15} /> New chat</button></div>
              {messages.map((message) => <ChatMessage key={message.id} message={message} sessionId={activeChatId} />)}
              {busy && !messages.some((message) => message.status === 'streaming') && <div className="thinking-row"><div className="assistant-avatar"><BotIcon /></div><div><strong>Reviewing your documentation</strong></div></div>}
              {error && <div className="chat-error conversation-error">{error}</div>}
              <div ref={conversationEnd} />
            </div>
          </div>
          <div className="chat-input-dock"><form className="question-composer followup-composer" onSubmit={handleSubmit}>
            <textarea ref={textarea} rows={1} value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submitQuestion() } }} placeholder="Ask a follow-up question…" aria-label="Ask a follow-up question" />
            <div className="composer-bottom"><span><Sparkles size={13} /> Each answer is checked against your docs</span><button className="send-button" type="submit" disabled={!draft.trim() || busy} aria-label="Send question">{busy ? <LoaderCircle size={17} className="spin" /> : <ArrowUp size={18} />}</button></div>
          </form><div className="input-disclaimer">DocuMind can make mistakes. Verify important details against the cited source.</div></div>
        </>}
      </div>}
    </main>
  </div>
}

function BotIcon() {
  return <BookOpenText size={17} />
}
