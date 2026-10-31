import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowDownToLine, Check, CircleAlert, FileCode2, FileText, Globe2, LoaderCircle, RefreshCw, UploadCloud, X } from 'lucide-react'
import { fetchDocuments, friendlyError, ingestFile, ingestUrl } from '../services/api.js'

const allowedExtensions = ['.pdf', '.docx', '.pptx', '.xlsx', '.csv', '.json', '.md', '.markdown', '.txt', '.html', '.htm']
const statusCopy = {
  indexed: { title: 'Indexed', tone: 'success' },
  updated: { title: 'Updated', tone: 'info' },
  already_indexed: { title: 'Already indexed', tone: 'neutral' },
}

function formatDate(value) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.valueOf()) ? '—' : new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', year: 'numeric' }).format(date)
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [pageError, setPageError] = useState('')
  const [notice, setNotice] = useState(null)
  const [busy, setBusy] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(null)
  const [url, setUrl] = useState('')
  const [dragging, setDragging] = useState(false)
  const fileInput = useRef(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setPageError('')
    try {
      setDocuments(await fetchDocuments())
    } catch (error) {
      setPageError(friendlyError(error))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  async function upload(file) {
    if (!file) return
    const extension = `.${file.name.split('.').pop()}`.toLowerCase()
    if (!allowedExtensions.includes(extension)) {
      setNotice({ type: 'error', text: 'Choose a PDF, DOCX, PPTX, XLSX, CSV, JSON, Markdown, TXT, or HTML file.' })
      return
    }
    setBusy(true)
    setUploadProgress(0)
    setNotice({ type: 'progress', text: 'Uploading document…' })
    try {
      const result = await ingestFile(file, setUploadProgress)
      setNotice({ type: result.status === 'already_indexed' ? 'neutral' : 'success', text: `${statusCopy[result.status]?.title || 'Processed'}: ${result.message || file.name}` })
      await refresh()
    } catch (error) {
      setNotice({ type: 'error', text: friendlyError(error) })
    } finally {
      setBusy(false)
      setUploadProgress(null)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  async function submitUrl(event) {
    event.preventDefault()
    const value = url.trim()
    if (!value) return
    setBusy(true)
    setNotice({ type: 'progress', text: 'Processing documentation URL…' })
    try {
      const result = await ingestUrl(value)
      setNotice({ type: result.status === 'already_indexed' ? 'neutral' : 'success', text: `${statusCopy[result.status]?.title || 'Processed'}: ${result.message || value}` })
      setUrl('')
      await refresh()
    } catch (error) {
      setNotice({ type: 'error', text: friendlyError(error) })
    } finally {
      setBusy(false)
    }
  }

  function onDrop(event) {
    event.preventDefault()
    setDragging(false)
    upload(event.dataTransfer.files?.[0])
  }

  return <div className="page-shell documents-page">
    <header className="page-heading-row">
      <div><div className="eyebrow">KNOWLEDGE BASE</div><h1>Documents</h1><p>Manage the technical references behind your answers.</p></div>
      <button className="secondary-button" onClick={refresh} disabled={loading}><RefreshCw size={15} className={loading ? 'spin' : ''} /> Refresh</button>
    </header>

    <div className="document-ingest-grid">
      <section className={`upload-card ${dragging ? 'drop-active' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
        <div className="upload-icon"><UploadCloud size={20} /></div>
        <div><h2>Upload a document</h2><p>Drag a file here, or choose one from your device.</p></div>
        <button className="primary-button" disabled={busy} onClick={() => fileInput.current?.click()}><ArrowDownToLine size={15} /> Choose file</button>
        <input ref={fileInput} type="file" accept=".pdf,.docx,.pptx,.xlsx,.csv,.json,.md,.markdown,.txt,.html,.htm,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,text/csv,application/json,text/markdown,text/plain,text/html" hidden onChange={(event) => upload(event.target.files?.[0])} />
        <div className="file-types"><FileText size={13} /> PDF <span>·</span> DOCX <span>·</span> PPTX <span>·</span> XLSX <span>·</span> CSV <span>·</span> JSON <span>·</span> Markdown / TXT / HTML</div>
        <small className="file-size-note">Scanned PDFs need OCR before upload. Word format: DOCX.</small>
        {busy && uploadProgress != null && <div className="upload-progress"><div><span>Uploading</span><span>{uploadProgress}%</span></div><div className="upload-progress-track"><span style={{ width: `${uploadProgress}%` }} /></div></div>}
      </section>

      <section className="url-card">
        <div className="url-card-title"><div className="url-icon"><Globe2 size={17} /></div><div><h2>Import from URL</h2><p>Index a public technical documentation page.</p></div></div>
        <form onSubmit={submitUrl} className="url-form"><label htmlFor="document-url">Documentation URL</label><div className="url-input-row"><input id="document-url" type="url" placeholder="https://docs.example.com/guide" value={url} onChange={(event) => setUrl(event.target.value)} disabled={busy} required /><button className="primary-button" disabled={busy || !url.trim()}>{busy ? <LoaderCircle size={15} className="spin" /> : <Globe2 size={15} />}{busy ? 'Working' : 'Import'}</button></div></form>
        <div className="url-footnote"><Check size={13} /> Content is parsed, chunked, and indexed for grounded answers.</div>
      </section>
    </div>

    {notice && <div className={`ingestion-notice notice-${notice.type}`} role="status">
      {notice.type === 'error' ? <CircleAlert size={15} /> : notice.type === 'progress' ? <LoaderCircle size={15} className="spin" /> : <Check size={15} />}
      <span>{notice.text}</span>{notice.type !== 'progress' && <button aria-label="Dismiss message" onClick={() => setNotice(null)}><X size={14} /></button>}
    </div>}
    {pageError && <div className="inline-error"><CircleAlert size={16} />{pageError}<button onClick={refresh}>Try again</button></div>}

    <section className="library-section">
      <div className="library-heading"><div><h2>Indexed documents</h2><p>Documents available to the assistant.</p></div><span className="document-count">{documents.length} {documents.length === 1 ? 'document' : 'documents'}</span></div>
      <div className="documents-table-wrap">
        <table className="documents-table"><thead><tr><th>DOCUMENT</th><th>SOURCE</th><th>VERSION</th><th>CHUNKS</th><th>STATUS</th><th>UPDATED</th></tr></thead>
          <tbody>{loading ? <tr><td colSpan="6"><div className="table-loading"><LoaderCircle size={17} className="spin" /> Loading documents…</div></td></tr> : documents.length === 0 ? <tr><td colSpan="6"><div className="table-empty"><div className="empty-file-icon"><FileCode2 size={20} /></div><strong>Your knowledge base is empty</strong><span>Upload a document or import a URL to get started.</span></div></td></tr> : documents.map((doc) => <tr key={doc.document_id}>
            <td><div className="document-name-cell"><div className="document-type-icon"><FileText size={16} /></div><div><strong>{doc.filename}</strong><small>{doc.document_id}</small></div></div></td>
            <td><span className="source-cell" title={doc.source_url || 'Local upload'}>{doc.source_url ? <><Globe2 size={13} />{doc.source_url}</> : 'Local upload'}</span></td>
            <td><span className="version-pill">v{doc.version}</span></td><td>{doc.chunk_count}</td>
            <td><span className="table-status status-success"><span />Indexed</span></td><td>{formatDate(doc.updated_at)}</td>
          </tr>)}</tbody>
        </table>
      </div>
    </section>
  </div>
}
