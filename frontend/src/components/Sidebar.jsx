import { useState } from 'react'
import {
  BookOpenText, ChevronRight, Clock3, FileText, MessageSquarePlus, Trash2,
  PanelLeftClose, PanelLeftOpen, Settings2, Sparkles,
} from 'lucide-react'

export default function Sidebar({ page, setPage, chats, activeChatId, onNewChat, onSelectChat, onDeleteChat }) {
  const [collapsed, setCollapsed] = useState(false)
  const navItems = [
    { id: 'chat', label: 'Assistant', icon: Sparkles },
    { id: 'documents', label: 'Documents', icon: FileText },
    { id: 'settings', label: 'Settings', icon: Settings2 },
  ]

  return (
    <aside className={`sidebar ${collapsed ? 'sidebar-collapsed' : ''}`}>
      <div className="brand-row">
        <div className="brand-mark"><BookOpenText size={19} strokeWidth={2.2} /></div>
        {!collapsed && <div className="brand-copy"><span>DocuMind</span><small>TECHNICAL ASSISTANT</small></div>}
        <button className="icon-button collapse-button" aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} onClick={() => setCollapsed(!collapsed)}>
          {collapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
        </button>
      </div>

      <button className="new-chat-button" onClick={onNewChat} title="Start a new chat">
        <MessageSquarePlus size={17} /> {!collapsed && <span>New chat</span>}
        {!collapsed && <span className="new-chat-shortcut">⌘ K</span>}
      </button>

      <nav className="side-nav" aria-label="Main navigation">
        {!collapsed && <div className="nav-caption">WORKSPACE</div>}
        {navItems.map(({ id, label, icon: Icon }) => (
          <button key={id} className={`nav-item ${page === id ? 'nav-item-active' : ''}`} onClick={() => setPage(id)} title={collapsed ? label : undefined}>
            <Icon size={17} strokeWidth={1.9} /><span>{!collapsed && label}</span>
          </button>
        ))}
      </nav>

      {!collapsed && <div className="chat-history">
        <div className="history-heading"><span>CHAT HISTORY</span><Clock3 size={14} /></div>
        <div className="history-list">
          {chats.length === 0 ? <p className="history-empty">Your conversations will appear here.</p> : chats.map((chat) => (
            <div key={chat.id} className="history-entry">
              <button className={`history-item ${activeChatId === chat.id && page === 'chat' ? 'history-item-active' : ''}`} onClick={() => onSelectChat(chat.id)}>
                <span className="history-dot" /><span className="history-title">{chat.title || 'New conversation'}</span>
                {activeChatId === chat.id && <ChevronRight size={14} className="history-chevron" />}
              </button>
              <button className="history-delete" aria-label={`Delete ${chat.title || 'conversation'}`} title="Delete conversation" onClick={() => onDeleteChat(chat.id)}><Trash2 size={13} /></button>
            </div>
          ))}
        </div>
      </div>}

    </aside>
  )
}
