import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import ReactMarkdown from 'react-markdown'

// Custom code block component
function CodeBlock({ inline, className, children, ...props }) {
    const match = /language-(\w+)/.exec(className || '')
    const language = match ? match[1] : ''

    if (inline) {
        return (
            <code className="bg-black/60 text-green-400 px-1.5 py-0.5 text-xs" {...props}>
                {children}
            </code>
        )
    }

    return (
        <div className="relative my-3">
            {language && (
                <span className="absolute top-0 right-0 text-[9px] text-white/30 px-2 py-1 bg-black/80 tracking-wider">
                    {language.toUpperCase()}
                </span>
            )}
            <pre className="code-block">
                <code {...props}>{children}</code>
            </pre>
        </div>
    )
}

function Chat() {
    const [messages, setMessages] = useState([])
    const [input, setInput] = useState('')
    const [connected, setConnected] = useState(false)
    const [currentPolicy, setCurrentPolicy] = useState(null)
    const wsRef = useRef(null)
    const messagesEndRef = useRef(null)

    useEffect(() => {
        connectWebSocket()
        return () => {
            if (wsRef.current) {
                wsRef.current.close()
            }
        }
    }, [])

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    const connectWebSocket = () => {
        const ws = new WebSocket('ws://localhost:8000/api/chat/ws')
        wsRef.current = ws

        ws.onopen = () => setConnected(true)

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data)

            if (data.type === 'init') {
                setMessages(data.messages.map(m => ({
                    role: m.role,
                    content: m.content,
                })))
            } else if (data.type === 'message') {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: data.message.content,
                }])
                if (data.policy) {
                    setCurrentPolicy(data.policy)
                }
            }
        }

        ws.onclose = () => setConnected(false)
    }

    const sendMessage = () => {
        if (!input.trim() || !wsRef.current) return

        setMessages(prev => [...prev, {
            role: 'user',
            content: input,
        }])

        wsRef.current.send(JSON.stringify({
            type: 'message',
            content: input,
        }))

        setInput('')
    }

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            sendMessage()
        }
    }

    const markdownComponents = {
        code: CodeBlock,
        h1: ({ children }) => <h1 className="text-base font-bold mt-4 mb-2 text-white">{children}</h1>,
        h2: ({ children }) => <h2 className="text-sm font-bold mt-3 mb-2 text-white">{children}</h2>,
        h3: ({ children }) => <h3 className="text-xs font-bold mt-2 mb-1 text-white">{children}</h3>,
        p: ({ children }) => <p className="mb-2 text-white/80">{children}</p>,
        ul: ({ children }) => <ul className="list-disc list-inside mb-2 ml-2 text-white/70">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal list-inside mb-2 ml-2 text-white/70">{children}</ol>,
        li: ({ children }) => <li className="mb-1">{children}</li>,
        strong: ({ children }) => <strong className="font-bold text-white">{children}</strong>,
        em: ({ children }) => <em className="italic">{children}</em>,
        blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-white/30 pl-3 my-2 italic text-white/60">
                {children}
            </blockquote>
        ),
        a: ({ href, children }) => (
            <a href={href} className="text-blue-400/80 underline hover:text-blue-400" target="_blank" rel="noopener noreferrer">
                {children}
            </a>
        ),
    }

    return (
        <div className="max-w-6xl mx-auto h-[calc(100vh-4rem)] flex flex-col">
            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-6 flex justify-between items-center"
            >
                <div>
                    <motion.div
                        className="inline-block mb-3"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.2 }}
                    >
                        <span className="text-[10px] tracking-[0.2em] text-white/40 border border-white/20 px-3 py-1">
                            CHAT_MODULE
                        </span>
                    </motion.div>
                    <h1 className="text-2xl font-bold tracking-tight text-white mb-1">
                        CHAT :: ASSISTANT
                    </h1>
                    <p className="text-sm text-white/50">
                        Design policies through natural conversation
                    </p>
                </div>
                <span className={`label-tag ${connected ? 'label-tag-success' : 'label-tag-error'}`}>
                    <span className={`status-dot ${connected ? 'online' : 'offline'} mr-2`} />
                    {connected ? 'CONNECTED' : 'DISCONNECTED'}
                </span>
            </motion.div>

            <div className="flex-1 flex gap-6 min-h-0">
                {/* Chat Messages */}
                <motion.div
                    initial={{ opacity: 0, x: -30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="flex-1 card-window flex flex-col"
                >
                    {/* Window Header */}
                    <div className="card-header">
                        <div className="window-dots">
                            <div className="window-dot red" />
                            <div className="window-dot yellow" />
                            <div className="window-dot green" />
                        </div>
                        <span className="text-[10px] text-white/40">chat_session.log</span>
                    </div>

                    {/* Messages */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-4">
                        {messages.map((msg, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className={`p-4 ${msg.role === 'user'
                                    ? 'message-user ml-12'
                                    : msg.role === 'system'
                                        ? 'message-system'
                                        : 'message-assistant mr-8'
                                    }`}
                            >
                                <span className="text-[9px] tracking-[0.15em] block mb-2 opacity-50">
                                    {msg.role.toUpperCase()}
                                </span>
                                <div className="text-sm markdown-content">
                                    <ReactMarkdown components={markdownComponents}>
                                        {msg.content}
                                    </ReactMarkdown>
                                </div>
                            </motion.div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input */}
                    <div className="p-4 border-t border-white/10 flex gap-3">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder="Describe your dataset or requirements..."
                            className="flex-1"
                            disabled={!connected}
                        />
                        <motion.button
                            onClick={sendMessage}
                            disabled={!connected || !input.trim()}
                            whileHover={{ scale: 1.02 }}
                            whileTap={{ scale: 0.98 }}
                            className="pixel-btn"
                        >
                            SEND →
                        </motion.button>
                    </div>
                </motion.div>

                {/* Policy Preview Sidebar */}
                <motion.div
                    initial={{ opacity: 0, x: 30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                    className="w-80 card flex flex-col"
                >
                    <h2 className="text-sm font-bold tracking-[0.12em] text-white mb-6">
                        CURRENT_POLICY
                    </h2>

                    {currentPolicy ? (
                        <div className="flex-1 overflow-y-auto space-y-5">
                            <div>
                                <span className="text-[10px] tracking-wider text-white/40 block mb-1">NAME</span>
                                <p className="font-bold text-white">{currentPolicy.name}</p>
                            </div>
                            <div>
                                <span className="text-[10px] tracking-wider text-white/40 block mb-1">DOMAIN</span>
                                <span className="label-tag inline-block">
                                    {currentPolicy.domain?.toUpperCase()}
                                </span>
                            </div>
                            <div>
                                <span className="text-[10px] tracking-wider text-white/40 block mb-2">TRANSFORMS</span>
                                <div className="space-y-2">
                                    {currentPolicy.transforms?.map((t, i) => (
                                        <motion.div
                                            key={i}
                                            initial={{ opacity: 0, x: -10 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: i * 0.05 }}
                                            className="p-3 border border-white/10 bg-white/5 text-xs"
                                        >
                                            <span className="font-bold text-white">{t.name}</span>
                                            <span className="text-white/40 ml-2">p={t.probability}</span>
                                        </motion.div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="flex-1 flex items-center justify-center">
                            <div className="text-center">
                                <motion.div
                                    className="w-12 h-12 mx-auto mb-4 text-white/15"
                                    animate={{ rotate: [0, 360] }}
                                    transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
                                >
                                    <svg viewBox="0 0 24 24" className="w-full h-full">
                                        <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="1" strokeDasharray="4 2" />
                                        <circle cx="12" cy="12" r="4" fill="none" stroke="currentColor" strokeWidth="1" />
                                    </svg>
                                </motion.div>
                                <p className="text-xs text-white/30">
                                    Start chatting to generate a policy
                                </p>
                            </div>
                        </div>
                    )}

                    {currentPolicy && (
                        <motion.button
                            whileHover={{ scale: 1.02, y: -2 }}
                            whileTap={{ scale: 0.98 }}
                            className="pixel-btn w-full mt-4 text-[10px]"
                        >
                            EXPORT_POLICY →
                        </motion.button>
                    )}
                </motion.div>
            </div>
        </div>
    )
}

export default Chat
