import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'

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

        ws.onopen = () => {
            setConnected(true)
        }

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

        ws.onclose = () => {
            setConnected(false)
        }
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

    return (
        <div className="max-w-6xl mx-auto h-[calc(100vh-3rem)] flex flex-col">
            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-4 flex justify-between items-center"
            >
                <div>
                    <h1 className="text-2xl font-bold tracking-widest mb-1">
                        CHAT :: ASSISTANT
                    </h1>
                    <p className="text-sm opacity-70">
                        Design policies through natural conversation
                    </p>
                </div>
                <span className={`label-tag ${connected ? 'bg-green-200' : 'bg-red-200'}`}>
                    {connected ? '🟢 CONNECTED' : '🔴 DISCONNECTED'}
                </span>
            </motion.div>

            <div className="flex-1 flex gap-4 min-h-0">
                {/* Chat Messages */}
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="flex-1 card flex flex-col"
                >
                    <div className="flex-1 overflow-y-auto space-y-4 mb-4">
                        {messages.map((msg, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className={`p-3 ${msg.role === 'user'
                                        ? 'bg-black text-white ml-8'
                                        : msg.role === 'system'
                                            ? 'bg-gray-200 text-gray-600 text-sm'
                                            : 'bg-white/80 border-2 border-black mr-8'
                                    }`}
                            >
                                <span className="text-[10px] tracking-widest block mb-1 opacity-50">
                                    {msg.role.toUpperCase()}
                                </span>
                                <div className="text-sm whitespace-pre-wrap">
                                    {msg.content}
                                </div>
                            </motion.div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input */}
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={handleKeyPress}
                            placeholder="Describe your dataset or requirements..."
                            className="flex-1 bg-white/50 border-2 border-black px-4 py-3 text-sm font-mono"
                            disabled={!connected}
                        />
                        <button
                            onClick={sendMessage}
                            disabled={!connected || !input.trim()}
                            className="pixel-btn"
                        >
                            SEND
                        </button>
                    </div>
                </motion.div>

                {/* Policy Preview Sidebar */}
                <motion.div
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                    className="w-80 card flex flex-col"
                >
                    <h2 className="text-sm font-bold tracking-widest mb-4">
                        CURRENT_POLICY
                    </h2>

                    {currentPolicy ? (
                        <div className="flex-1 overflow-y-auto">
                            <div className="mb-4">
                                <span className="text-[10px] tracking-widest">NAME</span>
                                <p className="font-bold">{currentPolicy.name}</p>
                            </div>
                            <div className="mb-4">
                                <span className="text-[10px] tracking-widest">DOMAIN</span>
                                <p className="label-tag inline-block mt-1">
                                    {currentPolicy.domain?.toUpperCase()}
                                </p>
                            </div>
                            <div>
                                <span className="text-[10px] tracking-widest">TRANSFORMS</span>
                                <div className="space-y-2 mt-2">
                                    {currentPolicy.transforms?.map((t, i) => (
                                        <div key={i} className="bg-black text-white p-2 text-xs">
                                            <span className="font-bold">{t.name}</span>
                                            <span className="opacity-50 ml-2">p={t.probability}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ) : (
                        <p className="text-sm opacity-50 text-center py-8">
                            Start chatting to generate a policy
                        </p>
                    )}

                    {currentPolicy && (
                        <button className="pixel-btn w-full mt-4 text-[10px]">
                            EXPORT POLICY
                        </button>
                    )}
                </motion.div>
            </div>
        </div>
    )
}

export default Chat
