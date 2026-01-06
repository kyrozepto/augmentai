import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: 'easeOut' } }
}

// Transform preview modal component
function TransformPreviewModal({ transform, onClose }) {
    const [preview, setPreview] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    useEffect(() => {
        fetchPreview()
    }, [transform])

    const fetchPreview = async () => {
        setLoading(true)
        setError(null)
        try {
            const res = await fetch('/api/policies/preview-transform', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    transform_name: transform.name,
                    probability: 1.0,  // Always apply for preview
                    parameters: transform.parameters || {},
                }),
            })
            if (res.ok) {
                setPreview(await res.json())
            } else {
                const err = await res.json()
                setError(err.detail || 'Failed to generate preview')
            }
        } catch (err) {
            setError('Failed to connect to server')
        }
        setLoading(false)
    }

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4"
            onClick={onClose}
        >
            <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.9, opacity: 0 }}
                className="card-window max-w-3xl w-full"
                onClick={e => e.stopPropagation()}
            >
                <div className="card-header">
                    <div className="window-dots">
                        <motion.div className="window-dot red" whileHover={{ scale: 1.2 }} onClick={onClose} />
                        <div className="window-dot yellow" />
                        <div className="window-dot green" />
                    </div>
                    <span className="text-xs text-white/50">
                        preview://{transform.name}
                    </span>
                </div>

                <div className="p-6">
                    <h2 className="text-lg font-bold text-white mb-1">{transform.name}</h2>
                    <p className="text-sm text-white/40 mb-6">Transform preview with sample image</p>

                    {loading ? (
                        <div className="text-center py-12">
                            <motion.div
                                className="w-12 h-12 mx-auto mb-4 text-white/30"
                                animate={{ rotate: 360 }}
                                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                            >
                                <svg viewBox="0 0 24 24" className="w-full h-full">
                                    <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="1" strokeDasharray="20 40" />
                                </svg>
                            </motion.div>
                            <p className="text-sm text-white/30">Generating preview...</p>
                        </div>
                    ) : error ? (
                        <div className="text-center py-12 border border-red-400/30 bg-red-400/10">
                            <p className="text-sm text-red-400/80">{error}</p>
                            <button
                                onClick={fetchPreview}
                                className="pixel-btn pixel-btn-secondary pixel-btn-sm mt-4"
                            >
                                RETRY
                            </button>
                        </div>
                    ) : preview && (
                        <div className="grid grid-cols-2 gap-6">
                            <div>
                                <h3 className="text-[10px] text-white/40 tracking-wider mb-2">BEFORE</h3>
                                <div className="border border-white/20 aspect-square overflow-hidden">
                                    <img
                                        src={preview.before}
                                        alt="Before transform"
                                        className="w-full h-full object-contain bg-black"
                                    />
                                </div>
                            </div>
                            <div>
                                <h3 className="text-[10px] text-white/40 tracking-wider mb-2">AFTER</h3>
                                <div className="border border-white/20 aspect-square overflow-hidden">
                                    <img
                                        src={preview.after}
                                        alt="After transform"
                                        className="w-full h-full object-contain bg-black"
                                    />
                                </div>
                            </div>
                        </div>
                    )}

                    <div className="flex justify-between items-center mt-6 pt-4 border-t border-white/10">
                        <button
                            onClick={fetchPreview}
                            className="pixel-btn pixel-btn-secondary pixel-btn-sm"
                        >
                            ↻ REGENERATE
                        </button>
                        <button
                            onClick={onClose}
                            className="pixel-btn pixel-btn-sm"
                        >
                            CLOSE
                        </button>
                    </div>
                </div>
            </motion.div>
        </motion.div>
    )
}

function PolicyBuilder() {
    const [domains, setDomains] = useState([])
    const [selectedDomain, setSelectedDomain] = useState('natural')
    const [transforms, setTransforms] = useState([])
    const [pipeline, setPipeline] = useState([])
    const [policyName, setPolicyName] = useState('my_policy')
    const [activeTab, setActiveTab] = useState(0)
    const [previewTransform, setPreviewTransform] = useState(null)

    const mockTabs = ['TRANSFORMS', 'PREVIEW', 'EXPORT']

    useEffect(() => {
        fetch('/api/domains')
            .then(res => res.json())
            .then(data => setDomains(data))
            .catch(console.error)
    }, [])

    useEffect(() => {
        if (selectedDomain) {
            fetch(`/api/domains/${selectedDomain}/transforms`)
                .then(res => res.json())
                .then(data => setTransforms(data))
                .catch(console.error)
        }
    }, [selectedDomain])

    const addTransform = (transform) => {
        setPipeline([...pipeline, {
            ...transform,
            id: Date.now(),
            probability: 0.5,
            parameters: transform.default_parameters || {},
        }])
    }

    const removeTransform = (id) => {
        setPipeline(pipeline.filter(t => t.id !== id))
    }

    const updateProbability = (id, prob) => {
        setPipeline(pipeline.map(t =>
            t.id === id ? { ...t, probability: prob } : t
        ))
    }

    const exportPolicy = async (format) => {
        const policy = {
            name: policyName,
            domain: selectedDomain,
            transforms: pipeline.map(t => ({
                name: t.name,
                probability: t.probability,
                parameters: t.parameters || {},
            })),
        }

        const res = await fetch('/api/policies', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(policy),
        })

        if (res.ok) {
            const data = await res.json()
            const exportRes = await fetch(`/api/policies/${data.id}/export?format=${format}`)
            const exportData = await exportRes.json()

            const blob = new Blob([exportData.content], { type: 'text/plain' })
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `${policyName}.${format === 'python' ? 'py' : format}`
            a.click()
        }
    }

    return (
        <div className="max-w-6xl mx-auto">
            {/* Transform Preview Modal */}
            <AnimatePresence>
                {previewTransform && (
                    <TransformPreviewModal
                        transform={previewTransform}
                        onClose={() => setPreviewTransform(null)}
                    />
                )}
            </AnimatePresence>

            {/* Header */}
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-8 flex justify-between items-start"
            >
                <div>
                    <motion.div
                        className="inline-block mb-3"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.2 }}
                    >
                        <span className="text-[10px] tracking-[0.2em] text-white/40 border border-white/20 px-3 py-1">
                            BUILDER_MODULE
                        </span>
                    </motion.div>
                    <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white mb-2">
                        POLICY :: BUILDER
                    </h1>
                    <p className="text-sm text-white/50">
                        Design domain-safe augmentation pipelines with visual controls
                    </p>
                </div>
                <div className="flex gap-2">
                    <motion.button
                        onClick={() => exportPolicy('yaml')}
                        whileHover={{ scale: 1.02, y: -2 }}
                        whileTap={{ scale: 0.98 }}
                        className="pixel-btn pixel-btn-secondary pixel-btn-sm"
                    >
                        EXPORT_YAML
                    </motion.button>
                    <motion.button
                        onClick={() => exportPolicy('python')}
                        whileHover={{ scale: 1.02, y: -2 }}
                        whileTap={{ scale: 0.98 }}
                        className="pixel-btn pixel-btn-sm"
                    >
                        EXPORT_PYTHON
                    </motion.button>
                </div>
            </motion.div>

            {/* Main Window */}
            <motion.div
                initial={{ opacity: 0, y: 40, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.6, ease: 'easeOut' }}
                className="card-window"
            >
                {/* Window Title Bar */}
                <div className="card-header">
                    <div className="flex items-center gap-3">
                        <div className="window-dots">
                            <motion.div className="window-dot red" whileHover={{ scale: 1.2 }} />
                            <motion.div className="window-dot yellow" whileHover={{ scale: 1.2 }} />
                            <motion.div className="window-dot green" whileHover={{ scale: 1.2 }} />
                        </div>
                        <span className="text-xs text-white/50 ml-2">
                            augmentai://policy/{policyName}
                        </span>
                    </div>
                    <span className="text-[10px] text-white/30">
                        domain: {selectedDomain}
                    </span>
                </div>

                {/* Tab Bar */}
                <div className="tab-bar">
                    {mockTabs.map((tab, i) => (
                        <motion.button
                            key={tab}
                            onClick={() => setActiveTab(i)}
                            className={`tab-item ${activeTab === i ? 'active' : ''}`}
                            whileHover={{ y: -2 }}
                            whileTap={{ y: 0 }}
                        >
                            {tab}
                        </motion.button>
                    ))}
                </div>

                {/* Content Area */}
                <div className="p-6 min-h-[400px] relative">
                    {/* Grid Background */}
                    <div className="absolute inset-0 grid-bg-dense opacity-50" />

                    <div className="relative grid md:grid-cols-3 gap-6">
                        {/* Sidebar - Domain & Transforms */}
                        <div className="space-y-6">
                            <div>
                                <div className="text-[10px] text-white/30 mb-3 tracking-wider">
                                    // SELECT_DOMAIN
                                </div>
                                <select
                                    value={selectedDomain}
                                    onChange={(e) => setSelectedDomain(e.target.value)}
                                    className="w-full"
                                >
                                    {domains.map(d => (
                                        <option key={d.id} value={d.id}>{d.name}</option>
                                    ))}
                                </select>
                            </div>

                            <div>
                                <div className="text-[10px] text-white/30 mb-3 tracking-wider">
                                    // AVAILABLE_TRANSFORMS
                                </div>
                                <p className="text-[9px] text-white/20 mb-2">
                                    Click to add • Right-click to preview
                                </p>
                                <div className="space-y-2 max-h-64 overflow-y-auto">
                                    {transforms.map((t, i) => (
                                        <motion.div
                                            key={t.name}
                                            initial={{ opacity: 0, x: -20 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: i * 0.03 }}
                                            onClick={() => addTransform(t)}
                                            onContextMenu={(e) => {
                                                e.preventDefault()
                                                setPreviewTransform(t)
                                            }}
                                            className="transform-item group"
                                        >
                                            <div className="indicator" />
                                            <span className="flex-1">{t.name}</span>
                                            <motion.span
                                                className="text-[9px] text-white/30 opacity-0 group-hover:opacity-100"
                                                whileHover={{ scale: 1.1 }}
                                                onClick={(e) => {
                                                    e.stopPropagation()
                                                    setPreviewTransform(t)
                                                }}
                                            >
                                                👁
                                            </motion.span>
                                        </motion.div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        {/* Main Panel - Pipeline */}
                        <div className="md:col-span-2">
                            <div className="preview-panel h-full">
                                <div className="preview-panel-header">
                                    <span className="text-xs text-white/40">PIPELINE_BUILDER</span>
                                    <input
                                        type="text"
                                        value={policyName}
                                        onChange={(e) => setPolicyName(e.target.value)}
                                        className="bg-transparent border-b border-white/30 px-2 py-1 text-sm w-40 text-right"
                                        placeholder="policy_name"
                                    />
                                </div>

                                {pipeline.length === 0 ? (
                                    <div className="text-center py-16">
                                        <motion.div
                                            className="w-16 h-16 mx-auto mb-4 text-white/15"
                                            animate={{ rotate: [0, 5, -5, 0] }}
                                            transition={{ duration: 4, repeat: Infinity }}
                                        >
                                            <svg viewBox="0 0 24 24" className="w-full h-full">
                                                <path d="M12 2 L12 22 M2 12 L22 12" stroke="currentColor" strokeWidth="1" />
                                                <circle cx="12" cy="12" r="3" fill="none" stroke="currentColor" strokeWidth="1" />
                                            </svg>
                                        </motion.div>
                                        <p className="text-sm text-white/30">
                                            Click transforms on the left to add them
                                        </p>
                                        <p className="text-xs text-white/20 mt-1">
                                            Right-click any transform to see preview
                                        </p>
                                    </div>
                                ) : (
                                    <div className="space-y-3">
                                        {pipeline.map((t, i) => (
                                            <motion.div
                                                key={t.id}
                                                initial={{ opacity: 0, scale: 0.9, x: -20 }}
                                                animate={{ opacity: 1, scale: 1, x: 0 }}
                                                exit={{ opacity: 0, scale: 0.9 }}
                                                whileHover={{ x: 4, borderColor: 'rgba(255,255,255,0.4)' }}
                                                className="flex items-center gap-4 p-4 border border-white/15 bg-white/5 transition-colors group"
                                            >
                                                <span className="text-[10px] font-bold text-white/30 w-6">
                                                    {String(i + 1).padStart(2, '0')}
                                                </span>
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-sm font-bold text-white">{t.name}</span>
                                                        <motion.button
                                                            onClick={() => setPreviewTransform(t)}
                                                            whileHover={{ scale: 1.1 }}
                                                            className="text-[10px] text-white/30 hover:text-white/60 opacity-0 group-hover:opacity-100 transition-opacity"
                                                        >
                                                            👁 PREVIEW
                                                        </motion.button>
                                                    </div>
                                                    <div className="param-slider mt-2">
                                                        <span className="label">probability</span>
                                                        <input
                                                            type="range"
                                                            min="0"
                                                            max="1"
                                                            step="0.05"
                                                            value={t.probability}
                                                            onChange={(e) => updateProbability(t.id, parseFloat(e.target.value))}
                                                            className="flex-1"
                                                        />
                                                        <span className="value">{t.probability.toFixed(2)}</span>
                                                    </div>
                                                </div>
                                                <motion.button
                                                    onClick={() => removeTransform(t.id)}
                                                    whileHover={{ scale: 1.2 }}
                                                    className="text-red-400/60 hover:text-red-400 text-sm"
                                                >
                                                    ✕
                                                </motion.button>
                                            </motion.div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Status Bar */}
                <div className="status-bar">
                    <div className="flex items-center gap-4">
                        <span className="flex items-center gap-2 text-green-400/80">
                            <span className="status-dot online" />
                            READY
                        </span>
                        <span className="text-white/30">
                            {pipeline.length} transform{pipeline.length !== 1 ? 's' : ''} in pipeline
                        </span>
                    </div>
                    <span className="text-white/30">v0.1.2</span>
                </div>
            </motion.div>
        </div>
    )
}

export default PolicyBuilder
